import threading
import time
import numpy as np
import itertools
import torch
import os
import fnmatch
import queue
import json
import warnings
import importlib
import torch.distributed as dist
from typing import Optional, Any
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from .dataset_wrappers import WrappedDataset, WrappedIterableDataset, MapToIterableDataset
try:
    from torchdata.datapipes.iter import IterableWrapper, Cycler, ShardingFilter, Shuffler, Batcher, Prefetcher, Multiplexer, SampleMultiplexer
    TORCHDATA_AVAILABLE = True
except ImportError:
    TORCHDATA_AVAILABLE = False
    warnings.warn("torchdata not available. Multi-dataset mixing will not be supported.")

def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

class BackgroundPrefetcher:
    """
    Fetches Epoch N result blockingly from the queue, 
    then starts prefetching Epoch N+1 in the background. Relies solely on 
    the queue for state transfer between threads.
    """
    def __init__(self, loader: DataLoader):
        internal_len = -1
        try:
            internal_len = len(loader)
            if internal_len <= 0:
                raise ValueError("DataLoader length must be positive.")
        except TypeError:
            raise TypeError("BackgroundPrefetcher requires the wrapped DataLoader to have a __len__ method.")
        self.loader = loader
        self.epoch = 0 # Tracks the epoch we *expect* Trainer to ask for next
        self.prefetch_thread: Optional[threading.Thread] = None
        
        # Thread-safe queue to store prefetched results: (iterator, first_batch, epoch_fetched) or Exception or StopIteration
        self.data_queue = queue.Queue(maxsize=1) 
        self._stop_event = threading.Event()

        # --- Initial blocking prefetch for Epoch 0 ---
        start_time = time.time()
        # Run prefetch job directly in the main thread for the first epoch
        # This will put Epoch 0's result into self.data_queue
        self._prefetch_job(self.epoch) 
        # Epoch 0 data is now in the queue, ready for the first __iter__ call.

    def _prefetch_job(self, epoch_to_fetch: int):
        """
        Runs in main or background thread.
        Sets sampler epoch, creates iterator, fetches first batch, puts result in queue.
        """
        # Check if shutdown requested before starting work
        if self._stop_event.is_set():
            # Ensure queue doesn't block indefinitely if shutdown happens during prefetch
            # Use a unique signal or rely on timeout in get() if necessary
            # For simplicity, putting StopIteration might work if handled correctly downstream
            self.data_queue.put(StopIteration) 
            return
             
        try:
            # 1. Set sampler epoch
            if isinstance(self.loader.sampler, Sampler) and hasattr(self.loader.sampler, "set_epoch"):
                self.loader.sampler.set_epoch(epoch_to_fetch)
            # 2. Create iterator and fetch first batch (the cold start)
            epoch_iter = iter(self.loader) # <-- Creates the iterator
            first_batch = next(epoch_iter) # <-- Fetches the first batch (blocking)
            # 3. Put successful result in queue
            self.data_queue.put((epoch_iter, first_batch, epoch_to_fetch))
            
        except StopIteration:
            self.data_queue.put(StopIteration) # Use StopIteration as signal
        except Exception as e:
            self.data_queue.put(e) # Pass the exception through the queue

    def _start_background_prefetch(self, epoch_to_fetch: int):
        """ Safely starts the background prefetch thread """
        # Ensure previous thread (if any) is finished before starting a new one
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join()
             
        # Check stop event before starting new thread
        if self._stop_event.is_set():
            return

        self.prefetch_thread = threading.Thread(
            target=self._prefetch_job, 
            args=(epoch_to_fetch,),
            daemon=True
        )
        self.prefetch_thread.start()

    def __iter__(self):
        """ Trainer calls this at the start of each epoch (e.g., Epoch N) """
        # 1. Block and get the result for the CURRENT epoch (Epoch N)
        # This result was put into the queue by the *previous* call's background thread (or by __init__ for Epoch 0)
        try:
            # Use timeout to prevent indefinite blocking if something goes wrong
            data = self.data_queue.get(timeout=600) # Timeout after 10 minutes
        except queue.Empty:
            raise TimeoutError(f"Timeout waiting for prefetched data for Epoch {self.epoch}. Background thread might be stuck.")
        
        # Check for error signals from the prefetch job
        if isinstance(data, Exception): 
            raise data
        if data is StopIteration:
            # If we received StopIteration, the dataloader is exhausted for this epoch
            # Signal this to the Trainer by raising StopIteration from __iter__
            raise StopIteration 
            
        # Unpack successful prefetch result
        current_iter, first_batch, fetched_epoch = data

        # 2. Start prefetching the *NEXT* epoch (Epoch N+1) in the background *NOW*
        # This allows maximum overlap with the upcoming GPU work for Epoch N
        next_epoch = self.epoch + 1
        self._start_background_prefetch(next_epoch)
        # 3. Define the generator for the CURRENT epoch (Epoch N)
        def generator(first, iterator, epoch_num):
            try:
                yield first # Yield the already fetched first batch
                
                # Yield the remaining batches from the iterator
                batch_count = 1 # Already yielded one batch
                for batch in iterator:
                    yield batch
                    batch_count += 1
                    
            
            except Exception as e:
                 # Ensure the exception is propagated up to the Trainer
                raise e
            finally:
                # Cleanup specific to this generator instance, if any
                pass 

        # Increment internal epoch counter *after* starting the next prefetch
        # and *before* returning the generator for the current epoch
        self.epoch += 1 
        return generator(first_batch, current_iter, fetched_epoch)

    def __len__(self):
        # Trainer needs this to calculate steps_in_epoch
        # It relies on the length of the underlying loader
        return len(self.loader) 
    
    @property
    def sampler(self):
        # Trainer needs access to the sampler, primarily for set_epoch
        return self.loader.sampler

    def set_epoch(self, epoch: int):
        # This is called by Trainer *before* __iter__. 
        # Our internal prefetch logic uses its own epoch counter and handles setting the sampler epoch.
        # This method just needs to exist for compatibility.
        # We could add a check here: if epoch != self.epoch: warn, but Trainer might call set_epoch(0) multiple times initially.
        pass

    def shutdown(self):
        """ Gracefully stop the background thread and clear queue """
        self._stop_event.set()
        
        # Try to unblock the queue if the background thread is stuck waiting to put
        try:
            self.data_queue.put_nowait(StopIteration) 
        except queue.Full:
            pass # Queue was already full or another signal was put

        # Wait for the background thread to finish
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=10) # Add timeout
        # Clear the queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
      
def get_dataloader(train_dataset, val_dataset=None, processor=None, collator=None, args=None):
    """
    Create DataLoader from single dataset or multiple datasets.
    
    Args:
        train_dataset: Single dataset or list of datasets
        val_dataset: Optional validation dataset
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
    
    Returns:
        (train_loader, eval_loader)
    """
    if isinstance(train_dataset, list):
        # Multiple datasets - use torchdata for mixing
        
        print(f"Using mixed dataset pipeline with {len(train_dataset)} datasets")
        train_loader = _create_mixed_dataloader(train_dataset, processor, collator, args)
        
        # Handle validation dataset
        eval_loader = None
        if val_dataset is not None:
            if isinstance(val_dataset, list):
                eval_loader = _create_mixed_dataloader(val_dataset, processor, collator, args, is_training=False)
            else:
                eval_loader = _create_single_dataloader(val_dataset, processor, collator, args, is_training=False)
        
        return train_loader, eval_loader
    
    else:
        # Single dataset - use existing logic
        return _create_single_dataloader(train_dataset, processor, collator, args, is_training=True)

def is_rlds_data(ds):
    try:
        from .datasets.rlds_wrapper import WrappedRLDSDataset
        from .datasets.droid import DroidDataset
        from .datasets.vlaos import VLAOSDataset
    except:
        return False
    return isinstance(ds, WrappedRLDSDataset) or isinstance(ds, DroidDataset) or isinstance(ds, VLAOSDataset)

def is_map_data(dataset):
    return hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')

def is_iter_data(dataset):
    return hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))

def _create_single_dataloader(dataset, processor, collator, args, is_training=True):
    """Create DataLoader for a single dataset (map-style or iterable)"""
    # Identify the type of the dataset: iter or map
    if is_map_data(dataset):
        wrapped_data = WrappedDataset(dataset, processor)
        sampler = DistributedSampler(wrapped_data) if is_training and is_distributed() else None
        loader = DataLoader(
            wrapped_data,
            batch_size=args.per_device_train_batch_size,
            shuffle=(sampler is None and is_training),
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
            drop_last=is_training,
            pin_memory=args.dataloader_pin_memory,
            persistent_workers=False,
            prefetch_factor=2,
        )
        loader = BackgroundPrefetcher(loader)
    elif is_iter_data(dataset):
        if is_rlds_data(dataset.dataset):
            # RLDS dataset
            wrapped_data = WrappedIterableDataset(dataset, processor)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            loader = DataLoader(
                wrapped_data,  
                batch_size=batch_size,
                num_workers=0,
                collate_fn=collator,
                drop_last=True,
            )
        else:
            # Pytorch Iterable dataset
            # Wrap iterable dataset with processor
            wrapped_data = WrappedIterableDataset(dataset, processor)
            pipe = IterableWrapper(wrapped_data, deepcopy=False)
            # For iterable datasets, we cannot use DistributedSampler
            pipe = pipe.sharding_filter()
            pipe = pipe.cycle()
            buffer_size = getattr(args, 'shuffle_buffer_size', 1000)
            pipe = pipe.shuffle(buffer_size=buffer_size)
            batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
            #### Prefetch is handled by DataLoader
            loader = DataLoader(
                pipe,
                batch_size=batch_size,  
                # num_workers=args.dataloader_num_workers,
                collate_fn=collator, 
                pin_memory=args.dataloader_pin_memory,
                # persistent_workers=args.dataloader_num_workers>0,
                drop_last=True,
            )
    else:
        raise ValueError("Dataset must be either map-style or iterable.")
    return loader, None if is_training else loader

def _create_mixed_dataloader(datasets, processor, collator, args, is_training=True):
    """
    Create DataLoader for multiple datasets using torchdata pipeline.
    
    Args:
        datasets: List of datasets (can be map-style or iterable)
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
        is_training: Whether this is for training (affects shuffle, drop_last)
    
    Returns:
        DataLoader with mixed pipeline
    """
    # Create DataLoader for MapStyleDataset, IterableDataset, and RLDSDataset, respectively
    all_map_datasets, all_iter_datasets, all_rlds_datasets = [], [], []
    for data in datasets:
        if is_map_data(data): all_map_datasets.append(data)
        elif is_iter_data(data):
            if is_rlds_data(data):
                all_rlds_datasets.append(data)
            else:
                all_iter_datasets.append(data)
        else:
            raise TypeError("Dataset must be either map-style or iterable.")
    # Mix map-style datasets using ConcatDataset
    if len(all_map_datasets)>0:
        mixed_map_data = torch.utils.data.ConcatDataset(all_map_datasets)
        map_loader = _create_single_dataloader(dataset, processor, collator, args, is_training=is_training)
    else:
        map_loader = None
    # mix iterable datasets using huggingface's datasets
    if len(all_iter_datasets)>0:
        from datasets import interleave_datasets
        mixed_iter_data = interleave_datasets(all_iter_datasets
        )
        iter_loader = _create_single_dataloader(mixed_iter_data, processor, collator, args, is_training=is_training)
    else:
        iter_loader = None
    # mix rlds datasets using tf.data
    if len(all_rlds_datasets)>0:
        import dlimp as dl
        mixed_rlds_data = dl.DLataset.sample_from_datasets([ds.dataset for ds in all_rlds_datasets])
        rlds_loader = _create_single_dataloader(mixed_rlds_data, processor, collator, args, is_training=is_training)
    else:
        rlds_loader = None
    # Combine all loaders using SampleMultiplexer    
    if rlds_loader is None and iter_loader is None: return map_loader
    elif map_loader is None and rlds_loader is None: return iter_loader
    elif map_loader is None and iter_loader is None: return rlds_loader
    else:
        raise NotImplementedError("Mixing map-style, iterable, and RLDS datasets is not yet implemented.")
    
