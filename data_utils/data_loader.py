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

def _create_single_dataloader(dataset, processor, collator, args, is_training=True):
    """Create DataLoader for a single dataset (map-style or iterable)"""
    # Identify the type of the dataset: iter or map
    is_map_dataset = hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')
    is_iter_dataset = hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))
    
    if is_map_dataset:
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
        if is_training:
            return loader, None
        else:
            return loader
    elif is_iter_dataset:
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
            if is_training:
                return loader, None
            else:
                return loader
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
            if is_training:
                return loader, None
            else:
                return loader
    else:
        raise ValueError("Dataset must be either map-style or iterable.")

def _create_mixed_dataloader(datasets, processor, collator, args, is_training=True):
    """
    Create DataLoader for multiple datasets using torchdata pipeline.
    
    Pipeline structure:
    1. Wrap/Convert each dataset to iterable
    2. Apply processor to each dataset
    3. Wrap as IterableWrapper (torchdata pipeline)
    4. Apply Cycle to prevent exhaustion
    5. Apply ShardingFilter for distributed training
    6. Apply Shuffler for randomization
    7. Apply Batcher (if collator expects unbatched data)
    8. Apply Prefetcher for performance
    9. Create DataLoader
    
    Args:
        datasets: List of datasets (can be map-style or iterable)
        processor: Function to process each sample
        collator: Function to collate samples into batches
        args: Training arguments
        is_training: Whether this is for training (affects shuffle, drop_last)
    
    Returns:
        DataLoader with mixed pipeline
    """
    print(f"Building mixed dataset pipeline:")
    
    # Step 1: Convert all datasets to iterable with processor applied
    iterable_datasets = []
    for i, dataset in enumerate(datasets):
        is_iter = hasattr(dataset, '__iter__') and (not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'))
        
        if is_iter:
            # Already iterable
            print(f"  Dataset {i}: Iterable dataset")
            wrapped = WrappedIterableDataset(dataset, processor)
        else:
            # Map-style: convert to iterable
            print(f"  Dataset {i}: Map-style dataset (size={len(dataset)}) -> converting to iterable")
            # First apply processor via WrappedDataset
            wrapped_map = WrappedDataset(dataset, processor)
            # Then convert to iterable
            wrapped = MapToIterableDataset(wrapped_map, shuffle=is_training, seed=args.seed if hasattr(args, 'seed') else None)
        
        iterable_datasets.append(wrapped)
    
    # Step 2: Create torchdata pipelines for each dataset
    print(f"Creating torchdata pipelines:")
    pipelines = []
    for i, dataset in enumerate(iterable_datasets):
        # Wrap as IterableWrapper
        pipe = IterableWrapper(dataset, deepcopy=False)
        
        # Apply Cycle to prevent exhaustion
        pipe = pipe.cycle()
        print(f"  Pipeline {i}: IterableWrapper -> Cycle")
        
        pipelines.append(pipe)
    
    # Step 3: Multiplex (mix) all pipelines
    # Use round-robin sampling from each pipeline
    print(f"Multiplexing {len(pipelines)} pipelines")
    mixed_pipe = Multiplexer(*pipelines)
    
    # Step 4: Apply ShardingFilter for distributed training
    # print(f"Applying ShardingFilter for distributed training")
    mixed_pipe = mixed_pipe.sharding_filter()
    
    # Step 5: Apply Shuffler for randomization (if training)
    # if is_training:
    buffer_size = getattr(args, 'shuffle_buffer_size', 1000)
    # print(f"Applying Shuffler with buffer_size={buffer_size}")
    mixed_pipe = mixed_pipe.shuffle(buffer_size=buffer_size)
    
    # Step 6: Batching is handled by DataLoader's batch_size parameter
    # We don't use Batcher here because collator might need custom logic
    batch_size = args.per_device_train_batch_size if is_training else args.per_device_eval_batch_size
    # mixed_pipe = mixed_pipe.batch(
    #     batch_size=batch_size, 
    #     drop_last=is_training
    # )
    # mixed_pipe = mixed_pipe.collate(collate_fn=collator)
    
    # Step 7: Apply Prefetcher for performance
    prefetch_size = getattr(args, 'prefetch_size', 10)
    print(f"Applying Prefetcher with buffer_size={prefetch_size}")
    mixed_pipe = mixed_pipe.prefetch(buffer_size=prefetch_size)
    
    # Step 8: Create DataLoader
    print(f"Creating DataLoader with batch_size={args.per_device_train_batch_size}")
    loader = DataLoader(
        mixed_pipe,
        batch_size=batch_size,  # **职责划分**: Batching 已在 pipeline 中完成
        num_workers=args.dataloader_num_workers,
        collate_fn=collator, # **职责划分**: Collate 已在 pipeline 中完成
        pin_memory=args.dataloader_pin_memory,
        persistent_workers=True,
    )
    
    return loader
    
