import time
from .basic import BasicActionManager


class DelayFreeManager(BasicActionManager):
    """Remove the outdated actions from each chunk"""
    
    def __init__(self, config):
        super().__init__(config)
        self.duration = getattr(config, 'duration', 0.05)

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None: 
            super().put(chunk, timestamp)
            return
        else:
            delay_time = time.perf_counter() - timestamp
            delayed_start_idx = int(delay_time // self.duration)

            if delayed_start_idx < len(chunk):
                chunk = chunk[delayed_start_idx:]
                with self._lock:
                    self._chunk_buffer = chunk
                    self.current_step = 0

