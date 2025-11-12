import threading
from .base import AbstractActionManager


class BasicActionManager(AbstractActionManager):
    """Drop out the previous chunk directly whenever the new one comes"""
    
    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self.current_step = 0
        self._buffer = None
        self._chunk_buffer = None

    def put(self, chunk, timestamp: float = None):
        with self._lock:
            self._chunk_buffer = chunk
            self.current_step = 0

    def get(self, timestamp: float = None):
        with self._lock:
            if self._chunk_buffer is None or self.current_step >= len(self._chunk_buffer):
                return None
            action = self._chunk_buffer[self.current_step]
            self.current_step += 1
            return action

