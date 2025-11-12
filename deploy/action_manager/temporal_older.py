from .temporal_agg import TemporalAggManager


class TemporalOlderManager(TemporalAggManager):
    """Refuse newly coming chunks until the last chunk ends x%"""
    
    def __init__(self, config):
        super().__init__(config)
        self.older_coef = getattr(config, 'older_coef', 0.75)

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
        else:
            with self._lock:
                if self.current_step < int(len(self._chunk_buffer) * self.older_coef):
                    return
            super().put(chunk, timestamp)

