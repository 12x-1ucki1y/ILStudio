from .basic import BasicActionManager


class OlderFirstManager(BasicActionManager):
    """Refuse newly coming chunks unless the last chunk ends x%"""
    
    def __init__(self, config):
        super().__init__(config)
        self.coef = getattr(config, 'coef', 1.0)
        print(f"OlderFirstManager initialized with coef {self.coef}")

    def put(self, chunk, timestamp: float = None):
        if self._chunk_buffer is None:
            super().put(chunk, timestamp)
        else:
            with self._lock:
                if self.current_step < int(len(self._chunk_buffer) * self.coef):
                    return
            super().put(chunk, timestamp)

