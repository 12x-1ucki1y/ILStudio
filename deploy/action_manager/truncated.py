from .basic import BasicActionManager


class TruncatedManager(BasicActionManager):
    """
    Truncate action chunks by dropping the first and last portions.
    
    This manager discards:
    - First len(chunk) * start_ratio actions (warm-up phase) - NOT applied to the first chunk
    - Last len(chunk) * end_ratio actions (uncertain predictions)
    
    Special behavior: The first chunk keeps all beginning actions (start_ratio is ignored)
    to ensure smooth startup. Subsequent chunks apply both start_ratio and end_ratio.
    
    Then executes the remaining middle portion like OlderFirstManager,
    refusing new chunks until the current chunk is sufficiently executed.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.start_ratio = getattr(config, 'start_ratio', 0.0)
        self.end_ratio = getattr(config, 'end_ratio', 0.0)
        self.older_coef = getattr(config, 'older_coef', 1.0)
        
        print(f"TruncatedManager initialized:")
        print(f"  - start_ratio: {self.start_ratio} (drop first {self.start_ratio*100:.1f}%)")
        print(f"  - end_ratio: {self.end_ratio} (drop last {self.end_ratio*100:.1f}%)")
        print(f"  - older_coef: {self.older_coef} (accept new chunk after {self.older_coef*100:.1f}% executed)")

    def put(self, chunk, timestamp: float = None):
        # Truncate the chunk first
        if chunk is not None and len(chunk) > 0:
            total_len = len(chunk)
            
            # For the first chunk, don't drop the beginning (start_idx = 0)
            # For subsequent chunks, apply start_ratio normally
            is_first_chunk = (self._chunk_buffer is None)
            start_idx = 0 if is_first_chunk else int(total_len * self.start_ratio)
            end_idx = total_len - int(total_len * self.end_ratio)
            
            # Ensure we have at least one action
            if end_idx <= start_idx:
                # If truncation would remove everything, keep at least the middle action
                mid_idx = total_len // 2
                chunk = [chunk[mid_idx]]
                print(f"[TruncatedManager] Warning: Truncation too aggressive, keeping only middle action")
            else:
                chunk = chunk[start_idx:end_idx]
                if start_idx > 0 or end_idx < total_len:
                    chunk_type = "first chunk" if is_first_chunk else "chunk"
                    print(f"[TruncatedManager] Truncated {chunk_type}: {total_len} -> {len(chunk)} actions (kept [{start_idx}:{end_idx}])")
        
        # Now apply OlderFirst logic
        if self._chunk_buffer is None:
            # First chunk, accept directly
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
        else:
            # Check if current chunk is sufficiently executed
            with self._lock:
                if self.current_step < int(len(self._chunk_buffer) * self.older_coef):
                    # Current chunk not done enough, refuse new chunk
                    print(f"[TruncatedManager] Refusing new chunk: {self.current_step}/{len(self._chunk_buffer)} executed ({self.current_step/len(self._chunk_buffer)*100:.1f}% < {self.older_coef*100:.1f}%)")
                    return
            
            # Accept new chunk
            with self._lock:
                self._chunk_buffer = chunk
                self.current_step = 0
                print(f"[TruncatedManager] Accepted new chunk: {len(chunk)} actions")

