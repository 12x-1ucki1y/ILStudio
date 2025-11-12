import numpy as np
from abc import ABC, abstractmethod


class AbstractActionManager(ABC):
    """Abstract base class for action managers."""
    
    @abstractmethod
    def put(self, chunk: np.ndarray, timestamp: float = None):
        """Put action chunk into local cache"""
        pass

    @abstractmethod
    def get(self, timestamp: float = None):
        """Get one-step action from local cache"""
        pass

