"""
Refactored Data Loader - Now domain-agnostic.
"""

import numpy as np
from typing import Tuple
from pathlib import Path


class DataLoader:
    """Generic loader for tensor data files."""
    
    def __init__(self, data_dir: str, domain):
        """Initialize with data directory and domain strategy.
        
        Args:
            data_dir: Path to data directory (relative or absolute)
            domain: Domain strategy instance (e.g., HPCDomain)
        """
        self.data_dir = Path(data_dir)
        self.domain = domain
        self._original_data = None
        self._time_axis = None
        self._tensor_X = None
        self._tensor_y = None
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all data files and return them."""
        # Generic file names (domain-independent)
        prefix = f"{self.domain.name}_"
        
        self._original_data = np.load(self.data_dir / f"{prefix}time_original.npy")
        self._time_axis = np.load(self.data_dir / f"{prefix}time_axis.npy")
        self._tensor_X = np.load(self.data_dir / f"{prefix}tensor_X.npy")
        self._tensor_y = np.load(self.data_dir / f"{prefix}tensor_y.npy")
        
        return self._original_data, self._time_axis, self._tensor_X, self._tensor_y
    
    @property
    def original_data(self) -> np.ndarray:
        if self._original_data is None:
            self.load_all()
        return self._original_data
    
    @property
    def time_axis(self) -> np.ndarray:
        if self._time_axis is None:
            self.load_all()
        return self._time_axis
    
    @property
    def tensor_X(self) -> np.ndarray:
        if self._tensor_X is None:
            self.load_all()
        return self._tensor_X
    
    @property
    def tensor_y(self) -> np.ndarray:
        if self._tensor_y is None:
            self.load_all()
        return self._tensor_y
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (T, S, V) shape of tensor."""
        return self.tensor_X.shape
    
    @property
    def n_classes(self) -> int:
        """Return number of unique classes."""
        return len(np.unique(self.tensor_y))
