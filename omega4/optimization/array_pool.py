"""
Array pooling for performance optimization
Reduces memory allocation overhead by reusing numpy arrays
"""

import numpy as np
import threading
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ArrayPool:
    """Thread-safe array pool for reusing numpy arrays"""
    
    def __init__(self, max_arrays_per_shape: int = 10):
        """
        Initialize array pool
        
        Args:
            max_arrays_per_shape: Maximum arrays to keep per shape/dtype combination
        """
        self.pools: Dict[Tuple[Tuple[int, ...], np.dtype], List[np.ndarray]] = {}
        self.max_arrays_per_shape = max_arrays_per_shape
        self.lock = threading.Lock()
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'returns': 0,
            'pool_misses': 0
        }
        
    def get_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Get an array from the pool or allocate a new one
        
        Args:
            shape: Shape of the array
            dtype: Data type of the array
            
        Returns:
            numpy array of requested shape and dtype
        """
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                # Reuse existing array
                array = self.pools[key].pop()
                self.stats['reuses'] += 1
                # Clear array to avoid data leakage
                array.fill(0)
                return array
            else:
                # Allocate new array
                self.stats['allocations'] += 1
                self.stats['pool_misses'] += 1
                return np.zeros(shape, dtype=dtype)
    
    def return_array(self, array: np.ndarray):
        """
        Return an array to the pool for reuse
        
        Args:
            array: Array to return to pool
        """
        if array is None:
            return
            
        key = (array.shape, array.dtype)
        
        with self.lock:
            if key not in self.pools:
                self.pools[key] = []
            
            # Only keep up to max_arrays_per_shape
            if len(self.pools[key]) < self.max_arrays_per_shape:
                self.pools[key].append(array)
                self.stats['returns'] += 1
    
    def clear(self):
        """Clear all arrays from the pool"""
        with self.lock:
            self.pools.clear()
            logger.info(f"Array pool cleared. Stats: {self.stats}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self.lock:
            stats = self.stats.copy()
            stats['pool_size'] = sum(len(arrays) for arrays in self.pools.values())
            stats['unique_shapes'] = len(self.pools)
            return stats
    
    def get_or_zeros(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Convenience method that always returns a zeroed array"""
        return self.get_array(shape, dtype)
    
    def get_or_empty(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Get array without clearing (faster but unsafe)"""
        key = (shape, dtype)
        
        with self.lock:
            if key in self.pools and self.pools[key]:
                array = self.pools[key].pop()
                self.stats['reuses'] += 1
                return array
            else:
                self.stats['allocations'] += 1
                self.stats['pool_misses'] += 1
                return np.empty(shape, dtype=dtype)


class ScopedArray:
    """Context manager for automatic array return to pool"""
    
    def __init__(self, pool: ArrayPool, shape: Tuple[int, ...], dtype: np.dtype = np.float32):
        self.pool = pool
        self.shape = shape
        self.dtype = dtype
        self.array = None
        
    def __enter__(self) -> np.ndarray:
        self.array = self.pool.get_array(self.shape, self.dtype)
        return self.array
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.array is not None:
            self.pool.return_array(self.array)


# Global array pool instance
_global_pool = ArrayPool()


def get_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    """Get array from global pool"""
    return _global_pool.get_array(shape, dtype)


def return_array(array: np.ndarray):
    """Return array to global pool"""
    _global_pool.return_array(array)


def clear_pool():
    """Clear global pool"""
    _global_pool.clear()


def get_pool_stats() -> Dict[str, int]:
    """Get global pool statistics"""
    return _global_pool.get_stats()


def with_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> ScopedArray:
    """Get a scoped array that automatically returns to pool"""
    return ScopedArray(_global_pool, shape, dtype)