"""
Memory Pool System for OMEGA-4 Audio Analyzer
Pre-allocated array management for reduced memory allocation overhead
"""

import numpy as np
import threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MemoryPool:
    """Thread-safe memory pool for pre-allocated arrays"""
    
    def __init__(self, common_sizes: List[int] = None):
        """Initialize memory pool with common buffer sizes
        
        Args:
            common_sizes: List of common array sizes to pre-allocate
                         Default: [512, 1024, 2048, 4096]
        """
        self.common_sizes = common_sizes or [512, 1024, 2048, 4096]
        
        # Pool storage: size -> list of available arrays
        self.pools: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # In-use tracking: array id -> size (for returning)
        self.in_use: Dict[int, int] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'misses': 0,
            'returns': 0
        }
        
        # Pre-allocate common sizes
        self._preallocate()
    
    def _preallocate(self):
        """Pre-allocate arrays for common sizes"""
        with self.lock:
            for size in self.common_sizes:
                # Pre-allocate 2 arrays of each common size
                for _ in range(2):
                    arr = np.zeros(size, dtype=np.float32)
                    self.pools[size].append(arr)
                    self.stats['allocations'] += 1
                    
            logger.debug(f"Pre-allocated {len(self.common_sizes) * 2} arrays")
    
    def get_array(self, size: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """Get an array from the pool or allocate a new one
        
        Args:
            size: Required array size
            dtype: Array data type (default: float32)
            
        Returns:
            Zero-initialized numpy array
        """
        with self.lock:
            # Check if we have an available array of this size
            if size in self.pools and self.pools[size]:
                # Reuse existing array
                arr = self.pools[size].pop()
                self.in_use[id(arr)] = size
                self.stats['reuses'] += 1
                
                # Ensure array is zeroed
                arr.fill(0)
                return arr
            
            # No available array, allocate new one
            arr = np.zeros(size, dtype=dtype)
            self.in_use[id(arr)] = size
            self.stats['allocations'] += 1
            self.stats['misses'] += 1
            
            return arr
    
    def return_array(self, arr: np.ndarray):
        """Return an array to the pool for reuse
        
        Args:
            arr: Array to return to pool
        """
        arr_id = id(arr)
        
        with self.lock:
            # Check if this array is tracked
            if arr_id not in self.in_use:
                logger.warning("Attempting to return untracked array")
                return
            
            # Get the size and return to appropriate pool
            size = self.in_use.pop(arr_id)
            self.pools[size].append(arr)
            self.stats['returns'] += 1
    
    def get_window(self, size: int, window_type: str = 'hann') -> np.ndarray:
        """Get a windowed array (e.g., Hanning window)
        
        Args:
            size: Window size
            window_type: Type of window ('hann', 'hamming', 'blackman')
            
        Returns:
            Windowed array
        """
        # Get base array
        arr = self.get_array(size)
        
        # Apply window
        if window_type == 'hann':
            window = np.hanning(size)
        elif window_type == 'hamming':
            window = np.hamming(size)
        elif window_type == 'blackman':
            window = np.blackman(size)
        else:
            window = np.ones(size)
            
        arr[:] = window
        return arr
    
    def get_fft_arrays(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get paired arrays for FFT operations (input and output)
        
        Args:
            size: Array size
            
        Returns:
            Tuple of (input_array, output_array)
        """
        input_arr = self.get_array(size)
        # FFT output size for real input
        output_size = size // 2 + 1
        output_arr = self.get_array(output_size, dtype=np.complex64)
        
        return input_arr, output_arr
    
    def clear_pool(self, size: Optional[int] = None):
        """Clear arrays from pool to free memory
        
        Args:
            size: Specific size to clear, or None to clear all
        """
        with self.lock:
            if size is not None:
                if size in self.pools:
                    count = len(self.pools[size])
                    self.pools[size].clear()
                    logger.debug(f"Cleared {count} arrays of size {size}")
            else:
                total = sum(len(arrays) for arrays in self.pools.values())
                self.pools.clear()
                logger.debug(f"Cleared {total} arrays from pool")
    
    def get_stats(self) -> Dict[str, any]:
        """Get pool statistics
        
        Returns:
            Dictionary with pool statistics
        """
        with self.lock:
            pool_sizes = {size: len(arrays) for size, arrays in self.pools.items()}
            
            return {
                'allocations': self.stats['allocations'],
                'reuses': self.stats['reuses'],
                'misses': self.stats['misses'],
                'returns': self.stats['returns'],
                'reuse_rate': self.stats['reuses'] / max(1, self.stats['reuses'] + self.stats['misses']),
                'in_use_count': len(self.in_use),
                'pool_sizes': pool_sizes,
                'total_pooled': sum(pool_sizes.values())
            }
    
    def optimize_pool_sizes(self):
        """Analyze usage patterns and adjust pool sizes"""
        with self.lock:
            # Find frequently missed sizes
            # This is a simplified version - in production, track miss counts per size
            
            # Add new common sizes based on usage
            for size, arrays in list(self.pools.items()):
                if len(arrays) == 0 and size not in self.common_sizes:
                    # This size is being used but not pre-allocated
                    self.common_sizes.append(size)
                    # Pre-allocate one array for this size
                    arr = np.zeros(size, dtype=np.float32)
                    self.pools[size].append(arr)
                    self.stats['allocations'] += 1
                    logger.debug(f"Added size {size} to common sizes")


class ScopedArray:
    """Context manager for automatic array return"""
    
    def __init__(self, pool: MemoryPool, size: int, dtype: np.dtype = np.float32):
        self.pool = pool
        self.size = size
        self.dtype = dtype
        self.array = None
    
    def __enter__(self) -> np.ndarray:
        self.array = self.pool.get_array(self.size, self.dtype)
        return self.array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.array is not None:
            self.pool.return_array(self.array)


# Global memory pool instance
_global_pool = None


def get_global_pool() -> MemoryPool:
    """Get or create global memory pool instance"""
    global _global_pool
    if _global_pool is None:
        _global_pool = MemoryPool()
    return _global_pool


def with_pooled_array(size: int, dtype: np.dtype = np.float32) -> ScopedArray:
    """Convenience function to get a scoped array from global pool
    
    Usage:
        with with_pooled_array(1024) as arr:
            # Use arr
            pass
        # arr is automatically returned to pool
    """
    return ScopedArray(get_global_pool(), size, dtype)