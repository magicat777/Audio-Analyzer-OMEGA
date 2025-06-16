"""
FFT Result Caching for Performance Optimization
Caches FFT results for static/silent audio to avoid redundant calculations
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, Any
import hashlib
import logging

logger = logging.getLogger(__name__)


class FFTCache:
    """Caches FFT results to avoid redundant calculations on static audio"""
    
    def __init__(self, cache_size: int = 10, ttl_seconds: float = 0.5):
        """
        Initialize FFT cache
        
        Args:
            cache_size: Maximum number of cached results
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = []
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'silent_hits': 0
        }
        
        # Special handling for silence
        self.silence_threshold = 1e-6
        self.silence_cache_key = "_silence_"
        
    def _compute_hash(self, audio_data: np.ndarray) -> str:
        """Compute fast hash of audio data"""
        # Check for silence first
        if np.max(np.abs(audio_data)) < self.silence_threshold:
            return self.silence_cache_key
            
        # For non-silent audio, sample the data for faster hashing
        # Take every 64th sample to speed up hashing
        sampled = audio_data[::64]
        
        # Convert to bytes and hash
        data_bytes = sampled.tobytes()
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Get cached FFT result if available
        
        Args:
            audio_data: Audio chunk to process
            
        Returns:
            Cached FFT result or None if not found/expired
        """
        cache_key = self._compute_hash(audio_data)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            
            # Check if entry is still valid
            if time.time() - timestamp < self.ttl_seconds:
                self.stats['hits'] += 1
                if cache_key == self.silence_cache_key:
                    self.stats['silent_hits'] += 1
                    
                # Update access order
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
                
                # Return deep copy to prevent modification
                return self._deep_copy_result(result)
            else:
                # Expired entry
                del self.cache[cache_key]
                self.access_order.remove(cache_key)
        
        self.stats['misses'] += 1
        return None
    
    def put(self, audio_data: np.ndarray, fft_result: Dict[str, Any]):
        """
        Cache FFT result
        
        Args:
            audio_data: Audio chunk that was processed
            fft_result: FFT processing result to cache
        """
        cache_key = self._compute_hash(audio_data)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.cache_size and cache_key not in self.cache:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
                self.stats['evictions'] += 1
        
        # Store result with timestamp
        self.cache[cache_key] = (self._deep_copy_result(fft_result), time.time())
        
        # Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
    
    def _deep_copy_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create deep copy of FFT result"""
        copied = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            elif isinstance(value, dict):
                copied[key] = self._deep_copy_result(value)
            elif isinstance(value, list):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()
        self.access_order.clear()
        logger.info(f"FFT cache cleared. Stats: {self.stats}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            'size': len(self.cache),
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def create_silence_result(self, num_bins: int) -> Dict[str, Any]:
        """Create a pre-computed result for silent audio"""
        return {
            'spectrum': np.zeros(num_bins, dtype=np.float32),
            'magnitude': np.zeros(num_bins, dtype=np.float32),
            'fft_complex': np.zeros(num_bins, dtype=np.complex64),
            'is_silent': True
        }


# Global cache instance
_global_fft_cache = FFTCache()


def get_cached_fft(audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
    """Get cached FFT result from global cache"""
    return _global_fft_cache.get(audio_data)


def cache_fft_result(audio_data: np.ndarray, fft_result: Dict[str, Any]):
    """Store FFT result in global cache"""
    _global_fft_cache.put(audio_data, fft_result)


def clear_fft_cache():
    """Clear global FFT cache"""
    _global_fft_cache.clear()


def get_fft_cache_stats() -> Dict[str, Any]:
    """Get global FFT cache statistics"""
    return _global_fft_cache.get_stats()