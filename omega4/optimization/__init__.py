# Performance optimization module for OMEGA-4

from .adaptive_updater import AdaptiveUpdater
from .array_pool import (
    ArrayPool, ScopedArray,
    get_array, return_array, clear_pool, get_pool_stats, with_array
)
from .fft_cache import (
    FFTCache,
    get_cached_fft, cache_fft_result, clear_fft_cache, get_fft_cache_stats
)
from .freq_mapper import PrecomputedFrequencyMapper, FrequencyMapping

__all__ = [
    'AdaptiveUpdater',
    'ArrayPool',
    'ScopedArray', 
    'get_array',
    'return_array',
    'clear_pool',
    'get_pool_stats',
    'with_array',
    'FFTCache',
    'get_cached_fft',
    'cache_fft_result',
    'clear_fft_cache',
    'get_fft_cache_stats',
    'PrecomputedFrequencyMapper',
    'FrequencyMapping'
]