"""
Multi-Resolution FFT Module for OMEGA-4 Audio Analyzer
Phase 5: Extract FFT processing with psychoacoustic weighting
Enhanced with robust error handling and performance optimizations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class WindowType(Enum):
    """Window function types"""
    BLACKMAN = "blackman"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN_HARRIS = "blackman_harris"

@dataclass
class FFTConfig:
    """Configuration for a single FFT resolution"""
    freq_range: Tuple[float, float]
    fft_size: int
    hop_size: int
    weight: float
    window_type: WindowType = WindowType.BLACKMAN

    def __post_init__(self):
        """Validate configuration"""
        if self.freq_range[0] >= self.freq_range[1]:
            raise ValueError(f"Invalid frequency range: {self.freq_range}")
        if self.fft_size <= 0 or (self.fft_size & (self.fft_size - 1)) != 0:
            raise ValueError(f"FFT size must be power of 2: {self.fft_size}")
        if self.hop_size <= 0:
            raise ValueError(f"Hop size must be positive: {self.hop_size}")
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive: {self.weight}")

class FFTResult(NamedTuple):
    """Result from FFT processing"""
    magnitude: np.ndarray
    frequencies: np.ndarray
    config_index: int

class CircularBuffer:
    """Thread-safe circular buffer for audio data"""
    
    def __init__(self, size: int, dtype=np.float32):
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_pos = 0
        self.samples_written = 0
        self._lock = threading.Lock()
        
    def write(self, data: np.ndarray) -> bool:
        """Write data to buffer, returns True if successful"""
        if data is None or len(data) == 0:
            return False
            
        try:
            with self._lock:
                data_len = len(data)
                
                # Handle data larger than buffer
                if data_len >= self.size:
                    # Take only the last 'size' samples
                    self.buffer[:] = data[-self.size:]
                    self.write_pos = 0
                    self.samples_written = self.size
                else:
                    # Normal circular write
                    if self.write_pos + data_len <= self.size:
                        self.buffer[self.write_pos:self.write_pos + data_len] = data
                    else:
                        # Wrap around
                        first_part = self.size - self.write_pos
                        self.buffer[self.write_pos:] = data[:first_part]
                        self.buffer[:data_len - first_part] = data[first_part:]
                    
                    self.write_pos = (self.write_pos + data_len) % self.size
                    self.samples_written = min(self.samples_written + data_len, self.size)
                
                return True
                
        except Exception as e:
            logger.error(f"Buffer write failed: {e}")
            return False
    
    def read_latest(self, length: int) -> Optional[np.ndarray]:
        """Read latest 'length' samples from buffer"""
        if length <= 0 or length > self.size:
            return None
            
        try:
            with self._lock:
                if self.samples_written < length:
                    # Not enough data yet
                    return None
                
                output = np.zeros(length, dtype=self.buffer.dtype)
                
                if self.write_pos >= length:
                    # Simple case - no wrap around
                    output[:] = self.buffer[self.write_pos - length:self.write_pos]
                else:
                    # Wrap around case
                    first_part = length - self.write_pos
                    output[:first_part] = self.buffer[-first_part:]
                    if self.write_pos > 0:
                        output[first_part:] = self.buffer[:self.write_pos]
                
                return output
                
        except Exception as e:
            logger.error(f"Buffer read failed: {e}")
            return None
    
    def reset(self):
        """Reset buffer to empty state"""
        with self._lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.samples_written = 0

class MultiResolutionFFT:
    """Multi-resolution FFT analysis for enhanced low-end detail"""

    def __init__(self, sample_rate: int = 48000, max_freq: float = 20000):
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if max_freq <= 0 or max_freq > sample_rate / 2:
            raise ValueError("Max frequency must be positive and <= Nyquist")
            
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        self.max_freq = min(max_freq, self.nyquist)

        # Define FFT configurations with validation
        self.configs = [
            FFTConfig((20, 200), 4096, 1024, 1.5),      # High resolution bass
            FFTConfig((200, 1000), 2048, 512, 1.2),     # Low-mids
            FFTConfig((1000, 5000), 1024, 256, 1.0),    # Mids  
            FFTConfig((5000, 20000), 1024, 256, 1.5),   # Highs
        ]

        # Initialize processing components
        self._setup_windows()
        self._setup_buffers()
        self._setup_frequency_arrays()
        self._setup_working_arrays()
        
        # Performance tracking
        self.processing_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'error_count': 0
        }
        
        logger.info(f"MultiResolutionFFT initialized: {sample_rate}Hz, {len(self.configs)} resolutions")

    def _setup_windows(self):
        """Pre-compute window functions for each configuration"""
        self.windows = {}
        
        for i, config in enumerate(self.configs):
            try:
                if config.window_type == WindowType.BLACKMAN:
                    window = np.blackman(config.fft_size)
                elif config.window_type == WindowType.HANN:
                    window = np.hann(config.fft_size)
                elif config.window_type == WindowType.HAMMING:
                    window = np.hamming(config.fft_size)
                elif config.window_type == WindowType.BLACKMAN_HARRIS:
                    window = np.blackman(config.fft_size)  # Fallback to blackman
                else:
                    window = np.blackman(config.fft_size)  # Default fallback
                
                self.windows[i] = window.astype(np.float32)
                
            except Exception as e:
                logger.error(f"Window creation failed for config {i}: {e}")
                # Emergency fallback
                self.windows[i] = np.ones(config.fft_size, dtype=np.float32)

    def _setup_buffers(self):
        """Initialize circular buffers for each resolution"""
        self.buffers = {}
        
        for i, config in enumerate(self.configs):
            try:
                # Buffer size should be at least 2x FFT size for good overlap
                buffer_size = max(config.fft_size * 2, config.fft_size + config.hop_size)
                self.buffers[i] = CircularBuffer(buffer_size)
                
            except Exception as e:
                logger.error(f"Buffer creation failed for config {i}: {e}")
                # Create minimal buffer
                self.buffers[i] = CircularBuffer(config.fft_size)

    def _setup_frequency_arrays(self):
        """Pre-compute frequency arrays for each configuration"""
        self.freq_arrays = {}
        
        for i, config in enumerate(self.configs):
            self.freq_arrays[i] = np.fft.rfftfreq(config.fft_size, 1 / self.sample_rate)

    def _setup_working_arrays(self):
        """Pre-allocate working arrays to avoid repeated allocation"""
        self.working_arrays = {}
        
        for i, config in enumerate(self.configs):
            self.working_arrays[i] = {
                'audio_data': np.zeros(config.fft_size, dtype=np.float32),
                'windowed': np.zeros(config.fft_size, dtype=np.float32),
                'weights': np.ones(config.fft_size // 2 + 1, dtype=np.float32)
            }

    def process_audio_chunk(self, audio_chunk: np.ndarray, 
                          apply_weighting: bool = True) -> Dict[int, FFTResult]:
        """
        Process audio with multiple FFT resolutions
        
        Args:
            audio_chunk: Input audio data
            apply_weighting: Whether to apply psychoacoustic weighting
            
        Returns:
            Dictionary mapping config index to FFT results
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            logger.warning("Empty audio chunk received")
            return {}

        try:
            import time
            start_time = time.perf_counter()
            
            results = {}
            
            # Process each resolution
            for i, config in enumerate(self.configs):
                try:
                    # Write to circular buffer
                    if not self.buffers[i].write(audio_chunk):
                        logger.warning(f"Failed to write to buffer {i}")
                        continue
                    
                    # Read latest samples for FFT
                    audio_data = self.buffers[i].read_latest(config.fft_size)
                    if audio_data is None:
                        continue  # Not enough data yet
                    
                    # Use pre-allocated arrays
                    working = self.working_arrays[i]
                    np.copyto(working['audio_data'], audio_data)
                    
                    # Apply window
                    np.multiply(working['audio_data'], self.windows[i], 
                              out=working['windowed'])
                    
                    # Compute FFT
                    fft_result = np.fft.rfft(working['windowed'])
                    magnitude = np.abs(fft_result)
                    
                    # Apply psychoacoustic weighting
                    if apply_weighting:
                        magnitude = self._apply_psychoacoustic_weighting(
                            magnitude, config, i, working['weights']
                        )
                    
                    # Store result
                    results[i] = FFTResult(
                        magnitude=magnitude.copy(),  # Copy to avoid overwriting
                        frequencies=self.freq_arrays[i],
                        config_index=i
                    )
                    
                except Exception as e:
                    logger.error(f"FFT processing failed for config {i}: {e}")
                    self.processing_stats['error_count'] += 1
                    continue
            
            # Update stats
            self.processing_stats['total_calls'] += 1
            self.processing_stats['total_time'] += time.perf_counter() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"Multi-resolution FFT processing failed: {e}")
            self.processing_stats['error_count'] += 1
            return {}

    def _apply_psychoacoustic_weighting(self, magnitude: np.ndarray, 
                                      config: FFTConfig, config_index: int,
                                      weights_array: np.ndarray) -> np.ndarray:
        """Apply psychoacoustic weighting for perceptually accurate analysis"""
        try:
            # Reset weights array
            weights_array.fill(config.weight)
            
            freqs = self.freq_arrays[config_index]
            freq_range = config.freq_range
            
            # Vectorized weight application for better performance
            mask_60_120 = (freqs >= 60) & (freqs <= 120)
            mask_200_400 = (freqs >= 200) & (freqs <= 400)
            mask_2k_5k = (freqs >= 2000) & (freqs <= 5000)
            mask_20_80 = (freqs >= 20) & (freqs <= 80)
            mask_range = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            
            # Apply weights only within frequency range
            weights_array[mask_range & mask_60_120] *= 1.8    # Kick drum
            weights_array[mask_range & mask_200_400] *= 1.4   # Voice fundamental
            weights_array[mask_range & mask_2k_5k] *= 1.2     # Presence/clarity
            weights_array[mask_range & mask_20_80] *= 1.6     # Sub-bass
            
            # Apply weights to magnitude
            return magnitude * weights_array[:len(magnitude)]
            
        except Exception as e:
            logger.error(f"Psychoacoustic weighting failed: {e}")
            return magnitude  # Return unweighted on error

    def combine_results_optimized(self, results: Dict[int, FFTResult], 
                                target_bins: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized combination of multi-resolution FFT results
        
        Args:
            results: Dictionary of FFT results from different resolutions
            target_bins: Number of output frequency bins
            
        Returns:
            Tuple of (combined_magnitude, target_frequencies)
        """
        if not results:
            logger.warning("No FFT results to combine")
            return np.zeros(target_bins), np.linspace(0, self.max_freq, target_bins)

        try:
            # Create target frequency array
            target_freqs = np.linspace(0, self.max_freq, target_bins)
            combined_magnitude = np.zeros(target_bins, dtype=np.float32)
            weight_sum = np.zeros(target_bins, dtype=np.float32)
            
            # Process each resolution
            for result in results.values():
                config = self.configs[result.config_index]
                freqs = result.frequencies
                magnitude = result.magnitude
                freq_range = config.freq_range
                
                # Find valid frequency range
                valid_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                if not np.any(valid_mask):
                    continue
                    
                valid_freqs = freqs[valid_mask]
                valid_magnitude = magnitude[valid_mask]
                
                if len(valid_freqs) < 2:
                    continue
                
                # Find target frequency bins that overlap with this resolution
                target_mask = ((target_freqs >= freq_range[0]) & 
                             (target_freqs <= freq_range[1]))
                
                if not np.any(target_mask):
                    continue
                
                target_subset = target_freqs[target_mask]
                target_indices = np.where(target_mask)[0]
                
                # Vectorized interpolation for better performance
                interpolated = np.interp(target_subset, valid_freqs, valid_magnitude)
                
                # Add to combined result with weighting
                combined_magnitude[target_indices] += interpolated * config.weight
                weight_sum[target_indices] += config.weight
            
            # Normalize by weight sum
            valid_weights = weight_sum > 0
            combined_magnitude[valid_weights] /= weight_sum[valid_weights]
            
            return combined_magnitude, target_freqs
            
        except Exception as e:
            logger.error(f"FFT result combination failed: {e}")
            return np.zeros(target_bins), np.linspace(0, self.max_freq, target_bins)

    def get_frequency_arrays(self) -> Dict[int, np.ndarray]:
        """Get frequency arrays for each FFT configuration"""
        return self.freq_arrays.copy()

    def reset_all_buffers(self):
        """Reset all audio buffers"""
        try:
            for buffer in self.buffers.values():
                buffer.reset()
            logger.info("All buffers reset")
        except Exception as e:
            logger.error(f"Buffer reset failed: {e}")

    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing performance statistics"""
        stats = self.processing_stats.copy()
        if stats['total_calls'] > 0:
            stats['avg_time_ms'] = (stats['total_time'] / stats['total_calls']) * 1000
            stats['error_rate'] = stats['error_count'] / stats['total_calls']
        else:
            stats['avg_time_ms'] = 0.0
            stats['error_rate'] = 0.0
        return stats

    def get_buffer_status(self) -> Dict[int, Dict[str, int]]:
        """Get status of all buffers for debugging"""
        status = {}
        for i, buffer in self.buffers.items():
            status[i] = {
                'size': buffer.size,
                'write_pos': buffer.write_pos,
                'samples_written': buffer.samples_written,
                'utilization_pct': int((buffer.samples_written / buffer.size) * 100)
            }
        return status

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.reset_all_buffers()
            # Clear working arrays
            for working in self.working_arrays.values():
                for arr in working.values():
                    if hasattr(arr, 'fill'):
                        arr.fill(0)
            logger.info("MultiResolutionFFT cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Convenience function for simple usage
def create_default_multi_fft(sample_rate: int = 48000) -> MultiResolutionFFT:
    """Create MultiResolutionFFT with default settings"""
    return MultiResolutionFFT(sample_rate=sample_rate)


# Performance testing function
def benchmark_multi_fft(sample_rate: int = 48000, 
                       chunk_size: int = 512, 
                       num_iterations: int = 1000) -> Dict[str, float]:
    """Benchmark MultiResolutionFFT performance"""
    import time
    
    fft_processor = create_default_multi_fft(sample_rate)
    test_audio = np.random.random(chunk_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        fft_processor.process_audio_chunk(test_audio)
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        results = fft_processor.process_audio_chunk(test_audio)
        if results:  # Only combine if we have results
            combined, freqs = fft_processor.combine_results_optimized(results)
    
    total_time = time.perf_counter() - start_time
    
    return {
        'total_time_s': total_time,
        'avg_time_ms': (total_time / num_iterations) * 1000,
        'iterations_per_second': num_iterations / total_time,
        'stats': fft_processor.get_processing_stats()
    }