"""
GPU-Accelerated FFT Processing for OMEGA-4
Uses CuPy for NVIDIA GPU acceleration if available, falls back to NumPy
"""

import numpy as np
import threading
from typing import Dict, Optional, Tuple, Any

# Try to import CuPy for GPU acceleration
# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled for FFT operations")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - falling back to CPU-based FFT")
    cp = np  # Fallback to NumPy


class GPUAcceleratedFFT:
    """GPU-accelerated FFT processor with caching and optimization"""
    
    def __init__(self, max_fft_size: int = 16384):
        self.max_fft_size = max_fft_size
        self.gpu_available = CUPY_AVAILABLE
        
        # FFT result cache - stores both magnitude and complex results
        self.fft_cache = {}
        self.cache_lock = threading.Lock()
        
        # Pre-allocate GPU memory if available
        if self.gpu_available:
            try:
                # Test GPU allocation
                test = cp.zeros(max_fft_size, dtype=cp.float32)
                del test
                
                # Pre-allocate arrays on GPU
                self.gpu_input_buffer = cp.zeros(max_fft_size, dtype=cp.float32)
                self.gpu_output_buffer = cp.zeros(max_fft_size // 2 + 1, dtype=cp.complex64)
                
                # Test FFT functionality
                test_signal = cp.random.randn(1024).astype(cp.float32)
                test_fft = cp.fft.rfft(test_signal)
                del test_signal, test_fft
                
                # Create FFT plan for better performance
                self._create_fft_plans()
                
                print(f"GPU memory pre-allocated for FFT size up to {max_fft_size}")
            except Exception as e:
                print(f"GPU initialization failed: {e}")
                self.gpu_available = False
                
        # Pre-compute windows
        self.windows = {
            'hann': self._precompute_windows(np.hanning),
            'hamming': self._precompute_windows(np.hamming),
            'blackman': self._precompute_windows(np.blackman)
        }
        
    def _create_fft_plans(self):
        """Create cuFFT plans for common FFT sizes"""
        if not self.gpu_available:
            return
            
        self.fft_plans = {}
        common_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        
        for size in common_sizes:
            try:
                # CuPy automatically caches FFT plans internally
                # We just warm up the cache
                test_input = cp.zeros(size, dtype=cp.float32)
                cp.fft.rfft(test_input)
                del test_input
            except Exception:
                pass
                
    def _precompute_windows(self, window_func) -> Dict[int, np.ndarray]:
        """Pre-compute window functions for common sizes"""
        windows = {}
        sizes = [512, 1024, 2048, 4096, 8192, 16384]
        
        for size in sizes:
            windows[size] = window_func(size).astype(np.float32)
            
        return windows
        
    def compute_fft(self, audio_data: np.ndarray, 
                   window_type: str = 'hann',
                   return_complex: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute FFT with GPU acceleration if available
        
        Returns:
            magnitude_spectrum, complex_spectrum (if return_complex=True)
        """
        n_samples = len(audio_data)
        
        # Check cache first
        cache_key = (n_samples, window_type, audio_data.tobytes()[:100])  # Use first 100 bytes as key
        
        with self.cache_lock:
            if cache_key in self.fft_cache:
                cached_result = self.fft_cache[cache_key]
                if return_complex:
                    return cached_result['magnitude'], cached_result['complex']
                else:
                    return cached_result['magnitude'], None
        
        # Apply window
        if n_samples in self.windows.get(window_type, {}):
            windowed = audio_data * self.windows[window_type][n_samples]
        else:
            # Create window on the fly
            if window_type == 'hann':
                window = np.hanning(n_samples)
            elif window_type == 'hamming':
                window = np.hamming(n_samples)
            else:
                window = np.blackman(n_samples)
            windowed = audio_data * window.astype(np.float32)
        
        # Compute FFT
        if self.gpu_available and n_samples <= self.max_fft_size:
            try:
                # Transfer to GPU
                gpu_data = cp.asarray(windowed, dtype=cp.float32)
                
                # Compute FFT on GPU
                fft_complex_gpu = cp.fft.rfft(gpu_data)
                
                # Calculate magnitude on GPU
                magnitude_gpu = cp.abs(fft_complex_gpu)
                
                # Transfer back to CPU
                magnitude = cp.asnumpy(magnitude_gpu)
                if return_complex:
                    fft_complex = cp.asnumpy(fft_complex_gpu)
                else:
                    fft_complex = None
                    
                # Clean up GPU memory
                del gpu_data, fft_complex_gpu, magnitude_gpu
                
            except Exception as e:
                # Fallback to CPU if GPU fails
                print(f"GPU FFT failed, falling back to CPU: {e}")
                fft_complex = np.fft.rfft(windowed)
                magnitude = np.abs(fft_complex)
                if not return_complex:
                    fft_complex = None
        else:
            # CPU fallback
            fft_complex = np.fft.rfft(windowed)
            magnitude = np.abs(fft_complex)
            if not return_complex:
                fft_complex = None
        
        # Cache the result
        with self.cache_lock:
            self.fft_cache[cache_key] = {
                'magnitude': magnitude,
                'complex': fft_complex if return_complex else None,
                'timestamp': threading.current_thread().ident
            }
            
            # Limit cache size
            if len(self.fft_cache) > 10:
                # Remove oldest entry
                oldest_key = next(iter(self.fft_cache))
                del self.fft_cache[oldest_key]
        
        return magnitude, fft_complex
    
    def compute_multi_resolution_fft(self, audio_data: np.ndarray,
                                   resolutions: Dict[str, int],
                                   window_type: str = 'hann') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute multiple FFT resolutions in parallel on GPU
        
        Args:
            audio_data: Input audio samples
            resolutions: Dict of name -> FFT size
            window_type: Window function to use
            
        Returns:
            Dict of resolution_name -> {'magnitude': array, 'complex': array, 'freqs': array}
        """
        results = {}
        
        if self.gpu_available and len(resolutions) > 1:
            try:
                # Process all resolutions on GPU in batch
                for name, fft_size in resolutions.items():
                    # Get the appropriate chunk of audio
                    if len(audio_data) >= fft_size:
                        chunk = audio_data[-fft_size:]
                    else:
                        # Pad with zeros if needed
                        chunk = np.pad(audio_data, (0, fft_size - len(audio_data)))
                    
                    # Compute FFT
                    magnitude, fft_complex = self.compute_fft(chunk, window_type, return_complex=True)
                    
                    # Compute frequency array
                    freqs = np.fft.rfftfreq(fft_size, 1/48000)  # Assuming 48kHz sample rate
                    
                    results[name] = {
                        'magnitude': magnitude,
                        'complex': fft_complex,
                        'freqs': freqs
                    }
                    
            except Exception as e:
                print(f"Multi-resolution GPU FFT failed: {e}")
                # Fallback to sequential CPU processing
                results = self._compute_multi_resolution_cpu(audio_data, resolutions, window_type)
        else:
            # CPU fallback
            results = self._compute_multi_resolution_cpu(audio_data, resolutions, window_type)
            
        return results
    
    def _compute_multi_resolution_cpu(self, audio_data: np.ndarray,
                                    resolutions: Dict[str, int],
                                    window_type: str) -> Dict[str, Dict[str, np.ndarray]]:
        """CPU fallback for multi-resolution FFT"""
        results = {}
        
        for name, fft_size in resolutions.items():
            # Get the appropriate chunk of audio
            if len(audio_data) >= fft_size:
                chunk = audio_data[-fft_size:]
            else:
                # Pad with zeros if needed
                chunk = np.pad(audio_data, (0, fft_size - len(audio_data)))
            
            # Compute FFT
            magnitude, fft_complex = self.compute_fft(chunk, window_type, return_complex=True)
            
            # Compute frequency array
            freqs = np.fft.rfftfreq(fft_size, 1/48000)
            
            results[name] = {
                'magnitude': magnitude,
                'complex': fft_complex,
                'freqs': freqs
            }
            
        return results
    
    def clear_cache(self):
        """Clear FFT cache to free memory"""
        with self.cache_lock:
            self.fft_cache.clear()
            
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get GPU memory usage information"""
        if not self.gpu_available:
            return {'available': False}
            
        try:
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()
            
            return {
                'available': True,
                'used_mb': used_bytes / 1024 / 1024,
                'total_mb': total_bytes / 1024 / 1024,
                'utilization': used_bytes / total_bytes if total_bytes > 0 else 0
            }
        except Exception:
            return {'available': False}
    
    # Batch processing methods for integration with BatchedFFTProcessor
    def prepare_batch_arrays(self, batch_size: int, fft_size: int) -> Tuple[Any, Any]:
        """
        Prepare GPU arrays for batch processing
        
        Returns:
            (input_array, output_array) on GPU
        """
        if not self.gpu_available:
            return None, None
            
        try:
            # Allocate batch arrays on GPU
            input_batch = cp.zeros((batch_size, fft_size), dtype=cp.float32)
            output_batch = cp.zeros((batch_size, fft_size // 2 + 1), dtype=cp.complex64)
            return input_batch, output_batch
        except Exception as e:
            print(f"Failed to allocate batch arrays: {e}")
            return None, None
    
    def process_fft_batch(self, input_batch: Any, window_type: str = 'hann') -> Any:
        """
        Process a batch of FFTs on GPU
        
        Args:
            input_batch: GPU array of shape (batch_size, fft_size)
            window_type: Window function to apply
            
        Returns:
            GPU array of FFT results
        """
        if not self.gpu_available or input_batch is None:
            return None
            
        try:
            batch_size, fft_size = input_batch.shape
            
            # Get window for this size
            if fft_size in self.windows.get(window_type, {}):
                window = cp.asarray(self.windows[window_type][fft_size])
            else:
                # Create window on CPU and transfer
                if window_type == 'hann':
                    window_cpu = np.hanning(fft_size).astype(np.float32)
                elif window_type == 'hamming':
                    window_cpu = np.hamming(fft_size).astype(np.float32)
                else:
                    window_cpu = np.blackman(fft_size).astype(np.float32)
                window = cp.asarray(window_cpu)
            
            # Apply window to entire batch (broadcasting)
            windowed_batch = input_batch * window[np.newaxis, :]
            
            # Compute FFT batch
            fft_batch = cp.fft.rfft(windowed_batch, axis=1)
            
            return fft_batch
            
        except Exception as e:
            print(f"Batch FFT processing failed: {e}")
            return None
    
    def setup_memory_pool(self, size_mb: int = 256):
        """
        Setup GPU memory pool with specified size
        
        Args:
            size_mb: Size of memory pool in megabytes
        """
        if not self.gpu_available:
            return
            
        try:
            mempool = cp.get_default_memory_pool()
            # Set memory limit
            mempool.set_limit(size_mb * 1024 * 1024)
            print(f"GPU memory pool set to {size_mb}MB")
        except Exception as e:
            print(f"Failed to setup memory pool: {e}")
    
    def enable_zero_copy(self):
        """Enable zero-copy memory transfers between CPU and GPU"""
        if not self.gpu_available:
            return
            
        try:
            # Enable unified memory for zero-copy
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            print("Zero-copy memory enabled")
        except Exception as e:
            print(f"Failed to enable zero-copy: {e}")


# Global instance for easy access
_gpu_fft_instance = None

def get_gpu_fft_processor() -> GPUAcceleratedFFT:
    """Get or create the global GPU FFT processor instance"""
    global _gpu_fft_instance
    if _gpu_fft_instance is None:
        _gpu_fft_instance = GPUAcceleratedFFT()
    return _gpu_fft_instance