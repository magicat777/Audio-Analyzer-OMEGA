"""
Batched FFT Processor for OMEGA-4
Centralizes all FFT operations into single GPU batch call for optimal performance
"""

import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import time

# Try to import CuPy for GPU acceleration
# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # Fallback to NumPy

from .gpu_accelerated_fft import get_gpu_fft_processor


class FFTRequest:
    """Represents a single FFT computation request"""
    
    def __init__(self, request_id: str, audio_data: np.ndarray, 
                 fft_size: int, window_type: str = 'hann'):
        self.request_id = request_id
        self.audio_data = audio_data
        self.fft_size = fft_size
        self.window_type = window_type
        self.result = None
        self.completed = False


class BatchedFFTProcessor:
    """
    Processes multiple FFT requests in a single GPU batch operation
    Significantly reduces GPU overhead and improves FPS
    """
    
    def __init__(self, gpu_memory_limit_mb: int = 256):
        """
        Initialize the batched FFT processor
        
        Args:
            gpu_memory_limit_mb: Maximum GPU memory to use in MB
        """
        self.gpu_available = CUPY_AVAILABLE
        self.gpu_memory_limit = gpu_memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.gpu_warning_shown = False  # Track if we've shown GPU warning
        
        # Request management
        self.pending_requests = {}
        self.request_lock = threading.Lock()
        
        # Pre-allocated GPU memory pools for common FFT sizes
        self.gpu_buffers = {}
        self.common_fft_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        
        # Window functions cache
        self.window_cache = {}
        
        # Performance metrics
        self.batch_times = []
        self.last_batch_size = 0
        
        # Initialize GPU resources
        if self.gpu_available:
            self._initialize_gpu_buffers()
            
        # Get reference to GPU FFT processor
        self.gpu_fft = get_gpu_fft_processor()
        
    def _initialize_gpu_buffers(self):
        """Pre-allocate GPU buffers for common FFT sizes"""
        if not self.gpu_available:
            return
            
        try:
            # Allocate buffers for each common size
            for size in self.common_fft_sizes:
                # Input buffer (real)
                input_buffer = cp.zeros(size, dtype=cp.float32)
                # Output buffer (complex)
                output_buffer = cp.zeros(size // 2 + 1, dtype=cp.complex64)
                
                self.gpu_buffers[size] = {
                    'input': input_buffer,
                    'output': output_buffer,
                    'window': None  # Will be created on demand
                }
                
            print(f"[BatchedFFT] Initialized GPU buffers for sizes: {self.common_fft_sizes}")
            
        except Exception as e:
            print(f"[BatchedFFT] Failed to initialize GPU buffers: {e}")
            self.gpu_available = False
    
    def _get_window(self, size: int, window_type: str) -> np.ndarray:
        """Get cached window function"""
        cache_key = (size, window_type)
        
        if cache_key not in self.window_cache:
            if window_type == 'hann':
                window = np.hanning(size).astype(np.float32)
            elif window_type == 'hamming':
                window = np.hamming(size).astype(np.float32)
            elif window_type == 'blackman':
                window = np.blackman(size).astype(np.float32)
            else:
                window = np.ones(size, dtype=np.float32)
                
            self.window_cache[cache_key] = window
            
        return self.window_cache[cache_key]
    
    def prepare_batch(self, panel_id: str, audio_data: np.ndarray, 
                     fft_size: int, window_type: str = 'hann') -> str:
        """
        Add an FFT request to the current batch
        
        Args:
            panel_id: Unique identifier for the requesting panel
            audio_data: Input audio data
            fft_size: Size of FFT to compute
            window_type: Window function to apply
            
        Returns:
            Request ID for retrieving results
        """
        request_id = f"{panel_id}_{fft_size}_{time.time()}"
        
        # Ensure audio data is the right size
        if len(audio_data) > fft_size:
            audio_data = audio_data[-fft_size:]
        elif len(audio_data) < fft_size:
            audio_data = np.pad(audio_data, (0, fft_size - len(audio_data)))
            
        request = FFTRequest(request_id, audio_data, fft_size, window_type)
        
        with self.request_lock:
            self.pending_requests[request_id] = request
            
        return request_id
    
    def process_batch(self) -> int:
        """
        Process all pending FFT requests in a single GPU batch
        
        Returns:
            Number of FFTs processed
        """
        with self.request_lock:
            if not self.pending_requests:
                return 0
                
            # Group requests by FFT size for efficient batching
            size_groups = defaultdict(list)
            for request_id, request in self.pending_requests.items():
                size_groups[request.fft_size].append(request)
        
        start_time = time.perf_counter()
        total_processed = 0
        
        if self.gpu_available:
            try:
                # Process each size group on GPU
                for fft_size, requests in size_groups.items():
                    self._process_size_group_gpu(fft_size, requests)
                    total_processed += len(requests)
                    
            except Exception as e:
                if not self.gpu_warning_shown:
                    print(f"[BatchedFFT] GPU not available, using CPU processing")
                    self.gpu_warning_shown = True
                # Fallback to CPU processing
                for fft_size, requests in size_groups.items():
                    self._process_size_group_cpu(fft_size, requests)
                    total_processed += len(requests)
        else:
            # CPU processing
            for fft_size, requests in size_groups.items():
                self._process_size_group_cpu(fft_size, requests)
                total_processed += len(requests)
        
        # Record performance metrics
        batch_time = (time.perf_counter() - start_time) * 1000  # ms
        self.batch_times.append(batch_time)
        if len(self.batch_times) > 60:
            self.batch_times.pop(0)
        self.last_batch_size = total_processed
        
        return total_processed
    
    def _process_size_group_gpu(self, fft_size: int, requests: List[FFTRequest]):
        """Process a group of same-sized FFTs on GPU"""
        if not requests:
            return
            
        batch_size = len(requests)
        
        # Check if we have pre-allocated buffers
        use_preallocated = fft_size in self.gpu_buffers
        
        if use_preallocated and batch_size == 1:
            # Use pre-allocated buffers for single FFT
            buffers = self.gpu_buffers[fft_size]
            
            # Process each request
            for request in requests:
                # Apply window
                window = self._get_window(fft_size, request.window_type)
                windowed = request.audio_data * window
                
                # Transfer to GPU using pre-allocated buffer
                buffers['input'][:] = cp.asarray(windowed)
                
                # Compute FFT in-place
                cp.fft.rfft(buffers['input'], out=buffers['output'])
                
                # Get magnitude and complex results
                magnitude = cp.abs(buffers['output'])
                
                # Transfer back to CPU
                request.result = {
                    'magnitude': cp.asnumpy(magnitude),
                    'complex': cp.asnumpy(buffers['output']),
                    'frequencies': np.fft.rfftfreq(fft_size, 1/48000)
                }
                request.completed = True
                
        else:
            # Batch processing for multiple FFTs
            # Stack all audio data
            batch_input = np.zeros((batch_size, fft_size), dtype=np.float32)
            
            for i, request in enumerate(requests):
                window = self._get_window(fft_size, request.window_type)
                batch_input[i] = request.audio_data * window
            
            # Transfer entire batch to GPU
            gpu_batch = cp.asarray(batch_input)
            
            # Compute FFT batch
            gpu_fft_batch = cp.fft.rfft(gpu_batch, axis=1)
            
            # Compute magnitudes
            gpu_magnitude_batch = cp.abs(gpu_fft_batch)
            
            # Transfer results back to CPU
            magnitude_batch = cp.asnumpy(gpu_magnitude_batch)
            complex_batch = cp.asnumpy(gpu_fft_batch)
            
            # Distribute results
            frequencies = np.fft.rfftfreq(fft_size, 1/48000)
            for i, request in enumerate(requests):
                request.result = {
                    'magnitude': magnitude_batch[i],
                    'complex': complex_batch[i],
                    'frequencies': frequencies
                }
                request.completed = True
            
            # Clean up GPU memory
            del gpu_batch, gpu_fft_batch, gpu_magnitude_batch
            
    def _process_size_group_cpu(self, fft_size: int, requests: List[FFTRequest]):
        """Process a group of same-sized FFTs on CPU"""
        for request in requests:
            # Apply window
            window = self._get_window(fft_size, request.window_type)
            windowed = request.audio_data * window
            
            # Compute FFT
            fft_complex = np.fft.rfft(windowed)
            magnitude = np.abs(fft_complex)
            
            request.result = {
                'magnitude': magnitude,
                'complex': fft_complex,
                'frequencies': np.fft.rfftfreq(fft_size, 1/48000)
            }
            request.completed = True
    
    def distribute_results(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieve all completed FFT results
        
        Returns:
            Dictionary mapping request_id to FFT results
        """
        results = {}
        completed_ids = []
        
        with self.request_lock:
            for request_id, request in self.pending_requests.items():
                if request.completed:
                    results[request_id] = request.result
                    completed_ids.append(request_id)
            
            # Remove completed requests
            for request_id in completed_ids:
                del self.pending_requests[request_id]
                
        return results
    
    def get_result_for_panel(self, request_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get result for a specific request ID
        
        Args:
            request_id: The request ID returned by prepare_batch
            
        Returns:
            FFT results or None if not ready
        """
        with self.request_lock:
            request = self.pending_requests.get(request_id)
            if request and request.completed:
                result = request.result
                del self.pending_requests[request_id]
                return result
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'gpu_enabled': self.gpu_available,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'max_batch_time': max(self.batch_times) if self.batch_times else 0,
            'last_batch_size': self.last_batch_size,
            'pending_requests': len(self.pending_requests)
        }
        
        if self.gpu_available and hasattr(cp, 'get_default_memory_pool'):
            mempool = cp.get_default_memory_pool()
            stats['gpu_memory_used_mb'] = mempool.used_bytes() / 1024 / 1024
            stats['gpu_memory_total_mb'] = mempool.total_bytes() / 1024 / 1024
            
        return stats
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if self.gpu_available:
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                print("[BatchedFFT] GPU cache cleared")
            except Exception as e:
                print(f"[BatchedFFT] Failed to clear GPU cache: {e}")
    
    def shutdown(self):
        """Clean up resources"""
        self.clear_gpu_cache()
        self.gpu_buffers.clear()
        self.window_cache.clear()
        self.pending_requests.clear()


# Global instance for easy access
_batched_fft_instance = None

def get_batched_fft_processor() -> BatchedFFTProcessor:
    """Get or create the global batched FFT processor instance"""
    global _batched_fft_instance
    if _batched_fft_instance is None:
        _batched_fft_instance = BatchedFFTProcessor()
    return _batched_fft_instance