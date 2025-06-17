#!/usr/bin/env python3
"""
Performance Test Script for OMEGA-4 Optimizations
Tests GPU acceleration and compares performance
"""

import numpy as np
import time
import sys

def test_cpu_fft(size=16384, iterations=100):
    """Test CPU-based FFT performance"""
    data = np.random.randn(size).astype(np.float32)
    
    start_time = time.time()
    for _ in range(iterations):
        result = np.fft.rfft(data)
    cpu_time = (time.time() - start_time) / iterations * 1000
    
    return cpu_time

def test_gpu_fft(size=16384, iterations=100):
    """Test GPU-based FFT performance"""
    try:
        import cupy as cp
        
        # Prepare data on GPU
        data = cp.random.randn(size).astype(cp.float32)
        
        # Warm up GPU
        for _ in range(10):
            _ = cp.fft.rfft(data)
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(iterations):
            result = cp.fft.rfft(data)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU operations complete
        gpu_time = (time.time() - start_time) / iterations * 1000
        
        return gpu_time
        
    except ImportError:
        return None

def test_parallel_performance():
    """Test parallel panel update performance"""
    try:
        from omega4.optimization.parallel_panel_updater import ParallelPanelUpdater
        
        # Create dummy panel update functions
        def dummy_panel_update(data):
            # Simulate panel computation
            result = np.sum(data['test_data'] ** 2)
            time.sleep(0.002)  # Simulate 2ms of work
            return result
        
        # Test sequential updates
        start_time = time.time()
        test_data = {'test_data': np.random.randn(1000)}
        for _ in range(10):
            dummy_panel_update(test_data)
        sequential_time = (time.time() - start_time) * 1000
        
        # Test parallel updates
        updater = ParallelPanelUpdater(max_workers=4)
        for i in range(10):
            updater.register_panel(f'panel_{i}', lambda d: dummy_panel_update(d))
        
        start_time = time.time()
        futures = updater.update_all_panels(test_data)
        # Wait for completion
        import concurrent.futures
        concurrent.futures.wait(futures.values())
        parallel_time = (time.time() - start_time) * 1000
        
        updater.shutdown()
        
        return sequential_time, parallel_time
        
    except ImportError as e:
        print(f"Could not test parallel performance: {e}")
        return None, None

def main():
    print("OMEGA-4 Performance Test")
    print("========================\n")
    
    # Test FFT sizes
    fft_sizes = [1024, 4096, 16384]
    
    print("Testing FFT Performance...")
    print("-" * 50)
    
    for size in fft_sizes:
        print(f"\nFFT Size: {size}")
        
        # CPU test
        cpu_time = test_cpu_fft(size, 100)
        print(f"  CPU Time: {cpu_time:.2f} ms")
        
        # GPU test
        gpu_time = test_gpu_fft(size, 100)
        if gpu_time is not None:
            print(f"  GPU Time: {gpu_time:.2f} ms")
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.1f}x")
        else:
            print("  GPU: Not available (CuPy not installed)")
    
    print("\n" + "-" * 50)
    print("\nTesting Parallel Panel Updates...")
    
    seq_time, par_time = test_parallel_performance()
    if seq_time and par_time:
        print(f"Sequential time (10 panels): {seq_time:.1f} ms")
        print(f"Parallel time (10 panels): {par_time:.1f} ms")
        print(f"Speedup: {seq_time/par_time:.1f}x")
    
    print("\n" + "-" * 50)
    print("\nExpected Performance Improvements:")
    print("- With GPU: 10-50x faster FFT processing")
    print("- With parallel updates: 2-4x faster panel updates")
    print("- Overall: Should achieve stable 60 FPS")
    
    # Check if running from optimized version
    try:
        from omega4.optimization.gpu_accelerated_fft import get_gpu_fft_processor
        processor = get_gpu_fft_processor()
        
        print(f"\n✓ GPU FFT processor available: {'GPU' if processor.gpu_available else 'CPU'} mode")
        
        if processor.gpu_available:
            mem_info = processor.get_gpu_memory_info()
            if mem_info.get('available'):
                print(f"✓ GPU Memory: {mem_info['used_mb']:.1f}/{mem_info['total_mb']:.1f} MB used")
    except ImportError:
        print("\n✗ Optimization modules not found. Please ensure they are in the correct path.")

if __name__ == "__main__":
    main()