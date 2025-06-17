#!/usr/bin/env python3
"""
Test GPU FFT functionality
"""

import numpy as np
import time

print("Testing GPU FFT acceleration...")

try:
    import cupy as cp
    print("✓ CuPy imported successfully")
    
    # Test GPU memory allocation
    gpu_array = cp.zeros((1024,), dtype=cp.complex64)
    print("✓ GPU memory allocation successful")
    
    # Test FFT
    test_signal = cp.random.randn(4096).astype(cp.float32)
    fft_result = cp.fft.rfft(test_signal)
    print("✓ GPU FFT computation successful")
    
    # Performance test
    sizes = [1024, 2048, 4096, 8192]
    print("\nPerformance comparison:")
    print("-" * 50)
    print(f"{'FFT Size':<10} {'CPU (ms)':<15} {'GPU (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        # CPU timing
        cpu_signal = np.random.randn(size).astype(np.float32)
        start = time.perf_counter()
        for _ in range(100):
            cpu_fft = np.fft.rfft(cpu_signal)
        cpu_time = (time.perf_counter() - start) / 100 * 1000
        
        # GPU timing
        gpu_signal = cp.asarray(cpu_signal)
        cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
        start = time.perf_counter()
        for _ in range(100):
            gpu_fft = cp.fft.rfft(gpu_signal)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        gpu_time = (time.perf_counter() - start) / 100 * 1000
        
        speedup = cpu_time / gpu_time
        print(f"{size:<10} {cpu_time:<15.3f} {gpu_time:<15.3f} {speedup:<10.2f}x")
    
    print("\n✅ GPU acceleration is working correctly!")
    
    # Check CUDA version
    print(f"\nCUDA Runtime Version: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"CuPy Version: {cp.__version__}")
    
except Exception as e:
    print(f"❌ GPU test failed: {e}")
    import traceback
    traceback.print_exc()