#!/usr/bin/env python3
"""
Test script to verify GPU batched FFT performance improvement
"""

import numpy as np
import time
import sys

# Add omega4 to path
sys.path.insert(0, '.')

from omega4.optimization.batched_fft_processor import get_batched_fft_processor
from omega4.optimization.gpu_accelerated_fft import get_gpu_fft_processor

def test_individual_ffts(audio_data_list, fft_sizes, iterations=100):
    """Test individual FFT calls (old method)"""
    gpu_fft = get_gpu_fft_processor()
    
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        results = []
        for audio_data, fft_size in zip(audio_data_list, fft_sizes):
            magnitude, complex_fft = gpu_fft.compute_fft(audio_data, return_complex=True)
            results.append((magnitude, complex_fft))
    
    elapsed = (time.perf_counter() - start_time) * 1000
    avg_time = elapsed / iterations
    
    return avg_time, results

def test_batched_ffts(audio_data_list, fft_sizes, iterations=100):
    """Test batched FFT calls (new method)"""
    batched_fft = get_batched_fft_processor()
    
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        # Prepare batch
        request_ids = []
        for i, (audio_data, fft_size) in enumerate(zip(audio_data_list, fft_sizes)):
            request_id = batched_fft.prepare_batch(f'test_{i}', audio_data, fft_size)
            request_ids.append(request_id)
        
        # Process batch
        batched_fft.process_batch()
        
        # Get results
        results = batched_fft.distribute_results()
    
    elapsed = (time.perf_counter() - start_time) * 1000
    avg_time = elapsed / iterations
    
    return avg_time, results

def main():
    print("GPU Batched FFT Performance Test")
    print("=" * 50)
    
    # Simulate typical frame with multiple FFT requests
    sample_rate = 48000
    
    # Different FFT sizes used by panels
    test_configs = [
        (16384, 'main_spectrum'),
        (4096, 'music_analysis'),
        (2048, 'harmonic_1'),
        (2048, 'harmonic_2'),
        (1024, 'transient'),
    ]
    
    # Generate test audio data
    audio_data_list = []
    fft_sizes = []
    
    for fft_size, name in test_configs:
        # Generate test signal with some harmonics
        t = np.linspace(0, fft_size/sample_rate, fft_size)
        signal = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*880*t)
        signal = signal.astype(np.float32)
        
        audio_data_list.append(signal)
        fft_sizes.append(fft_size)
        print(f"  {name}: {fft_size} samples")
    
    print(f"\nTotal FFTs per frame: {len(test_configs)}")
    print("-" * 50)
    
    # Warm up
    print("\nWarming up...")
    test_individual_ffts(audio_data_list, fft_sizes, iterations=10)
    test_batched_ffts(audio_data_list, fft_sizes, iterations=10)
    
    # Test individual FFTs
    print("\nTesting INDIVIDUAL FFTs (old method)...")
    individual_time, _ = test_individual_ffts(audio_data_list, fft_sizes, iterations=100)
    print(f"  Average time per frame: {individual_time:.2f} ms")
    print(f"  Max theoretical FPS: {1000/individual_time:.1f}")
    
    # Test batched FFTs
    print("\nTesting BATCHED FFTs (new method)...")
    batched_time, _ = test_batched_ffts(audio_data_list, fft_sizes, iterations=100)
    print(f"  Average time per frame: {batched_time:.2f} ms")
    print(f"  Max theoretical FPS: {1000/batched_time:.1f}")
    
    # Calculate improvement
    print("\n" + "=" * 50)
    speedup = individual_time / batched_time
    fps_improvement = (1000/batched_time) - (1000/individual_time)
    
    print(f"SPEEDUP: {speedup:.2f}x faster")
    print(f"FPS IMPROVEMENT: +{fps_improvement:.1f} FPS")
    
    # Expected FPS with 16.67ms frame budget
    frame_budget = 16.67  # 60 FPS
    other_overhead = 10.0  # Estimated other processing time
    
    old_total = individual_time + other_overhead
    new_total = batched_time + other_overhead
    
    print(f"\nWith {other_overhead:.1f}ms other processing:")
    print(f"  Old method total: {old_total:.1f}ms → {1000/old_total:.1f} FPS")
    print(f"  New method total: {new_total:.1f}ms → {1000/new_total:.1f} FPS")
    
    # Get performance stats
    stats = get_batched_fft_processor().get_performance_stats()
    print(f"\nBatched FFT Stats:")
    print(f"  GPU Enabled: {stats['gpu_enabled']}")
    if stats['gpu_enabled']:
        print(f"  GPU Memory Used: {stats.get('gpu_memory_used_mb', 0):.1f} MB")

if __name__ == "__main__":
    main()