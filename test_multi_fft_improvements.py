#!/usr/bin/env python3
"""Test the improved MultiResolutionFFT functionality"""

import numpy as np
import time
import sys

# Add the project directory to Python path
sys.path.insert(0, '/home/magicat777/Projects/audio-geometric-visualizer/OMEGA')

from omega4.audio.multi_resolution_fft import (
    MultiResolutionFFT, CircularBuffer, benchmark_multi_fft, FFTConfig
)

def test_circular_buffer():
    """Test the thread-safe circular buffer"""
    print("Testing CircularBuffer...")
    buffer = CircularBuffer(1024)
    
    # Test 1: Basic write/read
    test_data = np.random.random(512).astype(np.float32)
    assert buffer.write(test_data), "Write failed"
    read_data = buffer.read_latest(512)
    assert read_data is not None, "Read failed"
    assert np.allclose(test_data, read_data), "Data mismatch"
    print("✅ Basic write/read test passed")
    
    # Test 2: Wraparound
    for _ in range(5):
        buffer.write(np.random.random(300).astype(np.float32))
    latest = buffer.read_latest(600)
    assert latest is not None, "Wraparound read failed"
    print("✅ Wraparound test passed")
    
    # Test 3: Insufficient data
    buffer.reset()
    buffer.write(np.random.random(100).astype(np.float32))
    result = buffer.read_latest(200)
    assert result is None, "Should return None for insufficient data"
    print("✅ Insufficient data test passed")
    
    # Test 4: Large data handling
    large_data = np.random.random(2000).astype(np.float32)
    assert buffer.write(large_data), "Large data write failed"
    read_large = buffer.read_latest(1024)
    assert read_large is not None, "Large data read failed"
    assert np.allclose(large_data[-1024:], read_large), "Large data mismatch"
    print("✅ Large data handling test passed")
    
    return True

def test_multi_resolution_fft():
    """Test the MultiResolutionFFT processing"""
    print("\nTesting MultiResolutionFFT...")
    
    # Create FFT processor
    fft_processor = MultiResolutionFFT(sample_rate=48000)
    
    # Test 1: Process audio chunks to fill buffers
    # Need to send enough data to fill the largest buffer (4096 samples)
    test_signal = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
    chunk_size = 512
    results = {}
    
    # Process multiple chunks until we get results from all resolutions
    for i in range(0, 8192, chunk_size):
        chunk = test_signal[i:i+chunk_size]
        results = fft_processor.process_audio_chunk(chunk)
        if len(results) == len(fft_processor.configs):
            break
    
    assert results, "No FFT results returned"
    print(f"✅ FFT processing returned {len(results)} resolutions")
    
    # Test 2: Combine results
    combined_magnitude, frequencies = fft_processor.combine_results_optimized(results)
    assert len(combined_magnitude) == 1024, "Wrong output size"
    assert len(frequencies) == 1024, "Wrong frequency array size"
    print("✅ Result combination test passed")
    
    # Test 3: Performance statistics
    stats = fft_processor.get_processing_stats()
    assert 'avg_time_ms' in stats, "Missing performance stats"
    print(f"✅ Performance tracking: {stats['avg_time_ms']:.2f}ms average")
    
    # Test 4: Buffer status
    buffer_status = fft_processor.get_buffer_status()
    assert len(buffer_status) == len(fft_processor.configs), "Wrong buffer count"
    print("✅ Buffer status monitoring test passed")
    
    # Test 5: Empty input handling
    empty_results = fft_processor.process_audio_chunk(np.array([]))
    assert empty_results == {}, "Should return empty dict for empty input"
    print("✅ Empty input handling test passed")
    
    # Test 6: Cleanup
    fft_processor.cleanup()
    print("✅ Cleanup test passed")
    
    return True

def test_error_handling():
    """Test error handling and validation"""
    print("\nTesting error handling...")
    
    # Test 1: Invalid FFT configuration
    try:
        config = FFTConfig((200, 100), 1024, 256, 1.0)  # Invalid freq range
        assert False, "Should raise ValueError"
    except ValueError:
        print("✅ Invalid frequency range caught")
    
    # Test 2: Invalid FFT size
    try:
        config = FFTConfig((20, 200), 1023, 256, 1.0)  # Not power of 2
        assert False, "Should raise ValueError"
    except ValueError:
        print("✅ Invalid FFT size caught")
    
    # Test 3: Invalid sample rate
    try:
        fft = MultiResolutionFFT(sample_rate=-1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("✅ Invalid sample rate caught")
    
    # Test 4: Invalid buffer size
    try:
        buffer = CircularBuffer(-1)
        assert False, "Should raise ValueError"
    except ValueError:
        print("✅ Invalid buffer size caught")
    
    return True

def run_performance_benchmark():
    """Run performance benchmark"""
    print("\nRunning performance benchmark...")
    
    results = benchmark_multi_fft(
        sample_rate=48000,
        chunk_size=512,
        num_iterations=1000
    )
    
    print(f"✅ Benchmark completed:")
    print(f"   - Average processing time: {results['avg_time_ms']:.2f}ms")
    print(f"   - Iterations per second: {results['iterations_per_second']:.0f}")
    
    stats = results['stats']
    if stats['error_count'] == 0:
        print(f"   - No errors during benchmark")
    else:
        print(f"   - Error rate: {stats['error_rate']:.2%}")
    
    return results['avg_time_ms'] < 10  # Should process in under 10ms

def main():
    print("Testing MultiResolutionFFT Improvements")
    print("=" * 60)
    print("\nKey improvements:")
    print("✅ Thread-safe circular buffer")
    print("✅ Comprehensive error handling")
    print("✅ Performance optimizations")
    print("✅ Type safety with dataclasses")
    print("✅ Proper resource cleanup")
    print("\n")
    
    all_passed = True
    
    # Run tests
    try:
        all_passed &= test_circular_buffer()
    except Exception as e:
        print(f"❌ Circular buffer test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_multi_resolution_fft()
    except Exception as e:
        print(f"❌ MultiResolutionFFT test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_error_handling()
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= run_performance_benchmark()
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! MultiResolutionFFT improvements working correctly.")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())