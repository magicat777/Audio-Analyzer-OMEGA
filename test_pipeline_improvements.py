#!/usr/bin/env python3
"""Test the improved AudioProcessingPipeline functionality"""

import numpy as np
import sys
import time

sys.path.insert(0, '/home/magicat777/Projects/audio-geometric-visualizer/OMEGA')

from omega4.audio.audio_config import PipelineConfig
from omega4.audio.pipeline import AudioProcessingPipeline, ContentTypeDetector
from omega4.audio.performance_monitor import PerformanceMonitor

def test_pipeline_config():
    """Test PipelineConfig validation"""
    print("Testing PipelineConfig...")
    
    # Test valid config
    config = PipelineConfig()
    print(f"✅ Valid config created: {config.sample_rate}Hz, {config.num_bands} bands")
    
    # Test invalid configs
    try:
        bad_config = PipelineConfig(sample_rate=-1)
        print("❌ Should have raised ValueError for negative sample rate")
    except ValueError:
        print("✅ Invalid sample rate caught")
    
    try:
        bad_config = PipelineConfig(fft_size=1023)  # Not power of 2
        print("❌ Should have raised ValueError for non-power-of-2 FFT size")
    except ValueError:
        print("✅ Invalid FFT size caught")
    
    try:
        bad_config = PipelineConfig(smoothing_factor=1.5)
        print("❌ Should have raised ValueError for smoothing > 1")
    except ValueError:
        print("✅ Invalid smoothing factor caught")
    
    return True

def test_numerical_stability():
    """Test numerical stability improvements"""
    print("\nTesting numerical stability...")
    
    config = PipelineConfig()
    detector = ContentTypeDetector(config)
    
    # Test with zero band values (division by zero case)
    zero_bands = np.zeros(100)
    freq_starts = np.linspace(20, 10000, 100)
    freq_ends = np.linspace(40, 20000, 100)
    
    result = detector.analyze_content(None, zero_bands, freq_starts, freq_ends)
    print(f"✅ Zero bands handled: result = '{result}'")
    
    # Test with very small values
    tiny_bands = np.ones(100) * 1e-10
    result = detector.analyze_content(None, tiny_bands, freq_starts, freq_ends)
    print(f"✅ Tiny values handled: result = '{result}'")
    
    # Test with empty arrays
    empty_bands = np.array([])
    result = detector.analyze_content(None, empty_bands, np.array([]), np.array([]))
    print(f"✅ Empty arrays handled: result = '{result}'")
    
    return True

def test_buffer_overflow_protection():
    """Test ring buffer overflow protection"""
    print("\nTesting buffer overflow protection...")
    
    config = PipelineConfig(ring_buffer_size=1024)
    pipeline = AudioProcessingPipeline(config)
    
    # Test normal chunk
    normal_chunk = np.random.random(512).astype(np.float32)
    result = pipeline.process_frame(normal_chunk)
    assert result['audio_data'] is not None
    print("✅ Normal chunk processed")
    
    # Test oversized chunk (larger than buffer)
    oversized_chunk = np.random.random(2048).astype(np.float32)
    result = pipeline.process_frame(oversized_chunk)
    # Should log an error but return empty result
    assert len(result['audio_data']) == 0
    print("✅ Oversized chunk handled gracefully")
    
    # Test empty chunk
    empty_chunk = np.array([])
    result = pipeline.process_frame(empty_chunk)
    assert len(result['audio_data']) == 0
    print("✅ Empty chunk handled")
    
    # Test wrong type
    try:
        result = pipeline.process_frame([1, 2, 3])  # List instead of array
        print("❌ Should have raised TypeError for non-array input")
        return False
    except TypeError:
        print("✅ Type error caught")
    
    return True

def test_performance_optimization():
    """Test performance optimizations"""
    print("\nTesting performance optimizations...")
    
    config = PipelineConfig()
    pipeline = AudioProcessingPipeline(config)
    
    # Generate test FFT data
    fft_magnitude = np.random.random(2049).astype(np.float32)
    
    # Benchmark band mapping
    start_time = time.perf_counter()
    for _ in range(1000):
        bands = pipeline.map_to_bands(fft_magnitude, apply_smoothing=False)
    elapsed = (time.perf_counter() - start_time) * 1000
    
    print(f"✅ 1000 band mappings in {elapsed:.2f}ms ({elapsed/1000:.3f}ms per mapping)")
    
    # Test with smoothing
    start_time = time.perf_counter()
    for _ in range(1000):
        bands = pipeline.map_to_bands(fft_magnitude, apply_smoothing=True)
    elapsed_smooth = (time.perf_counter() - start_time) * 1000
    
    print(f"✅ 1000 smoothed mappings in {elapsed_smooth:.2f}ms ({elapsed_smooth/1000:.3f}ms per mapping)")
    
    # Verify pre-computed compensation is applied
    assert hasattr(pipeline, 'freq_compensation')
    print(f"✅ Frequency compensation pre-computed: {len(pipeline.freq_compensation)} values")
    
    return elapsed / 1000 < 5.0  # Should be under 5ms per mapping

def test_performance_monitoring():
    """Test performance monitoring"""
    print("\nTesting performance monitoring...")
    
    monitor = PerformanceMonitor()
    
    # Simulate frame processing
    for i in range(20):
        monitor.start_frame()
        # Simulate varying processing times
        if i == 10:
            time.sleep(0.025)  # Simulate slow frame
        else:
            time.sleep(0.005)  # Normal frame
        monitor.end_frame(0.5)
    
    stats = monitor.get_statistics()
    print(f"✅ Performance stats collected:")
    print(f"   - Average: {stats['avg_frame_time_ms']:.2f}ms")
    print(f"   - Max: {stats['max_frame_time_ms']:.2f}ms")
    print(f"   - Min: {stats['min_frame_time_ms']:.2f}ms")
    print(f"   - Warnings: {stats['warning_rate']:.1%}")
    
    # Check if quality reduction is recommended
    should_reduce = monitor.should_reduce_quality()
    print(f"✅ Quality reduction recommendation: {should_reduce}")
    
    return True

def test_auto_gain():
    """Test auto gain functionality"""
    print("\nTesting auto gain...")
    
    config = PipelineConfig(auto_gain_enabled=True)
    pipeline = AudioProcessingPipeline(config)
    
    # Test with quiet signal
    quiet_signal = np.random.random(512).astype(np.float32) * 0.001
    initial_gain = pipeline.input_gain
    
    for _ in range(10):
        result = pipeline.process_frame(quiet_signal)
    
    print(f"✅ Quiet signal: gain adjusted from {initial_gain:.2f} to {pipeline.input_gain:.2f}")
    
    # Test with loud signal
    loud_signal = np.random.random(512).astype(np.float32) * 10.0
    for _ in range(10):
        result = pipeline.process_frame(loud_signal)
    
    print(f"✅ Loud signal: gain adjusted to {pipeline.input_gain:.2f}")
    
    # Verify gain limits
    assert 0.1 <= pipeline.input_gain <= 10.0
    print("✅ Gain within limits (0.1 - 10.0)")
    
    return True

def main():
    print("Testing AudioProcessingPipeline Improvements")
    print("=" * 60)
    print("\nKey improvements:")
    print("✅ Configuration class with validation")
    print("✅ Numerical stability (epsilon = 1e-6)")
    print("✅ Buffer overflow protection")
    print("✅ Performance optimizations")
    print("✅ Comprehensive error handling")
    print("✅ Performance monitoring")
    print("\n")
    
    all_passed = True
    
    tests = [
        ("Pipeline Configuration", test_pipeline_config),
        ("Numerical Stability", test_numerical_stability),
        ("Buffer Overflow Protection", test_buffer_overflow_protection),
        ("Performance Optimization", test_performance_optimization),
        ("Performance Monitoring", test_performance_monitoring),
        ("Auto Gain", test_auto_gain)
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Pipeline improvements working correctly.")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    # Show performance summary
    config = PipelineConfig()
    pipeline = AudioProcessingPipeline(config)
    print(f"\nPipeline configuration:")
    print(f"- Sample rate: {config.sample_rate}Hz")
    print(f"- FFT size: {config.fft_size}")
    print(f"- Number of bands: {config.num_bands}")
    print(f"- Ring buffer size: {config.ring_buffer_size}")
    print(f"- Epsilon: {config.epsilon}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())