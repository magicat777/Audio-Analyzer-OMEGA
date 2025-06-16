#!/usr/bin/env python3
"""
Performance Test Suite for Pitch Detection Optimization
Tests the improvements made to pitch detection in OMEGA-4
"""

import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omega4.panels.pitch_detection import CepstralAnalyzer, PitchDetectionPanel
from omega4.panels.pitch_detection_config import PitchDetectionConfig, get_preset_config
from omega4.panels.memory_pool import get_global_pool, with_pooled_array


def generate_test_signal(frequency: float, duration: float, sample_rate: int = 48000, 
                        add_noise: bool = False, add_harmonics: bool = False) -> np.ndarray:
    """Generate test signal with known frequency"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Base sine wave
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add harmonics if requested
    if add_harmonics:
        signal += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # 2nd harmonic
        signal += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # 3rd harmonic
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 0.1, len(signal))
        signal += noise
    
    return signal.astype(np.float32)


def test_yin_performance():
    """Test YIN algorithm performance: standard vs vectorized"""
    print("\n=== YIN Algorithm Performance Test ===")
    
    # Create two analyzers - one with vectorization, one without
    config_vec = PitchDetectionConfig(yin_use_vectorization=True)
    config_std = PitchDetectionConfig(yin_use_vectorization=False)
    
    analyzer_vec = CepstralAnalyzer(48000, config_vec)
    analyzer_std = CepstralAnalyzer(48000, config_std)
    
    # Test signals
    test_frequencies = [110, 220, 440, 880]  # A2, A3, A4, A5
    window_size = 2048  # Larger window for better detection
    
    for freq in test_frequencies:
        signal = generate_test_signal(freq, window_size / 48000, add_harmonics=True)
        
        # Test standard implementation
        start = time.perf_counter()
        for _ in range(100):
            pitch_std, conf_std = analyzer_std.yin_pitch_detection(signal)
        time_std = (time.perf_counter() - start) / 100 * 1000  # ms
        
        # Test vectorized implementation
        start = time.perf_counter()
        for _ in range(100):
            pitch_vec, conf_vec = analyzer_vec.yin_pitch_detection(signal)
        time_vec = (time.perf_counter() - start) / 100 * 1000  # ms
        
        speedup = time_std / time_vec
        error_std = abs(pitch_std - freq) / freq * 100 if pitch_std > 0 else 100
        error_vec = abs(pitch_vec - freq) / freq * 100 if pitch_vec > 0 else 100
        
        print(f"\nFrequency: {freq}Hz")
        print(f"  Standard: {time_std:.2f}ms, detected {pitch_std:.1f}Hz (error: {error_std:.1f}%)")
        print(f"  Vectorized: {time_vec:.2f}ms, detected {pitch_vec:.1f}Hz (error: {error_vec:.1f}%)")
        print(f"  Speedup: {speedup:.2f}x")


def test_memory_pool_performance():
    """Test memory pool performance vs standard allocation"""
    print("\n=== Memory Pool Performance Test ===")
    
    pool = get_global_pool()
    sizes = [512, 1024, 2048, 4096]
    iterations = 1000
    
    for size in sizes:
        # Test standard allocation
        start = time.perf_counter()
        for _ in range(iterations):
            arr = np.zeros(size, dtype=np.float32)
            arr[0] = 1.0  # Touch memory
        time_std = (time.perf_counter() - start) * 1000  # ms
        
        # Test pool allocation
        start = time.perf_counter()
        for _ in range(iterations):
            arr = pool.get_array(size)
            arr[0] = 1.0  # Touch memory
            pool.return_array(arr)
        time_pool = (time.perf_counter() - start) * 1000  # ms
        
        speedup = time_std / time_pool
        print(f"\nSize {size}:")
        print(f"  Standard allocation: {time_std:.2f}ms")
        print(f"  Pool allocation: {time_pool:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Print pool statistics
    stats = pool.get_stats()
    print(f"\nPool Statistics:")
    print(f"  Total allocations: {stats['allocations']}")
    print(f"  Reuses: {stats['reuses']}")
    print(f"  Reuse rate: {stats['reuse_rate']:.1%}")


def test_filter_caching():
    """Test filter coefficient caching performance"""
    print("\n=== Filter Caching Performance Test ===")
    
    # Analyzer with caching
    config_cached = PitchDetectionConfig(cache_filter_coefficients=True)
    analyzer_cached = CepstralAnalyzer(48000, config_cached)
    
    # Analyzer without caching
    config_uncached = PitchDetectionConfig(cache_filter_coefficients=False)
    analyzer_uncached = CepstralAnalyzer(48000, config_uncached)
    
    signal = generate_test_signal(440, 0.1, add_noise=True)
    
    # Test uncached
    start = time.perf_counter()
    for _ in range(100):
        result = analyzer_uncached.detect_pitch_advanced(signal)
    time_uncached = (time.perf_counter() - start) / 100 * 1000  # ms
    
    # Test cached
    start = time.perf_counter()
    for _ in range(100):
        result = analyzer_cached.detect_pitch_advanced(signal)
    time_cached = (time.perf_counter() - start) / 100 * 1000  # ms
    
    speedup = time_uncached / time_cached
    print(f"Uncached filters: {time_uncached:.2f}ms per detection")
    print(f"Cached filters: {time_cached:.2f}ms per detection")
    print(f"Speedup: {speedup:.2f}x")


def test_profile_performance():
    """Test different configuration profiles"""
    print("\n=== Configuration Profile Performance Test ===")
    
    profiles = ['low_latency', 'balanced', 'high_accuracy']
    signal = generate_test_signal(440, 0.1, add_noise=True, add_harmonics=True)
    
    for profile_name in profiles:
        config = get_preset_config(profile_name)
        analyzer = CepstralAnalyzer(48000, config)
        
        # Measure performance
        start = time.perf_counter()
        results = []
        for _ in range(50):
            result = analyzer.detect_pitch_advanced(signal)
            results.append(result)
        avg_time = (time.perf_counter() - start) / 50 * 1000  # ms
        
        # Calculate accuracy
        detected_pitches = [r['pitch'] for r in results if r['confidence'] > 0.3]
        if detected_pitches:
            avg_pitch = np.mean(detected_pitches)
            error = abs(avg_pitch - 440) / 440 * 100
        else:
            avg_pitch = 0
            error = 100
        
        print(f"\nProfile: {profile_name}")
        print(f"  Processing time: {avg_time:.2f}ms")
        print(f"  Detected pitch: {avg_pitch:.1f}Hz (error: {error:.1f}%)")
        print(f"  Window size: {config.yin_window_size}")


def test_error_handling():
    """Test error handling and input validation"""
    print("\n=== Error Handling Test ===")
    
    analyzer = CepstralAnalyzer(48000)
    
    # Test various invalid inputs
    test_cases = [
        ("None input", None),
        ("Empty array", np.array([])),
        ("Wrong dtype", np.array([1, 2, 3], dtype=np.int32)),
        ("Contains NaN", np.array([1.0, np.nan, 3.0], dtype=np.float32)),
        ("Contains Inf", np.array([1.0, np.inf, 3.0], dtype=np.float32)),
        ("Too short", np.zeros(10, dtype=np.float32)),
        ("Silent signal", np.zeros(1024, dtype=np.float32))
    ]
    
    for name, signal in test_cases:
        result = analyzer.detect_pitch_advanced_safe(signal)
        print(f"\n{name}:")
        print(f"  Error: {result.get('error', 'None')}")
        print(f"  Pitch: {result['pitch']}")
        print(f"  Confidence: {result['confidence']}")


def test_complex_audio():
    """Test pitch detection on complex audio (music-like signals)"""
    print("\n=== Complex Audio Test ===")
    
    # Create a complex signal with multiple components
    duration = 0.5
    sample_rate = 48000
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Vocal-like signal (220Hz fundamental with vibrato)
    vibrato_freq = 5  # Hz
    vibrato_depth = 0.02  # 2% pitch variation
    vocal_freq = 220 * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_freq * t))
    vocal = np.sin(2 * np.pi * vocal_freq * t)
    vocal += 0.3 * np.sin(2 * np.pi * vocal_freq * 2 * t)  # 2nd harmonic
    
    # Add bass (80Hz)
    bass = 0.5 * np.sin(2 * np.pi * 80 * t)
    
    # Add high frequencies (simulating cymbals/hi-hats)
    high_freq = 0.1 * np.random.normal(0, 1, len(t))
    high_freq = high_freq * (1 + 0.5 * np.sin(2 * np.pi * 8 * t))  # Amplitude modulation
    
    # Combine
    complex_signal = (vocal + bass + high_freq).astype(np.float32)
    
    # Test with different configurations
    configs = {
        'voice_optimized': get_preset_config('voice_optimized'),
        'music_optimized': get_preset_config('music_optimized'),
        'balanced': get_preset_config('balanced')
    }
    
    for name, config in configs.items():
        analyzer = CepstralAnalyzer(sample_rate, config)
        
        # Process in chunks (like real-time)
        chunk_size = 1024
        pitches = []
        confidences = []
        
        for i in range(0, len(complex_signal) - chunk_size, chunk_size // 2):
            chunk = complex_signal[i:i + chunk_size]
            result = analyzer.detect_pitch_advanced_safe(chunk)
            if result['confidence'] > 0.3:
                pitches.append(result['pitch'])
                confidences.append(result['confidence'])
        
        if pitches:
            avg_pitch = np.mean(pitches)
            avg_conf = np.mean(confidences)
            error = abs(avg_pitch - 220) / 220 * 100
        else:
            avg_pitch = 0
            avg_conf = 0
            error = 100
        
        print(f"\n{name}:")
        print(f"  Detected pitch: {avg_pitch:.1f}Hz (target: 220Hz)")
        print(f"  Error: {error:.1f}%")
        print(f"  Average confidence: {avg_conf:.2f}")
        print(f"  Detection rate: {len(pitches)}/{len(range(0, len(complex_signal) - chunk_size, chunk_size // 2))}")


def main():
    """Run all performance tests"""
    print("=" * 60)
    print("Pitch Detection Performance Test Suite")
    print("=" * 60)
    
    test_yin_performance()
    test_memory_pool_performance()
    test_filter_caching()
    test_profile_performance()
    test_error_handling()
    test_complex_audio()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()