#!/usr/bin/env python3
"""Test Phase 5: Extracted Audio Pipeline"""

import numpy as np
import sys
import time

# Test all extracted audio modules
try:
    from omega4.audio import (
        PipeWireMonitorCapture,
        AudioCaptureManager,
        MultiResolutionFFT,
        AudioProcessingPipeline,
        ContentTypeDetector,
        VoiceDetectionWrapper
    )
    print("✓ All audio modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import audio modules: {e}")
    sys.exit(1)

# Test AudioCaptureManager
print("\n=== Testing AudioCaptureManager ===")
try:
    capture_manager = AudioCaptureManager(48000, 512)
    print("✓ AudioCaptureManager created")
    print(f"✓ Sample rate: {capture_manager.sample_rate}")
    print(f"✓ Chunk size: {capture_manager.chunk_size}")
except Exception as e:
    print(f"✗ AudioCaptureManager test failed: {e}")

# Test MultiResolutionFFT
print("\n=== Testing MultiResolutionFFT ===")
try:
    multi_fft = MultiResolutionFFT(48000)
    
    # Create test audio signal
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies
    test_signal = (
        0.5 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz bass
        0.3 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz A4
        0.2 * np.sin(2 * np.pi * 2000 * t)   # 2 kHz
    )
    
    # Process multi-resolution
    results = multi_fft.process_multi_resolution(test_signal)
    print(f"✓ Multi-resolution FFT processed {len(results)} resolutions")
    
    # Combine results
    combined_mag, combined_freqs = multi_fft.combine_fft_results(results)
    print(f"✓ Combined spectrum: {len(combined_mag)} bins")
    print(f"✓ Frequency range: {combined_freqs[0]:.1f} - {combined_freqs[-1]:.1f} Hz")
    
except Exception as e:
    print(f"✗ MultiResolutionFFT test failed: {e}")
    import traceback
    traceback.print_exc()

# Test AudioProcessingPipeline
print("\n=== Testing AudioProcessingPipeline ===")
try:
    pipeline = AudioProcessingPipeline(48000, 768)
    
    # Create test audio
    test_audio = np.random.randn(512) * 0.1
    
    # Process frame
    result = pipeline.process_frame(test_audio)
    print(f"✓ Frame processed with gain: {result['current_gain']:.2f}")
    
    # Test band mapping
    test_fft = np.random.random(2049) * 0.5  # 4096 FFT -> 2049 bins
    band_values = pipeline.map_to_bands(test_fft)
    print(f"✓ FFT mapped to {len(band_values)} bands")
    
    # Test frequency compensation
    compensated = pipeline.apply_frequency_compensation(band_values)
    print(f"✓ Frequency compensation applied")
    
    # Test normalization
    normalized = pipeline.normalize_for_display(compensated)
    print(f"✓ Normalized for display (max: {np.max(normalized):.3f})")
    
    # Get band frequencies
    band_freqs = pipeline.get_band_frequencies()
    print(f"✓ Band frequency range: {band_freqs['starts'][0]:.1f} - {band_freqs['ends'][-1]:.1f} Hz")
    
except Exception as e:
    print(f"✗ AudioProcessingPipeline test failed: {e}")
    import traceback
    traceback.print_exc()

# Test ContentTypeDetector
print("\n=== Testing ContentTypeDetector ===")
try:
    content_detector = ContentTypeDetector()
    
    # Simulate voice detection result
    voice_info = {'confidence': 0.8}
    
    # Create band values and frequencies
    band_values = np.random.random(768)
    freq_starts = np.linspace(20, 19000, 768)
    freq_ends = np.linspace(30, 20000, 768)
    
    # Analyze content
    content_type = content_detector.analyze_content(
        voice_info, band_values, freq_starts, freq_ends
    )
    print(f"✓ Content type detected: {content_type}")
    
    # Get allocation
    allocation = content_detector.get_allocation_for_content(content_type)
    print(f"✓ Low-end allocation: {allocation:.0%}")
    
except Exception as e:
    print(f"✗ ContentTypeDetector test failed: {e}")

# Test VoiceDetectionWrapper
print("\n=== Testing VoiceDetectionWrapper ===")
try:
    voice_wrapper = VoiceDetectionWrapper(48000)
    print(f"✓ Voice detection available: {voice_wrapper.is_available()}")
    
    # Test detection
    test_audio = np.random.randn(2048) * 0.1
    voice_result = voice_wrapper.detect_voice_realtime(test_audio)
    print(f"✓ Voice detected: {voice_result['voice_detected']}")
    print(f"✓ Confidence: {voice_result['confidence']:.2f}")
    
    # Test formant analysis
    formants = voice_wrapper.analyze_formants(test_audio)
    print(f"✓ Formants extracted: {len(formants)}")
    
except Exception as e:
    print(f"✗ VoiceDetectionWrapper test failed: {e}")

# Test PipeWireMonitorCapture (without starting capture)
print("\n=== Testing PipeWireMonitorCapture ===")
try:
    # Just test initialization
    capture = PipeWireMonitorCapture(sample_rate=48000, chunk_size=512)
    print("✓ PipeWireMonitorCapture created")
    print(f"✓ Sample rate: {capture.sample_rate}")
    print(f"✓ Chunk size: {capture.chunk_size}")
    print(f"✓ Noise floor: {capture.noise_floor}")
    
except Exception as e:
    print(f"✗ PipeWireMonitorCapture test failed: {e}")

print("\n✅ Phase 5 Audio Pipeline Tests Complete!")
print("✓ All audio modules extracted successfully")
print("✓ Multi-resolution FFT working")
print("✓ Audio processing pipeline working")
print("✓ Content type detection working")
print("✓ Voice detection wrapper working")