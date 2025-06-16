#!/usr/bin/env python3
"""Test debug snapshot functionality with simulated audio"""

import numpy as np
import time
from collections import deque
from omega4_main import ProfessionalLiveAudioAnalyzer

def test_debug_snapshot():
    print("Testing debug snapshot functionality...")
    
    # Create analyzer instance
    analyzer = ProfessionalLiveAudioAnalyzer()
    
    # Simulate audio data with multiple frequency components
    sample_rate = 48000
    duration = 0.1  # 100ms of audio
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test signal with various frequencies
    audio_data = np.zeros_like(t)
    
    # Bass frequencies
    audio_data += 0.3 * np.sin(2 * np.pi * 60 * t)   # Kick drum
    audio_data += 0.2 * np.sin(2 * np.pi * 100 * t)  # Bass line
    audio_data += 0.1 * np.sin(2 * np.pi * 40 * t)   # Sub bass
    
    # Mid frequencies
    audio_data += 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
    audio_data += 0.2 * np.sin(2 * np.pi * 880 * t)  # A5
    audio_data += 0.1 * np.sin(2 * np.pi * 1320 * t) # E6
    
    # High frequencies
    audio_data += 0.1 * np.sin(2 * np.pi * 3000 * t)
    audio_data += 0.05 * np.sin(2 * np.pi * 5000 * t)
    audio_data += 0.05 * np.sin(2 * np.pi * 10000 * t)
    
    # Add some noise
    audio_data += 0.01 * np.random.randn(len(t))
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.5
    
    # Process the audio
    spectrum = analyzer.process_multi_resolution_fft(audio_data.astype(np.float32))
    
    # Create full spectrum data
    fft_result = np.fft.rfft(audio_data)
    full_spectrum = np.abs(fft_result)
    
    # Apply band mapping to get band values
    band_values = np.zeros(len(analyzer.band_indices))
    for i, (start, end) in enumerate(analyzer.band_indices):
        if end <= len(full_spectrum):
            # Use power scaling (square root of mean square)
            band_values[i] = np.sqrt(np.mean(full_spectrum[start:end]**2))
    
    # Normalize band values
    if np.max(band_values) > 0:
        band_values = band_values / np.max(band_values)
    
    # Create spectrum data dict
    spectrum_data = {
        'spectrum': full_spectrum,
        'band_values': band_values,
        'fft_complex': fft_result
    }
    
    # Enable bass zoom and update bass panel with test data
    analyzer.show_bass_zoom = True
    analyzer.bass_zoom_panel.bass_bar_values = np.random.rand(31) * 0.5  # Simulate bass detail
    
    # Set some test values for professional meters
    analyzer.professional_meters_panel.momentary_lufs = -16.5
    analyzer.professional_meters_panel.short_term_lufs = -17.2
    analyzer.professional_meters_panel.integrated_lufs = -18.1
    analyzer.professional_meters_panel.loudness_range = 8.5
    analyzer.professional_meters_panel.true_peak_db = -1.2
    
    # Set pitch detection
    analyzer.pitch_detection_panel.detected_pitch = "A4 (440.0 Hz)"
    analyzer.pitch_detection_panel.pitch_confidence = 85.0
    
    # Add some transient events
    current_time = time.time()
    analyzer.transient_events.append({'type': 'kick', 'time': current_time - 0.5, 'magnitude': 0.8})
    analyzer.transient_events.append({'type': 'snare', 'time': current_time - 0.3, 'magnitude': 0.6})
    analyzer.transient_events.append({'type': 'hihat', 'time': current_time - 0.1, 'magnitude': 0.4})
    
    # Print debug snapshot
    print("\n" + "="*80)
    print("CALLING DEBUG SNAPSHOT")
    print("="*80)
    analyzer.print_debug_snapshot(spectrum_data)
    
    print("\nDebug snapshot test completed!")
    
    # Cleanup
    analyzer.running = False

if __name__ == "__main__":
    test_debug_snapshot()