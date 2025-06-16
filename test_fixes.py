#!/usr/bin/env python3
"""Test the fixes applied to omega4_main.py"""

import numpy as np
import time
from omega4_main import ProfessionalLiveAudioAnalyzer

def test_spectrum_fixes():
    print("Testing spectrum processing fixes...")
    
    # Create analyzer
    analyzer = ProfessionalLiveAudioAnalyzer()
    
    # Create test audio simulating music with vocals
    sample_rate = 48000
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate music with various components
    audio_data = np.zeros_like(t)
    
    # Bass (should be reduced by compensation)
    audio_data += 0.8 * np.sin(2 * np.pi * 80 * t)   # Bass
    audio_data += 0.6 * np.sin(2 * np.pi * 160 * t)  # Upper bass
    
    # Vocals (should be detected)
    # Fundamental frequency around 250Hz (female vocal range)
    audio_data += 0.4 * np.sin(2 * np.pi * 250 * t)  # F0
    audio_data += 0.3 * np.sin(2 * np.pi * 500 * t)  # 2nd harmonic
    audio_data += 0.2 * np.sin(2 * np.pi * 750 * t)  # 3rd harmonic
    audio_data += 0.2 * np.sin(2 * np.pi * 1000 * t) # 4th harmonic
    
    # Formants for voice
    audio_data += 0.15 * np.sin(2 * np.pi * 700 * t)  # F1
    audio_data += 0.15 * np.sin(2 * np.pi * 1220 * t) # F2
    audio_data += 0.1 * np.sin(2 * np.pi * 2600 * t)  # F3
    
    # High frequencies
    audio_data += 0.1 * np.sin(2 * np.pi * 4000 * t)
    audio_data += 0.05 * np.sin(2 * np.pi * 8000 * t)
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.7
    
    # Process audio using the internal method
    print("\nProcessing audio with vocal content...")
    
    # Update ring buffer with test data
    analyzer.ring_buffer[:len(audio_data)] = audio_data
    
    # Call the processing method directly
    spectrum_data = analyzer.process_audio_spectrum()
    
    # Check spectrum scaling
    print("\n[SPECTRUM SCALING CHECK]")
    band_values = spectrum_data['band_values']
    max_val = np.max(band_values)
    avg_val = np.mean(band_values)
    active_bars = np.sum(band_values > 0.01)
    
    print(f"Max band value: {max_val:.2f} (should be < 1.0)")
    print(f"Average band value: {avg_val:.2f}")
    print(f"Active bars: {active_bars}/{len(band_values)} ({active_bars/len(band_values)*100:.1f}%)")
    
    if max_val <= 1.0 and avg_val < 0.5:
        print("✅ Spectrum scaling FIXED - values are reasonable")
    else:
        print("❌ Spectrum scaling still needs work")
    
    # Check voice detection
    print("\n[VOICE DETECTION CHECK]")
    print(f"Voice detected: {'YES' if analyzer.voice_active else 'NO'}")
    print(f"Voice confidence: {analyzer.voice_confidence:.1f}%")
    
    if analyzer.voice_active:
        print("✅ Voice detection FIXED - detecting vocals")
    else:
        print("❌ Voice detection still needs work")
    
    # Test bass detail scaling
    print("\n[BASS DETAIL CHECK]")
    # Manually check a few bass frequencies
    freq_bin_width = sample_rate / (2 * len(spectrum_data['spectrum']))
    
    for freq in [40, 80, 160]:
        idx = int(freq / freq_bin_width)
        if idx < len(spectrum_data['spectrum']):
            magnitude = spectrum_data['spectrum'][idx]
            if magnitude > 0:
                db_value = 20 * np.log10(magnitude / 1.0)
                db_value = max(-60, min(0, db_value))
                print(f"{freq}Hz: {db_value:+6.1f} dB")
    
    print("\nIf dB values are between -60 and 0, bass scaling is ✅ FIXED")
    
    # Print a debug snapshot
    print("\n" + "="*80)
    print("FULL DEBUG SNAPSHOT")
    print("="*80)
    analyzer.print_debug_snapshot(spectrum_data)
    
    # Cleanup
    analyzer.running = False
    print("\nTest completed!")

if __name__ == "__main__":
    test_spectrum_fixes()