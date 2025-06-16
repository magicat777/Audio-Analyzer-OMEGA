#!/usr/bin/env python3
"""Test frequency response of the analyzer"""

import numpy as np
from omega4_main import ProfessionalLiveAudioAnalyzer

def test_frequency_response():
    """Test if all frequencies are properly detected"""
    print("Testing frequency response...")
    
    # Create analyzer
    analyzer = ProfessionalLiveAudioAnalyzer()
    analyzer.freq_compensation_enabled = True
    analyzer.normalization_enabled = True
    
    # Test frequencies across the spectrum
    test_freqs = [
        50,    # Sub-bass
        100,   # Bass
        250,   # Upper bass
        500,   # Low mid
        1000,  # Mid
        2000,  # Upper mid  
        4000,  # Presence
        8000,  # Brilliance
        12000, # Air
        16000  # Ultra-high
    ]
    
    sample_rate = 48000
    duration = 0.1
    
    for test_freq in test_freqs:
        print(f"\nTesting {test_freq}Hz:")
        
        # Generate test tone
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * test_freq * t)
        
        # Process through analyzer
        analyzer.ring_buffer[:len(audio_data)] = audio_data
        spectrum_data = analyzer.process_audio_spectrum()
        
        if spectrum_data:
            # Find which bands contain this frequency
            freq_bin_width = sample_rate / (2 * len(spectrum_data['spectrum']))
            target_bin = int(test_freq / freq_bin_width)
            
            # Check spectrum value at this frequency
            if target_bin < len(spectrum_data['spectrum']):
                spectrum_val = spectrum_data['spectrum'][target_bin]
                print(f"  Spectrum value: {spectrum_val:.4f}")
            
            # Find corresponding bar
            for i, (start, end) in enumerate(analyzer.band_indices):
                if start <= target_bin < end:
                    if i < len(spectrum_data['band_values']):
                        bar_val = spectrum_data['band_values'][i]
                        print(f"  Bar {i}: value = {bar_val:.4f}")
                        break
            
            # Check frequency distribution
            freq_ranges = [
                (60, 250, "Bass"),
                (250, 500, "Low-mid"), 
                (500, 2000, "Mid"),
                (2000, 4000, "High-mid"),
                (4000, 6000, "Presence"),
                (6000, 10000, "Brilliance"),
                (10000, 20000, "Air")
            ]
            
            for low, high, name in freq_ranges:
                if low <= test_freq <= high:
                    # Calculate energy in this range
                    low_bin = int(low / freq_bin_width)
                    high_bin = int(high / freq_bin_width)
                    if high_bin <= len(spectrum_data['spectrum']):
                        avg_energy = np.mean(spectrum_data['spectrum'][low_bin:high_bin])
                        print(f"  {name} range average: {avg_energy:.4f}")
    
    # Cleanup
    analyzer.running = False
    print("\nFrequency response test completed!")

if __name__ == "__main__":
    test_frequency_response()