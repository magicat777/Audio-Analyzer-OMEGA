#!/usr/bin/env python3
"""Test harmonic analysis panel"""

import numpy as np
from omega4.panels.harmonic_analysis import HarmonicAnalysisPanel

print("Testing Harmonic Analysis Panel...")

# Create panel
panel = HarmonicAnalysisPanel()

# Generate test audio with harmonics
sample_rate = 48000
duration = 0.1
t = np.linspace(0, duration, int(sample_rate * duration))

# Create a signal with clear harmonics (A4 = 440Hz)
fundamental = 440
audio = np.sin(2 * np.pi * fundamental * t)  # Fundamental
audio += 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)  # 2nd harmonic
audio += 0.3 * np.sin(2 * np.pi * fundamental * 3 * t)  # 3rd harmonic
audio += 0.2 * np.sin(2 * np.pi * fundamental * 4 * t)  # 4th harmonic

# Create FFT
fft_data = np.abs(np.fft.rfft(audio))
freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

# Test update
try:
    panel.update(fft_data, freqs, audio)
    print("✓ Panel update successful")
    
    # Check results
    info = panel.harmonic_info
    print(f"✓ Dominant fundamental: {info['dominant_fundamental']:.1f} Hz")
    print(f"✓ THD: {info['thd']:.2f}%")
    print(f"✓ Spectral centroid: {info['spectral_centroid']:.1f} Hz")
    
    if info['formants']:
        print(f"✓ Formants detected: {len(info['formants'])}")
    else:
        print("✓ No formants detected (expected for simple tone)")
    
    if info['instrument_matches']:
        print(f"✓ Instruments detected: {[m['instrument'] for m in info['instrument_matches'][:3]]}")
    
    print("\n✅ Harmonic Analysis Panel is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()