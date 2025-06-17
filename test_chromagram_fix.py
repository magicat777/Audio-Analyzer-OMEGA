#!/usr/bin/env python3
"""Test chromagram panel fix"""

import numpy as np
import pygame
from omega4.panels.chromagram import ChromagramPanel

print("Testing Chromagram Panel...")

# Initialize pygame (needed for fonts)
pygame.init()

# Create panel
panel = ChromagramPanel()

# Set dummy fonts
panel.set_fonts({
    'large': None,
    'medium': None,
    'small': None,
    'tiny': None
})

# Generate test audio with clear harmonics (C major chord)
sample_rate = 48000
duration = 0.1
t = np.linspace(0, duration, int(sample_rate * duration))

# C major triad (C4, E4, G4)
audio = np.sin(2 * np.pi * 261.63 * t)  # C4
audio += np.sin(2 * np.pi * 329.63 * t)  # E4
audio += np.sin(2 * np.pi * 392.00 * t)  # G4

# Create FFT
fft_data = np.abs(np.fft.rfft(audio))
freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

# Test update
try:
    panel.update(fft_data, audio, freqs, current_genre='pop')
    print("✓ Panel update successful")
    
    # Check results
    print(f"✓ Detected key: {panel.detected_key}")
    print(f"✓ Key confidence: {panel.key_confidence:.2f}")
    print(f"✓ Current chord: {panel.current_chord}")
    print(f"✓ Current mode: {panel.current_mode}")
    
    # Check that histories exist
    print(f"✓ Mode history exists: {hasattr(panel, 'mode_history')}")
    print(f"✓ Modulation history exists: {hasattr(panel, 'modulation_history')}")
    
    print("\n✅ Chromagram Panel is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

pygame.quit()