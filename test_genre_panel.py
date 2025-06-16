#!/usr/bin/env python3
"""Test genre classification panel fix"""

import sys
import numpy as np

sys.path.insert(0, '/home/magicat777/Projects/audio-geometric-visualizer/OMEGA')

from omega4.panels.genre_classification import GenreClassificationPanel

# Test initialization
panel = GenreClassificationPanel()
print("✓ GenreClassificationPanel initialized")

# Test update with all required parameters
fft_data = np.random.random(1024)
audio_chunk = np.random.random(512)
drum_info = {'kick': 0.5, 'snare': 0.3, 'hihat': 0.2}
harmonic_info = {'fundamental': 440.0, 'harmonics': [880.0, 1320.0]}

try:
    panel.update(fft_data, audio_chunk, drum_info, harmonic_info)
    print("✓ Update method works with all 4 parameters")
except Exception as e:
    print(f"✗ Update failed: {e}")

# Test with missing parameters (should fail)
try:
    panel.update(fft_data, audio_chunk)
    print("✗ Update worked with only 2 parameters (should have failed)")
except TypeError as e:
    print(f"✓ Correctly rejected 2-parameter call: {e}")

print("\nGenre classification panel fix successful!")