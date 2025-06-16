#!/usr/bin/env python3
"""Quick test to verify spectrum visualization"""

import numpy as np
import time

# Generate test audio signal
sample_rate = 48000
duration = 0.1  # 100ms
t = np.linspace(0, duration, int(sample_rate * duration))

# Create test signal with multiple frequencies
signal = np.sin(2 * np.pi * 440 * t)  # A4
signal += 0.5 * np.sin(2 * np.pi * 880 * t)  # A5
signal += 0.3 * np.sin(2 * np.pi * 220 * t)  # A3
signal += 0.2 * np.sin(2 * np.pi * 1760 * t)  # A6

# Add some noise
signal += 0.1 * np.random.randn(len(signal))

# Compute FFT
fft_result = np.fft.rfft(signal)
spectrum = np.abs(fft_result)
freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)

# Find peaks
print("Test signal analysis:")
print(f"Signal shape: {signal.shape}")
print(f"FFT shape: {spectrum.shape}")
print(f"Max magnitude: {np.max(spectrum):.2f}")
print(f"Mean magnitude: {np.mean(spectrum):.2f}")

# Find top frequencies
peak_indices = np.argsort(spectrum)[-10:][::-1]
print("\nTop frequencies:")
for idx in peak_indices[:5]:
    print(f"  {freqs[idx]:.1f} Hz: magnitude {spectrum[idx]:.2f}")

# Test downsampling to 512 bars
bars = 512
indices = np.linspace(0, len(spectrum) - 1, bars).astype(int)
downsampled = spectrum[indices]

print(f"\nDownsampled to {bars} bars:")
print(f"Shape: {downsampled.shape}")
print(f"Max: {np.max(downsampled):.2f}")
print(f"Non-zero: {np.count_nonzero(downsampled)}")