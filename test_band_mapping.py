#!/usr/bin/env python3
"""Test band mapping fix"""

import numpy as np

# Simulate the band creation logic
n_bars = 1024

# Define frequency bands with musical distribution
bass_bars = int(n_bars * 0.25)      # 256
lowmid_bars = int(n_bars * 0.15)    # 153
mid_bars = int(n_bars * 0.25)       # 256
highmid_bars = int(n_bars * 0.20)   # 204
high_bars = n_bars - (bass_bars + lowmid_bars + mid_bars + highmid_bars)  # 155

print(f"Bar distribution:")
print(f"  Bass (20-250Hz): {bass_bars} bars")
print(f"  Low-mid (250-500Hz): {lowmid_bars} bars")
print(f"  Mid (500-2000Hz): {mid_bars} bars")
print(f"  High-mid (2000-6000Hz): {highmid_bars} bars")
print(f"  High (6000-20000Hz): {high_bars} bars")
print(f"  Total: {bass_bars + lowmid_bars + mid_bars + highmid_bars + high_bars} bars")

# Create frequency points
freq_points = []

# Bass: 20-250Hz (logarithmic)
bass_freqs = np.logspace(np.log10(20), np.log10(250), bass_bars, endpoint=False)
freq_points.extend(bass_freqs)
print(f"\nBass frequencies: {len(bass_freqs)} points, range {bass_freqs[0]:.1f}-{bass_freqs[-1]:.1f}Hz")

# Low-mid: 250-500Hz (logarithmic)
lowmid_freqs = np.logspace(np.log10(250), np.log10(500), lowmid_bars, endpoint=False)
freq_points.extend(lowmid_freqs)
print(f"Low-mid frequencies: {len(lowmid_freqs)} points, range {lowmid_freqs[0]:.1f}-{lowmid_freqs[-1]:.1f}Hz")

# Mid: 500-2000Hz (logarithmic)
mid_freqs = np.logspace(np.log10(500), np.log10(2000), mid_bars, endpoint=False)
freq_points.extend(mid_freqs)
print(f"Mid frequencies: {len(mid_freqs)} points, range {mid_freqs[0]:.1f}-{mid_freqs[-1]:.1f}Hz")

# High-mid: 2000-6000Hz (logarithmic)
highmid_freqs = np.logspace(np.log10(2000), np.log10(6000), highmid_bars, endpoint=False)
freq_points.extend(highmid_freqs)
print(f"High-mid frequencies: {len(highmid_freqs)} points, range {highmid_freqs[0]:.1f}-{highmid_freqs[-1]:.1f}Hz")

# High: 6000-20000Hz (logarithmic)
high_freqs = np.logspace(np.log10(6000), np.log10(20000), high_bars, endpoint=True)
freq_points.extend(high_freqs)
print(f"High frequencies: {len(high_freqs)} points, range {high_freqs[0]:.1f}-{high_freqs[-1]:.1f}Hz")

# Add final frequency point
freq_points.append(20000)

print(f"\nTotal frequency points: {len(freq_points)}")
print(f"Expected: {n_bars + 1}")

# Check first 10 frequency points
print("\nFirst 10 frequency points:")
for i in range(min(10, len(freq_points))):
    print(f"  Point {i}: {freq_points[i]:.1f} Hz")

# Map to FFT bins
FFT_SIZE_BASE = 4096
SAMPLE_RATE = 48000
freq_bin_width = SAMPLE_RATE / FFT_SIZE_BASE

print(f"\nFFT parameters:")
print(f"  FFT size: {FFT_SIZE_BASE}")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Bin width: {freq_bin_width:.1f} Hz")

# Show first 10 bands
print("\nFirst 10 bands:")
for i in range(min(10, n_bars)):
    if i >= len(freq_points) - 1:
        break
    start_freq = freq_points[i]
    end_freq = freq_points[i + 1]
    start_idx = int(start_freq / freq_bin_width)
    end_idx = int(end_freq / freq_bin_width)
    print(f"  Band {i}: {start_freq:.1f}-{end_freq:.1f} Hz (bins {start_idx}-{end_idx})")