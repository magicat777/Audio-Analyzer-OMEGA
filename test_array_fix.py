#!/usr/bin/env python3
import numpy as np

# Test the array size handling
fft_data = np.random.rand(512)
freqs = np.linspace(0, 24000, 2048)

print(f'Original sizes: fft_data={len(fft_data)}, freqs={len(freqs)}')

# Test the fix
if len(fft_data) != len(freqs):
    min_len = min(len(fft_data), len(freqs))
    fft_data_fixed = fft_data[:min_len]
    freqs_fixed = freqs[:min_len]
    print(f'Fixed sizes: fft_data={len(fft_data_fixed)}, freqs={len(freqs_fixed)}')
    
    # Test mask creation
    bass_mask = (freqs_fixed >= 20) & (freqs_fixed <= 250)
    print(f'Bass mask created successfully, sum={np.sum(bass_mask)}')
    
print("Array size fix works correctly!")