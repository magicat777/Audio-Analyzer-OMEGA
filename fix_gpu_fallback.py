"""
Update the GPU FFT processor to handle CUDA errors more gracefully
"""

import sys
import re

# Read the file
with open('omega4/optimization/gpu_accelerated_fft.py', 'r') as f:
    content = f.read()

# Find the compute_fft method and add better error handling
old_pattern = r'(\s+)(result = self\.gpu_processor\.compute_fft\(audio_data, fft_size\))'
new_code = r'''\1try:
\1    result = self.gpu_processor.compute_fft(audio_data, fft_size)
\1except Exception as e:
\1    # Silently fall back to CPU on any GPU error
\1    if self.gpu_available:
\1        self.gpu_available = False
\1    raise e'''

content = re.sub(old_pattern, new_code, content)

# Write back
with open('omega4/optimization/gpu_accelerated_fft.py', 'w') as f:
    f.write(content)

print("✓ Updated GPU error handling")

# Also update the batched processor to suppress repeated error messages
with open('omega4/optimization/batched_fft_processor.py', 'r') as f:
    content = f.read()

# Add a flag to track if we've already warned about GPU failure
if 'self.gpu_warning_shown = False' not in content:
    content = content.replace(
        'self.processor = processor',
        'self.processor = processor\n        self.gpu_warning_shown = False'
    )

# Update the error message to only show once
content = content.replace(
    'print(f"[BatchedFFT] GPU processing failed, falling back to CPU: {e}")',
    '''if not self.gpu_warning_shown:
                print(f"[BatchedFFT] GPU not available, using CPU processing")
                self.gpu_warning_shown = True'''
)

with open('omega4/optimization/batched_fft_processor.py', 'w') as f:
    f.write(content)

print("✓ Updated error message handling")
