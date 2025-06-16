#!/usr/bin/env python3
"""Debug MultiResolutionFFT issue"""

import numpy as np
import sys
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

sys.path.insert(0, '/home/magicat777/Projects/audio-geometric-visualizer/OMEGA')

from omega4.audio.multi_resolution_fft import MultiResolutionFFT

def debug_fft():
    print("Debugging MultiResolutionFFT...")
    
    # Create FFT processor
    fft_processor = MultiResolutionFFT(sample_rate=48000)
    
    # Create test signal - 440Hz sine wave
    duration = 1.0  # 1 second
    t = np.arange(0, duration, 1/48000)
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print(f"Test audio shape: {test_audio.shape}")
    print(f"Number of FFT configs: {len(fft_processor.configs)}")
    
    # Process small chunks to fill buffers
    chunk_size = 512
    for i in range(0, len(test_audio), chunk_size):
        chunk = test_audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            results = fft_processor.process_audio_chunk(chunk)
            print(f"Chunk {i//chunk_size}: {len(results)} results")
            
            if results:
                for idx, result in results.items():
                    print(f"  Config {idx}: magnitude shape = {result.magnitude.shape}")
                break
    
    # Check buffer status
    buffer_status = fft_processor.get_buffer_status()
    print("\nBuffer status:")
    for idx, status in buffer_status.items():
        print(f"  Buffer {idx}: {status}")

if __name__ == "__main__":
    debug_fft()