"""
Multi-Resolution FFT Module for OMEGA-4 Audio Analyzer
Phase 5: Extract FFT processing with psychoacoustic weighting
"""

import numpy as np
from typing import Dict, List, Tuple


class MultiResolutionFFT:
    """Multi-resolution FFT analysis for enhanced low-end detail"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Multi-resolution FFT configurations - optimized for performance
        self.fft_configs = [
            {
                "range": (20, 200),
                "fft_size": 4096,
                "hop_size": 1024,
                "weight": 1.5,
            },  # Good resolution bass
            {
                "range": (200, 1000),
                "fft_size": 2048,
                "hop_size": 512,
                "weight": 1.2,
            },  # Efficient low-mids
            {"range": (1000, 5000), "fft_size": 1024, "hop_size": 256, "weight": 1.0},  # Fast mids
            {
                "range": (5000, 20000),
                "fft_size": 1024,
                "hop_size": 256,
                "weight": 1.5,
            },  # Fast highs
        ]

        # Pre-compute windows for each configuration
        self.windows = {}
        for i, config in enumerate(self.fft_configs):
            fft_size = config["fft_size"]
            # Use Blackman-Harris window for better frequency resolution
            self.windows[i] = np.blackman(fft_size).astype(np.float32)

        # Audio buffers for each resolution
        self.audio_buffers = {}
        self.buffer_positions = {}
        for i, config in enumerate(self.fft_configs):
            fft_size = config["fft_size"]
            self.audio_buffers[i] = np.zeros(fft_size * 2, dtype=np.float32)
            self.buffer_positions[i] = 0

    def process_multi_resolution(self, audio_chunk: np.ndarray, apply_weighting: bool = True) -> Dict[int, np.ndarray]:
        """Process audio with multiple FFT resolutions"""
        results = {}

        for i, config in enumerate(self.fft_configs):
            fft_size = config["fft_size"]

            # Update ring buffer
            chunk_len = len(audio_chunk)
            buffer = self.audio_buffers[i]
            pos = self.buffer_positions[i]
            buffer_len = len(buffer)

            # Handle chunk larger than buffer
            if chunk_len >= buffer_len:
                # Take the last buffer_len samples
                buffer[:] = audio_chunk[-buffer_len:]
                self.buffer_positions[i] = 0
            elif pos + chunk_len <= buffer_len:
                buffer[pos : pos + chunk_len] = audio_chunk
                self.buffer_positions[i] = (pos + chunk_len) % buffer_len
            else:
                first_part = buffer_len - pos
                buffer[pos:] = audio_chunk[:first_part]
                buffer[: chunk_len - first_part] = audio_chunk[first_part:]
                self.buffer_positions[i] = chunk_len - first_part

            # Extract latest samples for FFT
            audio_data = np.zeros(fft_size, dtype=np.float32)
            
            # Check if we have enough data in the buffer
            total_samples = self.buffer_positions[i]
            if hasattr(self, '_total_samples_processed'):
                total_samples = self._total_samples_processed[i]
            else:
                self._total_samples_processed = [0] * len(self.fft_configs)
            
            self._total_samples_processed[i] += chunk_len
            
            if self._total_samples_processed[i] >= fft_size:
                if pos >= fft_size:
                    audio_data[:] = buffer[pos - fft_size : pos]
                elif pos > 0:
                    audio_data[-pos:] = buffer[:pos]
                    audio_data[:-pos] = buffer[-(fft_size - pos) :]
                else:
                    # Buffer wrapped around
                    audio_data[:] = buffer[-fft_size:]

            # Apply window and compute FFT
            windowed = audio_data * self.windows[i]
            fft_result = np.fft.rfft(windowed)
            magnitude = np.abs(fft_result)

            # Apply psychoacoustic weighting
            if apply_weighting:
                magnitude = self.apply_psychoacoustic_weighting(magnitude, config, i)

            results[i] = magnitude

        return results

    def apply_psychoacoustic_weighting(
        self, magnitude: np.ndarray, config: Dict, config_index: int
    ) -> np.ndarray:
        """Apply psychoacoustic weighting for perceptually accurate analysis"""
        freqs = np.fft.rfftfreq(config["fft_size"], 1 / self.sample_rate)
        weights = np.ones_like(magnitude)

        freq_range = config["range"]
        base_weight = config["weight"]

        # Apply frequency-dependent weighting
        for i, freq in enumerate(freqs):
            if freq_range[0] <= freq <= freq_range[1]:
                # Enhanced weighting for perceptually critical frequencies
                if 60 <= freq <= 120:  # Kick drum fundamental
                    weights[i] = base_weight * 1.8
                elif 200 <= freq <= 400:  # Voice fundamental
                    weights[i] = base_weight * 1.4
                elif 2000 <= freq <= 5000:  # Presence/clarity range
                    weights[i] = base_weight * 1.2
                elif 20 <= freq <= 80:  # Sub-bass
                    weights[i] = base_weight * 1.6
                else:
                    weights[i] = base_weight

        return magnitude * weights

    def get_frequency_arrays(self) -> Dict[int, np.ndarray]:
        """Get frequency arrays for each FFT configuration"""
        freq_arrays = {}
        for i, config in enumerate(self.fft_configs):
            freq_arrays[i] = np.fft.rfftfreq(config["fft_size"], 1 / self.sample_rate)
        return freq_arrays

    def combine_fft_results(
        self, multi_results: Dict[int, np.ndarray], target_bins: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multi-resolution FFT results into single spectrum
        
        Returns:
            Tuple of (magnitude_array, frequency_array)
        """
        # Get frequency arrays for each resolution
        freq_arrays = self.get_frequency_arrays()
        
        # Create target frequency array
        max_freq = self.nyquist
        target_freqs = np.linspace(0, max_freq, target_bins)
        combined_magnitude = np.zeros(target_bins)
        weight_sum = np.zeros(target_bins)
        
        # Combine results from each resolution
        for i, magnitude in multi_results.items():
            config = self.fft_configs[i]
            freqs = freq_arrays[i]
            freq_range = config["range"]
            
            # Find valid frequency indices
            valid_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            valid_freqs = freqs[valid_mask]
            valid_magnitude = magnitude[valid_mask]
            
            # Interpolate to target frequency bins
            if len(valid_freqs) > 1:
                # For each target frequency, find contribution from this resolution
                for j, target_freq in enumerate(target_freqs):
                    if freq_range[0] <= target_freq <= freq_range[1]:
                        # Find nearest frequencies in this resolution
                        idx = np.searchsorted(valid_freqs, target_freq)
                        
                        if idx == 0:
                            # Use first value
                            combined_magnitude[j] += valid_magnitude[0] * config["weight"]
                            weight_sum[j] += config["weight"]
                        elif idx >= len(valid_freqs):
                            # Use last value
                            combined_magnitude[j] += valid_magnitude[-1] * config["weight"]
                            weight_sum[j] += config["weight"]
                        else:
                            # Linear interpolation
                            f1 = valid_freqs[idx - 1]
                            f2 = valid_freqs[idx]
                            m1 = valid_magnitude[idx - 1]
                            m2 = valid_magnitude[idx]
                            
                            # Interpolation factor
                            t = (target_freq - f1) / (f2 - f1)
                            interpolated = m1 * (1 - t) + m2 * t
                            
                            combined_magnitude[j] += interpolated * config["weight"]
                            weight_sum[j] += config["weight"]
        
        # Normalize by weight sum
        valid_weights = weight_sum > 0
        combined_magnitude[valid_weights] /= weight_sum[valid_weights]
        
        return combined_magnitude, target_freqs

    def reset_buffers(self):
        """Reset all audio buffers"""
        for i in range(len(self.fft_configs)):
            self.audio_buffers[i].fill(0)
            self.buffer_positions[i] = 0