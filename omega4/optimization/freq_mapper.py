"""
Pre-computed frequency mappings for performance optimization
Avoids recalculating frequency-to-bin mappings every frame
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class FrequencyMapping:
    """Pre-computed frequency mapping data"""
    band_indices: List[Tuple[int, int]]
    freq_to_bin: np.ndarray
    bin_to_freq: np.ndarray
    mel_scale_factors: np.ndarray
    compensation_curve: np.ndarray
    frequency_points: np.ndarray
    
    
class PrecomputedFrequencyMapper:
    """Pre-computes and caches frequency mappings for performance"""
    
    def __init__(self, sample_rate: int, fft_size: int, num_bars: int):
        """
        Initialize frequency mapper with pre-computed mappings
        
        Args:
            sample_rate: Audio sample rate
            fft_size: FFT size
            num_bars: Number of frequency bars
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.num_bars = num_bars
        self.freq_bin_width = sample_rate / fft_size
        
        # Pre-compute all mappings
        self.mapping = self._precompute_all()
        
        # Cache for interpolation indices
        self.interp_cache = {}
        
        logger.info(f"Pre-computed frequency mappings for {num_bars} bars, "
                   f"FFT size {fft_size}, sample rate {sample_rate}")
        
    def _precompute_all(self) -> FrequencyMapping:
        """Pre-compute all frequency mappings"""
        # Create band indices using mel-scale mapping
        band_indices = self._create_mel_band_mapping()
        
        # Pre-compute frequency arrays
        num_bins = self.fft_size // 2 + 1
        freq_to_bin = np.arange(num_bins) * self.freq_bin_width
        bin_to_freq = freq_to_bin.copy()
        
        # Pre-compute mel scale factors for each bin
        mel_scale_factors = self._compute_mel_scale_factors(freq_to_bin)
        
        # Pre-compute frequency compensation curve
        compensation_curve = self._compute_compensation_curve(freq_to_bin)
        
        # Pre-compute frequency points for each bar
        frequency_points = np.zeros(self.num_bars)
        for i, (start_idx, end_idx) in enumerate(band_indices):
            if i < self.num_bars:
                center_idx = (start_idx + end_idx) // 2
                frequency_points[i] = center_idx * self.freq_bin_width
                
        return FrequencyMapping(
            band_indices=band_indices,
            freq_to_bin=freq_to_bin,
            bin_to_freq=bin_to_freq,
            mel_scale_factors=mel_scale_factors,
            compensation_curve=compensation_curve,
            frequency_points=frequency_points
        )
        
    def _create_mel_band_mapping(self) -> List[Tuple[int, int]]:
        """Create frequency band indices with mel-scale mapping"""
        bands = []
        
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create points in mel scale
        mel_min = hz_to_mel(20)
        mel_max = hz_to_mel(20000)
        mel_points = np.linspace(mel_min, mel_max, self.num_bars + 1)
        freq_points = [mel_to_hz(mel) for mel in mel_points]
        
        # Ensure bounds
        freq_points[0] = max(20, freq_points[0])
        freq_points[-1] = min(20000, freq_points[-1])
        
        # Map to FFT bins
        for i in range(self.num_bars):
            if i >= len(freq_points) - 1:
                break
                
            start_freq = freq_points[i]
            end_freq = freq_points[i + 1]
            
            start_idx = int(start_freq / self.freq_bin_width)
            end_idx = int(end_freq / self.freq_bin_width)
            
            # Ensure at least one bin per band
            if end_idx <= start_idx:
                end_idx = start_idx + 1
                
            # Clamp to valid range
            start_idx = max(0, min(start_idx, self.fft_size // 2))
            end_idx = max(start_idx + 1, min(end_idx, self.fft_size // 2 + 1))
            
            bands.append((start_idx, end_idx))
            
        return bands
        
    def _compute_mel_scale_factors(self, frequencies: np.ndarray) -> np.ndarray:
        """Pre-compute mel scale factors for each frequency bin"""
        # Mel scale weighting for perceptual balance
        mel_factors = np.ones_like(frequencies)
        
        # Apply different weights for frequency ranges
        for i, freq in enumerate(frequencies):
            if freq < 250:  # Bass
                mel_factors[i] = 1.5
            elif freq < 500:  # Low-mid
                mel_factors[i] = 1.3
            elif freq < 2000:  # Mid
                mel_factors[i] = 1.1
            elif freq < 6000:  # High-mid
                mel_factors[i] = 1.0
            else:  # High
                mel_factors[i] = 0.9
                
        return mel_factors
        
    def _compute_compensation_curve(self, frequencies: np.ndarray) -> np.ndarray:
        """Pre-compute frequency compensation curve"""
        # Equal loudness compensation (simplified ISO 226:2003)
        compensation = np.ones_like(frequencies)
        
        for i, freq in enumerate(frequencies):
            if freq > 0:
                # Simplified equal loudness curve
                if freq < 100:
                    compensation[i] = 1.0 + (100 - freq) / 100 * 0.5
                elif freq < 1000:
                    compensation[i] = 1.0
                elif freq < 4000:
                    compensation[i] = 1.0 + (freq - 1000) / 3000 * 0.3
                else:
                    compensation[i] = 1.3 - (freq - 4000) / 16000 * 0.5
                    
        return compensation
        
    def map_spectrum_to_bars(self, spectrum: np.ndarray, 
                            apply_compensation: bool = True) -> np.ndarray:
        """
        Map spectrum to frequency bars using pre-computed mappings
        
        Args:
            spectrum: Input spectrum data
            apply_compensation: Whether to apply frequency compensation
            
        Returns:
            Array of bar values
        """
        band_values = np.zeros(self.num_bars, dtype=np.float32)
        
        # Apply compensation if requested
        if apply_compensation and len(spectrum) == len(self.mapping.compensation_curve):
            spectrum = spectrum * self.mapping.compensation_curve
            
        # Map to bars using pre-computed indices
        for i, (start_idx, end_idx) in enumerate(self.mapping.band_indices):
            if i >= self.num_bars:
                break
                
            if end_idx > len(spectrum):
                break
                
            if end_idx > start_idx:
                band_values[i] = np.mean(spectrum[start_idx:end_idx])
            else:
                band_values[i] = spectrum[start_idx] if start_idx < len(spectrum) else 0
                
        return band_values
        
    def get_frequency_for_bar(self, bar_index: int) -> float:
        """Get center frequency for a given bar using pre-computed data"""
        if 0 <= bar_index < self.num_bars:
            return self.mapping.frequency_points[bar_index]
        return 0.0
        
    def get_bar_for_frequency(self, frequency: float) -> int:
        """Get bar index for a given frequency"""
        # Binary search in pre-computed frequency points
        idx = np.searchsorted(self.mapping.frequency_points, frequency)
        return max(0, min(idx, self.num_bars - 1))