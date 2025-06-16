"""
Audio Processing Pipeline for OMEGA-4 Audio Analyzer
Phase 5: Extract audio processing and band mapping logic
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import signal as scipy_signal


class ContentTypeDetector:
    """Detects content type for adaptive frequency allocation"""
    
    def __init__(self):
        self.history_size = 30  # 30 frames ~0.5 seconds
        self.voice_history = []
        self.energy_history = []
        self.spectral_history = []
        
    def analyze_content(self, voice_info: Dict, band_values: np.ndarray, 
                       freq_starts: np.ndarray, freq_ends: np.ndarray) -> str:
        """Analyze content type: 'music', 'speech', 'mixed', 'instrumental'"""
        
        # Voice detection confidence
        voice_confidence = voice_info.get('confidence', 0.0) if voice_info else 0.0
        self.voice_history.append(voice_confidence)
        
        # Energy distribution analysis
        bass_energy = np.mean(band_values[:int(len(band_values) * 0.3)])  # Low 30%
        mid_energy = np.mean(band_values[int(len(band_values) * 0.3):int(len(band_values) * 0.7)])  # Mid 40%
        high_energy = np.mean(band_values[int(len(band_values) * 0.7):])  # High 30%
        
        total_energy = bass_energy + mid_energy + high_energy
        if total_energy > 0:
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
        else:
            bass_ratio = mid_ratio = high_ratio = 0.33
            
        self.energy_history.append((bass_ratio, mid_ratio, high_ratio))
        
        # Spectral centroid (brightness measure)
        # Calculate center frequencies for each band
        band_centers = (freq_starts + freq_ends) / 2.0
        spectral_centroid = np.sum(band_centers * band_values) / (np.sum(band_values) + 1e-10)
        self.spectral_history.append(spectral_centroid)
        
        # Maintain history size
        if len(self.voice_history) > self.history_size:
            self.voice_history.pop(0)
            self.energy_history.pop(0)
            self.spectral_history.pop(0)
            
        # Analyze trends
        avg_voice_confidence = np.mean(self.voice_history) if self.voice_history else 0.0
        avg_bass_ratio = np.mean([e[0] for e in self.energy_history]) if self.energy_history else 0.33
        avg_centroid = np.mean(self.spectral_history) if self.spectral_history else 1000
        
        # Classification logic
        if avg_voice_confidence > 0.7:
            return 'speech'  # Clear vocals
        elif avg_voice_confidence > 0.4 and avg_centroid > 800:
            return 'mixed'   # Music with vocals
        elif avg_bass_ratio > 0.4 and avg_centroid < 800:
            return 'music'   # Bass-heavy music (relaxed thresholds)
        else:
            return 'instrumental'  # General instrumental
            
    def get_allocation_for_content(self, content_type: str) -> float:
        """Get low-end allocation percentage for content type"""
        allocations = {
            'music': 0.80,        # 80% for bass-heavy music
            'speech': 0.60,       # 60% for speech (more mid-range)
            'mixed': 0.70,        # 70% for mixed content
            'instrumental': 0.75  # 75% for general instrumental
        }
        return allocations.get(content_type, 0.75)


class AudioProcessingPipeline:
    """Central audio processing pipeline with gain control and band mapping"""
    
    def __init__(self, sample_rate: int = 48000, num_bands: int = 768):
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.nyquist = sample_rate / 2
        
        # Gain control
        self.input_gain = 4.0  # Default 12dB boost
        self.auto_gain_enabled = True
        self.gain_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.target_lufs = -16.0  # Target for good visualization
        
        # Audio buffer
        self.ring_buffer = np.zeros(4096 * 4, dtype=np.float32)
        self.buffer_pos = 0
        
        # Smoothing buffers
        self.smoothing_buffer = np.zeros(num_bands)
        self.smoothing_factor = 0.7
        
        # Create band mapping
        self.band_indices = self._create_enhanced_band_mapping()
        
    def _create_enhanced_band_mapping(self) -> Dict:
        """Create perceptual frequency band mapping"""
        band_indices = {'starts': [], 'ends': [], 'freq_starts': [], 'freq_ends': []}
        
        # Number of bins in the FFT
        fft_size = 4096  # Base FFT size
        freq_bin_width = self.nyquist / (fft_size // 2)
        
        # Perceptual scale parameters
        min_freq = 20
        max_freq = 20000
        
        # Create perceptual frequency scale
        # Use combination of logarithmic for low frequencies and linear for high
        transition_freq = 1000  # Transition at 1kHz
        transition_band = int(self.num_bands * 0.5)  # 50% of bands for low frequencies
        
        # Low frequency bands (20Hz - 1kHz) - logarithmic
        low_freqs = np.logspace(np.log10(min_freq), np.log10(transition_freq), transition_band)
        
        # High frequency bands (1kHz - 20kHz) - more linear
        high_freqs = np.linspace(transition_freq, max_freq, self.num_bands - transition_band + 1)[1:]
        
        # Combine frequency ranges
        all_freqs = np.concatenate([low_freqs, high_freqs])
        
        # Create bands
        for i in range(len(all_freqs) - 1):
            freq_start = all_freqs[i]
            freq_end = all_freqs[i + 1]
            
            # Convert frequencies to FFT bin indices
            bin_start = int(freq_start / freq_bin_width)
            bin_end = int(freq_end / freq_bin_width)
            
            # Ensure at least one bin per band
            if bin_end <= bin_start:
                bin_end = bin_start + 1
                
            # Clamp to valid range
            bin_start = max(0, min(bin_start, fft_size // 2 - 1))
            bin_end = max(bin_start + 1, min(bin_end, fft_size // 2))
            
            band_indices['starts'].append(bin_start)
            band_indices['ends'].append(bin_end)
            band_indices['freq_starts'].append(freq_start)
            band_indices['freq_ends'].append(freq_end)
            
        # Convert to numpy arrays
        band_indices['starts'] = np.array(band_indices['starts'])
        band_indices['ends'] = np.array(band_indices['ends'])
        band_indices['freq_starts'] = np.array(band_indices['freq_starts'])
        band_indices['freq_ends'] = np.array(band_indices['freq_ends'])
        
        return band_indices
    
    def apply_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply input gain with soft clipping"""
        if self.input_gain == 1.0:
            return audio_data
            
        # Apply gain
        gained = audio_data * self.input_gain
        
        # Soft clipping to prevent harsh distortion
        clip_threshold = 0.95
        over_threshold = np.abs(gained) > clip_threshold
        if np.any(over_threshold):
            # Soft knee compression
            gained[over_threshold] = np.sign(gained[over_threshold]) * (
                clip_threshold + (1 - clip_threshold) * np.tanh(
                    (np.abs(gained[over_threshold]) - clip_threshold) / (1 - clip_threshold)
                )
            )
            
        return gained
    
    def update_auto_gain(self, current_lufs: float):
        """Update automatic gain control based on LUFS measurement"""
        if not self.auto_gain_enabled or current_lufs <= -100:
            return
            
        # Calculate gain adjustment needed
        lufs_difference = self.target_lufs - current_lufs
        
        # Limit adjustment speed (0.1 dB per frame max)
        max_adjustment = 0.1
        adjustment = np.clip(lufs_difference * 0.02, -max_adjustment, max_adjustment)
        
        # Apply adjustment with limits
        new_gain = self.input_gain * (10 ** (adjustment / 20))
        self.input_gain = np.clip(new_gain, 0.5, 8.0)  # -6dB to +18dB range
        
        self.gain_history.append(self.input_gain)
    
    def process_frame(self, audio_data: np.ndarray) -> Dict:
        """Process single audio frame through the pipeline"""
        # Apply gain
        audio_data = self.apply_gain(audio_data)
        
        # Update ring buffer
        chunk_len = len(audio_data)
        if self.buffer_pos + chunk_len <= len(self.ring_buffer):
            self.ring_buffer[self.buffer_pos:self.buffer_pos + chunk_len] = audio_data
        else:
            first_part = len(self.ring_buffer) - self.buffer_pos
            self.ring_buffer[self.buffer_pos:] = audio_data[:first_part]
            self.ring_buffer[:chunk_len - first_part] = audio_data[first_part:]
            
        self.buffer_pos = (self.buffer_pos + chunk_len) % len(self.ring_buffer)
        
        return {
            'audio_data': audio_data,
            'buffer_position': self.buffer_pos,
            'current_gain': self.input_gain
        }
    
    def map_to_bands(self, fft_magnitude: np.ndarray, apply_smoothing: bool = True) -> np.ndarray:
        """Map FFT bins to display bands with optional smoothing"""
        band_values = np.zeros(self.num_bands)
        
        # Map FFT bins to bands
        num_available_bands = len(self.band_indices['starts'])
        for i in range(min(self.num_bands, num_available_bands)):
            start_bin = self.band_indices['starts'][i]
            end_bin = self.band_indices['ends'][i]
            
            if start_bin < len(fft_magnitude) and end_bin <= len(fft_magnitude):
                # Use max value in band for better peak visibility
                band_values[i] = np.max(fft_magnitude[start_bin:end_bin])
        
        # Apply smoothing if enabled
        if apply_smoothing:
            self.smoothing_buffer = (
                self.smoothing_factor * self.smoothing_buffer + 
                (1 - self.smoothing_factor) * band_values
            )
            return self.smoothing_buffer.copy()
        
        return band_values
    
    def get_band_frequencies(self) -> Dict[str, np.ndarray]:
        """Get frequency ranges for each band"""
        return {
            'starts': self.band_indices['freq_starts'],
            'ends': self.band_indices['freq_ends'],
            'centers': (self.band_indices['freq_starts'] + self.band_indices['freq_ends']) / 2
        }
    
    def apply_frequency_compensation(self, band_values: np.ndarray) -> np.ndarray:
        """Apply frequency response compensation curve"""
        compensated = band_values.copy()
        freq_centers = (self.band_indices['freq_starts'] + self.band_indices['freq_ends']) / 2
        
        for i, freq in enumerate(freq_centers):
            if freq < 100:
                compensated[i] *= 2.0  # Boost sub-bass
            elif freq < 250:
                compensated[i] *= 1.5  # Boost bass
            elif freq < 500:
                compensated[i] *= 1.2  # Slight boost lower mids
            elif freq < 2000:
                compensated[i] *= 1.0  # Neutral mids
            elif freq < 8000:
                compensated[i] *= 1.1  # Slight presence boost
            else:
                compensated[i] *= 1.3  # Boost highs for clarity
                
        return compensated
    
    def normalize_for_display(self, band_values: np.ndarray, 
                            target_max: float = 0.8) -> np.ndarray:
        """Normalize values for optimal display"""
        max_val = np.max(band_values)
        if max_val > 0:
            # Dynamic range compression for better visibility
            normalized = band_values / max_val
            # Soft knee compression
            compressed = np.tanh(normalized * 2) * target_max
            return compressed
        return band_values