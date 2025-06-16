"""
Audio Processing Pipeline for OMEGA-4 Audio Analyzer
Phase 5: Extract audio processing and band mapping logic
Enhanced with robust error handling and performance optimizations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import signal as scipy_signal

from .audio_config import PipelineConfig
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class ContentTypeDetector:
    """Detects content type for adaptive frequency allocation"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.history_size = config.content_history_size
        self.voice_history = deque(maxlen=self.history_size)
        self.energy_history = deque(maxlen=self.history_size)
        self.spectral_history = deque(maxlen=self.history_size)
        
    def analyze_content(self, voice_info: Dict, band_values: np.ndarray, 
                       freq_starts: np.ndarray, freq_ends: np.ndarray) -> str:
        """Analyze content type: 'music', 'speech', 'mixed', 'instrumental'"""
        try:
            # Input validation
            if len(band_values) == 0:
                logger.warning("Empty band values, defaulting to instrumental")
                return 'instrumental'
                
            if voice_info is None:
                voice_info = {'confidence': 0.0}
            
            # Voice detection confidence
            voice_confidence = float(voice_info.get('confidence', 0.0))
            self.voice_history.append(voice_confidence)
            
            # Energy distribution analysis with bounds checking
            num_bands = len(band_values)
            bass_end = min(int(num_bands * 0.3), num_bands)
            mid_start = bass_end
            mid_end = min(int(num_bands * 0.7), num_bands)
            high_start = mid_end
            
            bass_energy = np.mean(band_values[:bass_end]) if bass_end > 0 else 0
            mid_energy = np.mean(band_values[mid_start:mid_end]) if mid_end > mid_start else 0
            high_energy = np.mean(band_values[high_start:]) if high_start < num_bands else 0
            
            total_energy = bass_energy + mid_energy + high_energy
            if total_energy > self.config.epsilon:
                bass_ratio = bass_energy / total_energy
                mid_ratio = mid_energy / total_energy
                high_ratio = high_energy / total_energy
            else:
                bass_ratio = mid_ratio = high_ratio = 0.33
                
            self.energy_history.append((bass_ratio, mid_ratio, high_ratio))
            
            # Spectral centroid (brightness measure) with numerical stability
            if len(freq_starts) == len(band_values) and len(freq_ends) == len(band_values):
                band_centers = (freq_starts + freq_ends) / 2.0
                band_sum = np.sum(band_values)
                if band_sum > self.config.epsilon:
                    spectral_centroid = np.sum(band_centers * band_values) / band_sum
                else:
                    spectral_centroid = 1000.0  # Default centroid
            else:
                spectral_centroid = 1000.0
                
            self.spectral_history.append(spectral_centroid)
            
            # Analyze trends
            avg_voice_confidence = np.mean(self.voice_history) if len(self.voice_history) > 0 else 0.0
            avg_bass_ratio = np.mean([e[0] for e in self.energy_history]) if len(self.energy_history) > 0 else 0.33
            avg_centroid = np.mean(self.spectral_history) if len(self.spectral_history) > 0 else 1000
            
            # Classification logic
            if avg_voice_confidence > self.config.voice_confidence_threshold:
                return 'speech'
            elif avg_voice_confidence > self.config.mixed_confidence_threshold and avg_centroid > 800:
                return 'mixed'
            elif avg_bass_ratio > 0.4 and avg_centroid < 800:
                return 'music'
            else:
                return 'instrumental'
                
        except Exception as e:
            logger.error(f"Content analysis error: {e}")
            return 'instrumental'  # Safe default
            
    def get_allocation_for_content(self, content_type: str) -> float:
        """Get low-end allocation percentage for content type"""
        allocations = {
            'music': 0.80,
            'speech': 0.60,
            'mixed': 0.70,
            'instrumental': 0.75
        }
        return allocations.get(content_type, 0.75)


class AudioProcessingPipeline:
    """Central audio processing pipeline with gain control and band mapping"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.sample_rate = self.config.sample_rate
        self.num_bands = self.config.num_bands
        self.nyquist = self.sample_rate / 2
        
        # Gain control
        self.input_gain = self.config.input_gain
        self.auto_gain_enabled = self.config.auto_gain_enabled
        self.gain_history = deque(maxlen=self.config.gain_history_size)
        self.target_lufs = self.config.target_lufs
        
        # Audio buffer with validation
        self.ring_buffer = np.zeros(self.config.ring_buffer_size, dtype=np.float32)
        self.buffer_pos = 0
        
        # Smoothing buffers
        self.smoothing_buffer = np.zeros(self.num_bands, dtype=np.float32)
        self.smoothing_factor = self.config.smoothing_factor
        
        # Pre-allocate working arrays
        self._temp_band_values = np.zeros(self.num_bands, dtype=np.float32)
        
        # Create band mapping
        self.band_indices = self._create_enhanced_band_mapping(self.config.fft_size)
        
        # Pre-compute frequency compensation
        self._precompute_frequency_compensation()
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        logger.info(f"AudioProcessingPipeline initialized: {self.sample_rate}Hz, {self.num_bands} bands")
        
    def _precompute_frequency_compensation(self):
        """Pre-compute frequency response compensation multipliers"""
        if 'freq_starts' not in self.band_indices or 'freq_ends' not in self.band_indices:
            logger.warning("Band indices not properly initialized")
            self.freq_compensation = np.ones(self.num_bands)
            return
            
        freq_centers = (np.array(self.band_indices['freq_starts']) + 
                       np.array(self.band_indices['freq_ends'])) / 2
        self.freq_compensation = np.ones_like(freq_centers)
        
        # Vectorized compensation calculation
        self.freq_compensation[freq_centers < 100] = 2.0
        mask_bass = (freq_centers >= 100) & (freq_centers < 250)
        self.freq_compensation[mask_bass] = 1.5
        mask_low_mid = (freq_centers >= 250) & (freq_centers < 1000)
        self.freq_compensation[mask_low_mid] = 1.2
        mask_high = freq_centers >= 10000
        self.freq_compensation[mask_high] = 1.3
        
    def _create_enhanced_band_mapping(self, fft_size: int) -> Dict:
        """Create perceptual frequency band mapping with validation"""
        band_indices = {
            'starts': [], 
            'ends': [], 
            'freq_starts': [], 
            'freq_ends': []
        }
        
        try:
            # Validate FFT size
            if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
                logger.error(f"Invalid FFT size: {fft_size}, using default 4096")
                fft_size = 4096
                
            freq_bin_width = self.nyquist / (fft_size // 2)
            
            # Perceptual scale parameters
            min_freq = self.config.min_frequency
            max_freq = min(self.config.max_frequency, self.nyquist)
            transition_freq = self.config.transition_frequency
            transition_band = int(self.num_bands * self.config.low_freq_band_ratio)
            
            # Low frequency bands (logarithmic)
            low_freqs = np.logspace(np.log10(min_freq), np.log10(transition_freq), 
                                   transition_band)
            
            # High frequency bands (more linear)
            high_freqs = np.linspace(transition_freq, max_freq, 
                                    self.num_bands - transition_band + 1)[1:]
            
            # Combine frequency ranges
            all_freqs = np.concatenate([low_freqs, high_freqs])
            
            # Create bands with validation
            for i in range(min(len(all_freqs) - 1, self.num_bands)):
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
                
        except Exception as e:
            logger.error(f"Error creating band mapping: {e}")
            # Create simple linear fallback
            for i in range(self.num_bands):
                band_indices['starts'].append(i)
                band_indices['ends'].append(i + 1)
                band_indices['freq_starts'].append(i * 20)
                band_indices['freq_ends'].append((i + 1) * 20)
                
        return band_indices
        
    def process_frame(self, audio_data: np.ndarray) -> Dict:
        """Process single audio frame with validation and monitoring"""
        self.performance_monitor.start_frame()
        
        try:
            # Input validation
            if not isinstance(audio_data, np.ndarray):
                raise TypeError("audio_data must be numpy array")
                
            if len(audio_data) == 0:
                return self._empty_frame_result()
                
            if len(audio_data) > len(self.ring_buffer):
                logger.error(f"Audio chunk ({len(audio_data)}) larger than buffer ({len(self.ring_buffer)})")
                return self._empty_frame_result()
                
            # Convert to float32 if needed
            if audio_data.dtype not in [np.float32, np.float64]:
                audio_data = audio_data.astype(np.float32)
                
            # Update ring buffer with bounds checking
            chunk_len = len(audio_data)
            if self.buffer_pos + chunk_len <= len(self.ring_buffer):
                self.ring_buffer[self.buffer_pos:self.buffer_pos + chunk_len] = audio_data
            else:
                # Wrap around
                first_part = len(self.ring_buffer) - self.buffer_pos
                self.ring_buffer[self.buffer_pos:] = audio_data[:first_part]
                self.ring_buffer[:chunk_len - first_part] = audio_data[first_part:]
                
            self.buffer_pos = (self.buffer_pos + chunk_len) % len(self.ring_buffer)
            
            # Apply gain
            audio_data = audio_data * self.input_gain
            
            # Auto-adjust gain if enabled
            if self.auto_gain_enabled:
                self._auto_adjust_gain(audio_data)
                
            result = {
                'audio_data': audio_data,
                'buffer_pos': self.buffer_pos,
                'input_gain': self.input_gain,
                'buffer_fill': self.buffer_pos / len(self.ring_buffer)
            }
            
            # Update performance monitor
            self.performance_monitor.end_frame(result['buffer_fill'])
            
            # Check if we should reduce quality
            if self.performance_monitor.should_reduce_quality():
                logger.warning("Performance degraded, consider reducing quality")
                
            return result
            
        except TypeError:
            # Re-raise TypeError for invalid input types
            raise
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.performance_monitor.end_frame(0.0)
            return self._empty_frame_result()
            
    def map_to_bands(self, fft_magnitude: np.ndarray, apply_smoothing: bool = True) -> np.ndarray:
        """Optimized band mapping using numpy operations"""
        try:
            # Reset temp array
            self._temp_band_values.fill(0)
            
            # Validate input
            if fft_magnitude is None or len(fft_magnitude) == 0:
                return self._temp_band_values
                
            # Get valid band count
            num_valid_bands = min(
                self.num_bands, 
                len(self.band_indices['starts']),
                len(self.band_indices['ends'])
            )
            
            # Vectorized band mapping
            for i in range(num_valid_bands):
                start_bin = self.band_indices['starts'][i]
                end_bin = self.band_indices['ends'][i]
                
                if start_bin < len(fft_magnitude) and end_bin <= len(fft_magnitude):
                    # Use maximum value in band (can be changed to mean/rms)
                    self._temp_band_values[i] = np.max(fft_magnitude[start_bin:end_bin])
                    
            # Apply frequency compensation
            if hasattr(self, 'freq_compensation'):
                self._temp_band_values[:len(self.freq_compensation)] *= self.freq_compensation
                
            # Apply smoothing with in-place operations
            if apply_smoothing:
                self.smoothing_buffer *= self.smoothing_factor
                self.smoothing_buffer += (1 - self.smoothing_factor) * self._temp_band_values
                return self.smoothing_buffer.copy()
                
            return self._temp_band_values.copy()
            
        except Exception as e:
            logger.error(f"Error in band mapping: {e}")
            return np.zeros(self.num_bands)
            
    def _auto_adjust_gain(self, audio_data: np.ndarray):
        """Auto-adjust gain based on signal level"""
        try:
            # Calculate RMS with numerical stability
            rms = np.sqrt(np.mean(np.square(audio_data)))
            
            if rms > self.config.epsilon:
                # Convert to dB
                current_db = 20 * np.log10(rms)
                
                # Target is around -20dB for good headroom
                target_db = -20.0
                gain_adjustment = target_db - current_db
                
                # Limit adjustment speed
                gain_adjustment = np.clip(gain_adjustment, -0.5, 0.5)
                
                # Update gain with limits
                new_gain = self.input_gain * (10 ** (gain_adjustment / 20))
                self.input_gain = np.clip(new_gain, 0.1, 10.0)
                
                self.gain_history.append(self.input_gain)
                
        except Exception as e:
            logger.error(f"Error in auto gain adjustment: {e}")
            
    def _empty_frame_result(self) -> Dict:
        """Return empty frame result for error cases"""
        return {
            'audio_data': np.zeros(0),
            'buffer_pos': self.buffer_pos,
            'input_gain': self.input_gain,
            'buffer_fill': 0.0
        }
        
    def get_band_frequencies(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get frequency ranges for each band"""
        if 'freq_starts' in self.band_indices and 'freq_ends' in self.band_indices:
            return (np.array(self.band_indices['freq_starts']), 
                   np.array(self.band_indices['freq_ends']))
        else:
            # Return default ranges
            freqs = np.linspace(0, self.nyquist, self.num_bands + 1)
            return freqs[:-1], freqs[1:]
            
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.performance_monitor.get_statistics()
        
    def reset(self):
        """Reset pipeline state"""
        self.ring_buffer.fill(0)
        self.buffer_pos = 0
        self.smoothing_buffer.fill(0)
        self._temp_band_values.fill(0)
        self.gain_history.clear()
        self.performance_monitor.reset()
        logger.info("Pipeline reset")