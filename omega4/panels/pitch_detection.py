"""
Pitch Detection Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract pitch detection as self-contained module
OMEGA Feature: Advanced pitch detection with cepstral analysis
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from scipy import signal as scipy_signal
import logging
import time
from .pitch_detection_config import PitchDetectionConfig, ConfigurationManager

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Default to warning, can be changed via config


class CepstralAnalyzer:
    """Advanced pitch detection using cepstrum, autocorrelation, and YIN algorithm"""
    
    def __init__(self, sample_rate: int = 48000, config: Optional[PitchDetectionConfig] = None):
        self.sample_rate = sample_rate
        
        # Use provided config or create default
        self.config = config or PitchDetectionConfig(sample_rate=sample_rate)
        if self.config.enable_debug_logging:
            logger.setLevel(logging.DEBUG)
        
        # Initialize from config
        self.min_pitch = self.config.min_pitch
        self.max_pitch = self.config.max_pitch
        self.min_period = int(sample_rate / self.max_pitch)
        self.max_period = int(sample_rate / self.min_pitch)
        
        # YIN algorithm parameters from config
        self.yin_threshold = self.config.yin_threshold
        self.yin_window_size = self.config.yin_window_size
        
        # Confidence thresholds from config
        self.high_confidence = self.config.high_confidence_threshold
        self.medium_confidence = self.config.medium_confidence_threshold
        
        # History buffers
        self.pitch_history = deque(maxlen=self.config.max_history_length)
        self.confidence_history = deque(maxlen=self.config.max_history_length)
        
        # Pre-compute Hanning window for performance
        self._hanning_cache = {}
        
        # Cache filter coefficients
        self._filter_cache = {}
        self._init_filter_cache()
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100) if self.config.enable_performance_monitoring else None
        
        # Error statistics
        self.error_count = 0
        self.last_error_time = 0
        
    def validate_audio_input(self, signal: Optional[np.ndarray]) -> Tuple[bool, str]:
        """Validate audio input signal
        
        Returns:
            (is_valid, error_message)
        """
        if signal is None:
            return False, "Signal is None"
            
        if not isinstance(signal, np.ndarray):
            return False, f"Signal must be numpy array, got {type(signal)}"
            
        if len(signal) == 0:
            return False, "Signal is empty"
            
        if not np.isfinite(signal).all():
            return False, "Signal contains non-finite values (inf/nan)"
            
        if signal.dtype not in [np.float32, np.float64]:
            return False, f"Signal must be float32 or float64, got {signal.dtype}"
            
        if len(signal) < self.min_period * 2:
            return False, f"Signal too short for pitch detection (need at least {self.min_period * 2} samples)"
            
        # Check signal level
        signal_level = np.sqrt(np.mean(signal ** 2))
        if signal_level < 1e-6:
            return False, "Signal level too low (essentially silence)"
            
        return True, ""
    
    def get_hanning_window(self, size: int) -> np.ndarray:
        """Get cached Hanning window for given size"""
        if size not in self._hanning_cache:
            self._hanning_cache[size] = np.hanning(size)
        return self._hanning_cache[size]
    
    def _init_filter_cache(self):
        """Pre-compute filter coefficients if caching is enabled"""
        if self.config.cache_filter_coefficients and self.config.enable_preprocessing:
            nyquist = self.sample_rate / 2
            try:
                # Cache high-pass filter
                self._filter_cache['highpass'] = scipy_signal.butter(
                    self.config.highpass_order,
                    self.config.highpass_frequency / nyquist,
                    btype='high',
                    output='sos'
                )
                logger.debug("Filter coefficients cached successfully")
            except Exception as e:
                logger.warning(f"Failed to cache filter coefficients: {e}")
                self._filter_cache = {}
    
    def get_highpass_filter(self) -> np.ndarray:
        """Get cached or compute highpass filter coefficients"""
        if 'highpass' in self._filter_cache:
            return self._filter_cache['highpass']
        
        # Compute on demand if not cached
        nyquist = self.sample_rate / 2
        return scipy_signal.butter(
            self.config.highpass_order,
            self.config.highpass_frequency / nyquist,
            btype='high',
            output='sos'
        )
        
    def compute_cepstrum(self, signal: np.ndarray) -> np.ndarray:
        """Compute real cepstrum for pitch detection"""
        # Apply cached window to reduce spectral leakage
        window = self.get_hanning_window(len(signal))
        windowed = signal * window
        
        # Compute cepstrum via FFT
        spectrum = np.fft.rfft(windowed)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum)
        
        # Return only positive quefrency part
        return cepstrum[:len(cepstrum)//2]
    
    def detect_pitch_cepstral(self, cepstrum: np.ndarray) -> Tuple[float, float]:
        """Detect pitch using cepstral peak detection"""
        # Search for peak in valid pitch range
        min_sample = self.min_period
        max_sample = min(self.max_period, len(cepstrum) - 1)
        
        if max_sample <= min_sample:
            return 0.0, 0.0
            
        # Find peaks in cepstrum
        valid_range = cepstrum[min_sample:max_sample]
        peaks, properties = scipy_signal.find_peaks(
            valid_range, 
            height=np.max(valid_range) * 0.3,
            distance=20
        )
        
        if len(peaks) == 0:
            return 0.0, 0.0
            
        # Get strongest peak
        peak_idx = peaks[np.argmax(properties['peak_heights'])]
        period = peak_idx + min_sample
        pitch = self.sample_rate / period
        
        # Calculate confidence based on peak prominence
        confidence = min(1.0, properties['peak_heights'][np.argmax(properties['peak_heights'])] / np.std(cepstrum))
        
        return pitch, confidence
    
    def compute_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """Compute normalized autocorrelation function"""
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation using FFT for efficiency
        n = len(signal)
        # Pad to next power of 2 for FFT efficiency
        padded_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        # Compute autocorrelation via FFT
        fft = np.fft.rfft(signal, padded_size)
        power_spectrum = fft * np.conj(fft)
        autocorr = np.fft.irfft(power_spectrum)
        
        # Normalize and return relevant part
        autocorr = autocorr[:n]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        return autocorr
    
    def detect_pitch_autocorrelation(self, signal: np.ndarray) -> Tuple[float, float]:
        """Detect pitch using autocorrelation method"""
        autocorr = self.compute_autocorrelation(signal)
        
        # Find peaks in valid range
        min_lag = self.min_period
        max_lag = min(self.max_period, len(autocorr) - 1)
        
        if max_lag <= min_lag:
            return 0.0, 0.0
            
        # Find first significant peak after lag 0
        valid_range = autocorr[min_lag:max_lag]
        peaks, properties = scipy_signal.find_peaks(
            valid_range,
            height=0.3,  # At least 30% correlation
            distance=20
        )
        
        if len(peaks) == 0:
            return 0.0, 0.0
            
        # Get first strong peak (fundamental period)
        peak_idx = peaks[0]  # First peak is usually fundamental
        period = peak_idx + min_lag
        pitch = self.sample_rate / period
        
        # Confidence based on autocorrelation value
        confidence = autocorr[period]
        
        return pitch, confidence
    
    def yin_pitch_detection(self, signal: np.ndarray) -> Tuple[float, float]:
        """YIN algorithm for robust pitch detection - optimized vectorized version"""
        n = len(signal)
        if n < self.yin_window_size:
            return 0.0, 0.0
            
        # Use a window of the signal
        signal = signal[:self.yin_window_size]
        n = self.yin_window_size
        
        # Check if vectorization is enabled
        if self.config.yin_use_vectorization:
            return self._yin_pitch_detection_vectorized(signal)
        else:
            return self._yin_pitch_detection_standard(signal)
    
    def _yin_pitch_detection_vectorized(self, signal: np.ndarray) -> Tuple[float, float]:
        """Highly optimized vectorized YIN implementation"""
        n = len(signal)
        max_tau = min(self.max_period, n // 2)
        
        # Pre-allocate arrays
        diff_function = np.zeros(max_tau)
        
        # Step 1: Vectorized difference function calculation
        # Use broadcasting to compute all tau values more efficiently
        if max_tau > 0:
            # For small tau values, use fully vectorized approach
            small_tau_limit = min(100, max_tau)
            for tau in range(1, small_tau_limit):
                diff_function[tau] = np.sum((signal[:-tau] - signal[tau:])**2)
            
            # For larger tau values, use a more memory-efficient approach
            if max_tau > small_tau_limit:
                # Compute using sliding window correlation
                signal_padded = np.pad(signal, (0, max_tau), mode='constant')
                for tau in range(small_tau_limit, max_tau):
                    # More cache-friendly computation
                    diff_function[tau] = np.dot(signal, signal) + \
                                       np.dot(signal_padded[tau:tau+n], signal_padded[tau:tau+n]) - \
                                       2 * np.dot(signal, signal_padded[tau:tau+n])
        
        # Step 2: Cumulative mean normalized difference (vectorized)
        cumulative_mean = np.ones(max_tau)
        cumsum = np.cumsum(diff_function)
        
        # Vectorized normalization
        tau_range = np.arange(1, max_tau)
        cumulative_mean[1:] = diff_function[1:] * tau_range / cumsum[1:]
        
        # Step 3: Find first minimum below threshold (vectorized search)
        # Find all candidates below threshold
        valid_range = slice(self.min_period, max_tau - 1)
        candidates = np.where(cumulative_mean[valid_range] < self.yin_threshold)[0]
        
        if len(candidates) == 0:
            return 0.0, 0.0
        
        # Add offset to get actual indices
        candidates += self.min_period
        
        # Check for local minima
        local_minima = candidates[
            (cumulative_mean[candidates] < cumulative_mean[candidates - 1]) & 
            (cumulative_mean[candidates] < cumulative_mean[candidates + 1])
        ]
        
        if len(local_minima) == 0:
            return 0.0, 0.0
        
        # Use first local minimum
        tau = local_minima[0]
        
        # Step 4: Parabolic interpolation
        if 0 < tau < max_tau - 1:
            tau_float = self._parabolic_interpolation(
                cumulative_mean[tau - 1],
                cumulative_mean[tau],
                cumulative_mean[tau + 1],
                tau
            )
        else:
            tau_float = float(tau)
        
        pitch = self.sample_rate / tau_float
        confidence = 1.0 - cumulative_mean[tau]
        
        return pitch, confidence
    
    def _yin_pitch_detection_standard(self, signal: np.ndarray) -> Tuple[float, float]:
        """Standard YIN implementation (fallback)"""
        n = len(signal)
        
        # Step 1: Calculate difference function
        diff_function = np.zeros(n // 2)
        for tau in range(1, min(self.max_period, n // 2)):
            diff_function[tau] = np.sum((signal[:-tau] - signal[tau:])**2)
        
        # Step 2: Calculate cumulative mean normalized difference
        cumulative_mean = np.zeros(n // 2)
        cumulative_mean[0] = 1.0
        
        running_sum = 0.0
        for tau in range(1, n // 2):
            running_sum += diff_function[tau]
            cumulative_mean[tau] = diff_function[tau] / (running_sum / tau) if running_sum > 0 else 1.0
        
        # Step 3: Find first minimum below threshold
        tau = self.min_period
        while tau < min(self.max_period, n // 2 - 1):
            if cumulative_mean[tau] < self.yin_threshold:
                if tau + 1 < n // 2 and cumulative_mean[tau] < cumulative_mean[tau - 1] and cumulative_mean[tau] < cumulative_mean[tau + 1]:
                    break
            tau += 1
        
        # Step 4: Refine using parabolic interpolation
        if tau < min(self.max_period, n // 2 - 1):
            if tau > 0 and tau < n // 2 - 1:
                tau_float = self._parabolic_interpolation(
                    cumulative_mean[tau - 1],
                    cumulative_mean[tau],
                    cumulative_mean[tau + 1],
                    tau
                )
            else:
                tau_float = float(tau)
            
            pitch = self.sample_rate / tau_float
            confidence = 1.0 - cumulative_mean[int(tau)]
            
            return pitch, confidence
        
        return 0.0, 0.0
    
    def _parabolic_interpolation(self, y1: float, y2: float, y3: float, x: int) -> float:
        """Parabolic interpolation for sub-sample accuracy"""
        a = (y1 - 2 * y2 + y3) / 2
        b = (y3 - y1) / 2
        
        if a != 0:
            x_offset = -b / (2 * a)
            return x + x_offset
        return float(x)
    
    def combine_pitch_estimates(self, signal: np.ndarray, cepstrum: np.ndarray) -> Dict[str, Any]:
        """Combine multiple pitch detection methods with confidence weighting"""
        # Method 1: Cepstral analysis
        cepstral_pitch, cepstral_conf = self.detect_pitch_cepstral(cepstrum)
        
        # Method 2: Autocorrelation
        autocorr_pitch, autocorr_conf = self.detect_pitch_autocorrelation(signal)
        
        # Method 3: YIN algorithm
        yin_pitch, yin_conf = self.yin_pitch_detection(signal)
        
        # Collect valid estimates with lower thresholds for complex music
        estimates = []
        if cepstral_conf > 0.15 and self.min_pitch <= cepstral_pitch <= self.max_pitch:
            estimates.append((cepstral_pitch, cepstral_conf * 0.8))  # Slightly lower weight for cepstral
            
        if autocorr_conf > 0.2 and self.min_pitch <= autocorr_pitch <= self.max_pitch:
            estimates.append((autocorr_pitch, autocorr_conf * 0.9))
            
        if yin_conf > 0.25 and self.min_pitch <= yin_pitch <= self.max_pitch:
            estimates.append((yin_pitch, yin_conf * 1.1))  # YIN typically most reliable, boost weight
        
        # Store individual method results
        methods = {
            'cepstral': (cepstral_pitch, cepstral_conf),
            'autocorr': (autocorr_pitch, autocorr_conf),
            'yin': (yin_pitch, yin_conf)
        }
        
        if not estimates:
            return {
                'pitch': 0.0,
                'confidence': 0.0,
                'methods': methods
            }
        
        # Weighted average of estimates
        total_weight = sum(conf for _, conf in estimates)
        weighted_pitch = sum(pitch * conf for pitch, conf in estimates) / total_weight
        
        # Overall confidence is weighted average of individual confidences
        overall_confidence = total_weight / len(estimates) if estimates else 0.0
        
        # Check consistency between methods for confidence adjustment
        if len(estimates) >= 2:
            pitches = [pitch for pitch, _ in estimates]
            pitch_std = np.std(pitches)
            mean_pitch = np.mean(pitches)
            
            # If methods agree well (within 2%), boost confidence
            if pitch_std / mean_pitch < 0.02:
                overall_confidence = min(1.0, overall_confidence * 1.2)
            # If methods disagree significantly, reduce confidence
            elif pitch_std / mean_pitch > 0.05:
                overall_confidence *= 0.8
        
        return {
            'pitch': weighted_pitch,
            'confidence': overall_confidence,
            'methods': methods
        }
    
    def pitch_to_note(self, pitch: float) -> Tuple[str, int, int]:
        """Convert frequency to musical note with cents offset"""
        if pitch <= 0:
            return "", 0, 0
            
        # Reference: A4 = 440 Hz
        A4 = 440.0
        C0 = A4 * (2 ** (-4.75))  # C0 frequency
        
        # Calculate semitones from C0
        semitones = 12 * np.log2(pitch / C0)
        rounded_semitones = int(round(semitones))
        
        # Calculate cents offset
        cents_offset = int((semitones - rounded_semitones) * 100)
        
        # Get note name and octave
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = rounded_semitones % 12
        octave = rounded_semitones // 12
        
        return note_names[note_index], octave, cents_offset
    
    def calculate_pitch_stability(self) -> float:
        """Calculate stability of recent pitch estimates"""
        if len(self.pitch_history) < 10:
            return 0.0
            
        recent_pitches = list(self.pitch_history)[-20:]
        recent_confidences = list(self.confidence_history)[-20:]
        
        # Filter out low confidence estimates
        valid_pitches = [p for p, c in zip(recent_pitches, recent_confidences) if c > 0.3 and p > 0]
        
        if len(valid_pitches) < 5:
            return 0.0
            
        # Calculate coefficient of variation
        mean_pitch = np.mean(valid_pitches)
        std_pitch = np.std(valid_pitches)
        
        if mean_pitch > 0:
            cv = std_pitch / mean_pitch
            # Convert to stability score (lower CV = higher stability)
            stability = max(0.0, 1.0 - cv * 10)  # Scale factor of 10
        else:
            stability = 0.0
            
        return stability
    
    def detect_pitch_advanced(self, signal: np.ndarray) -> Dict[str, Any]:
        """Main entry point for advanced pitch detection"""
        if len(signal) < self.yin_window_size:
            return {
                'pitch': 0.0,
                'confidence': 0.0,
                'note': '',
                'cents_offset': 0,
                'octave': 0,
                'stability': 0.0
            }
            
        # Pre-process signal if enabled
        if self.config.enable_preprocessing:
            try:
                # Apply cached high-pass filter
                sos = self.get_highpass_filter()
                filtered_signal = scipy_signal.sosfilt(sos, signal)
                
                # Apply compression if enabled
                if self.config.enable_compression:
                    rms = np.sqrt(np.mean(filtered_signal ** 2))
                    if rms > 0.01:  # Only compress if signal is significant
                        # Soft knee compression
                        threshold = self.config.compression_threshold
                        if rms > threshold:
                            ratio = self.config.compression_ratio
                            gain = threshold + (rms - threshold) * ratio
                            filtered_signal = filtered_signal * (gain / rms)
                else:
                    # If signal is too quiet, use original
                    if np.sqrt(np.mean(filtered_signal ** 2)) < 0.01:
                        filtered_signal = signal
                        
            except Exception as e:
                # If filtering fails, use original signal
                logger.debug(f"Preprocessing failed: {e}")
                filtered_signal = signal
        else:
            # Skip preprocessing
            filtered_signal = signal
            
        # Compute cepstrum once for efficiency (use filtered signal)
        cepstrum = self.compute_cepstrum(filtered_signal)
        
        # Combine all methods (pass both original and filtered signals)
        # Some methods work better with filtered, others with original
        result = self.combine_pitch_estimates(signal, cepstrum)
        
        # Extract pitch and confidence
        pitch = result['pitch']
        confidence = result['confidence']
        
        # Update history
        self.pitch_history.append(pitch)
        self.confidence_history.append(confidence)
        
        # Convert to musical note
        note, octave, cents = self.pitch_to_note(pitch)
        
        # Calculate stability
        stability = self.calculate_pitch_stability()
        
        return {
            'pitch': pitch,
            'confidence': confidence,
            'note': note,
            'octave': octave,
            'cents_offset': cents,
            'stability': stability,
            'methods': result.get('methods', {})
        }
    
    def detect_pitch_advanced_safe(self, signal: np.ndarray) -> Dict[str, Any]:
        """Safe wrapper for pitch detection with comprehensive error handling
        
        This method validates input and handles all exceptions gracefully,
        returning default values on error.
        """
        start_time = time.time() if self.config.enable_performance_monitoring else None
        
        # Default result for errors
        default_result = {
            'pitch': 0.0,
            'confidence': 0.0,
            'note': '',
            'octave': 0,
            'cents_offset': 0,
            'stability': 0.0,
            'methods': {},
            'error': None
        }
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_audio_input(signal)
            if not is_valid:
                logger.debug(f"Invalid audio input: {error_msg}")
                default_result['error'] = error_msg
                return default_result
            
            # Ensure correct data type
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32)
            
            # Call main detection method
            result = self.detect_pitch_advanced(signal)
            
            # Record performance if enabled
            if self.config.enable_performance_monitoring and start_time:
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processing_times.append(processing_time)
                if self.config.log_processing_times:
                    logger.debug(f"Pitch detection took {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            # Log error
            current_time = time.time()
            if current_time - self.last_error_time > 1.0:  # Rate limit error logging
                logger.error(f"Pitch detection error: {str(e)}", exc_info=True)
                self.last_error_time = current_time
            
            self.error_count += 1
            default_result['error'] = str(e)
            return default_result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics if monitoring is enabled"""
        if not self.config.enable_performance_monitoring or not self.processing_times:
            return {}
        
        times = list(self.processing_times)
        return {
            'avg_time_ms': np.mean(times),
            'max_time_ms': np.max(times),
            'min_time_ms': np.min(times),
            'std_time_ms': np.std(times),
            'error_count': self.error_count
        }


class PitchDetectionPanel:
    """OMEGA Pitch Detection Panel with advanced visualization"""
    
    def __init__(self, sample_rate: int = 48000, config: Optional[PitchDetectionConfig] = None):
        self.sample_rate = sample_rate
        
        # Create configuration manager if not provided
        if config is None:
            self.config_manager = ConfigurationManager()
            config = self.config_manager.get_config()
        else:
            self.config_manager = None
        
        self.analyzer = CepstralAnalyzer(sample_rate, config)
        
        # Display state
        self.pitch_info = {
            'pitch': 0.0,
            'confidence': 0.0,
            'note': '',
            'octave': 0,
            'cents_offset': 0,
            'stability': 0.0,
            'methods': {},
            'error': None
        }
        
        # Pitch history for graph
        self.pitch_graph_history = deque(maxlen=60)  # ~1 second at 60 FPS
        
        # Fonts will be set by main app
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None
        
        # Error display state
        self.last_error_display_time = 0
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        self.font_tiny = fonts.get('tiny')
        
    def update(self, audio_data: np.ndarray):
        """Update pitch detection with new audio data"""
        try:
            # Use safe detection method
            self.pitch_info = self.analyzer.detect_pitch_advanced_safe(audio_data)
            
            # Update graph history
            if self.pitch_info.get('confidence', 0) > 0.3:
                self.pitch_graph_history.append(self.pitch_info.get('pitch', 0))
            else:
                self.pitch_graph_history.append(0)
                
        except Exception as e:
            # Fallback to prevent panel crash
            logger.error(f"PitchDetectionPanel update error: {e}")
            self.pitch_info = {
                'pitch': 0.0,
                'confidence': 0.0,
                'note': '',
                'octave': 0,
                'cents_offset': 0,
                'stability': 0.0,
                'methods': {},
                'error': str(e)
            }
            self.pitch_graph_history.append(0)
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """OMEGA: Draw advanced pitch detection information overlay"""
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Draw background with purple tint
        draw_panel_background(screen, x, y, width, height,
                            bg_color=(25, 20, 35), border_color=(110, 90, 140), alpha=230)
        
        # Create scaled fonts if ui_scale is different from 1.0
        if ui_scale != 1.0 and ui_scale > 0.5:
            # Create temporary scaled fonts
            font_medium = pygame.font.Font(None, int(28 * ui_scale)) if self.font_medium else None
            font_small = pygame.font.Font(None, int(24 * ui_scale)) if self.font_small else None
            font_tiny = pygame.font.Font(None, int(20 * ui_scale)) if self.font_tiny else None
        else:
            font_medium = self.font_medium
            font_small = self.font_small
            font_tiny = self.font_tiny
        
        # Draw centered header
        if font_medium:
            y_offset = draw_panel_header(screen, "OMEGA Pitch Detection", font_medium,
                                       x, y, width, bg_color=(25, 20, 35),
                                       border_color=(110, 90, 140),
                                       text_color=(220, 200, 255))
        else:
            # Fallback if no font is set
            fallback_font = pygame.font.Font(None, int(24 * ui_scale))
            y_offset = draw_panel_header(screen, "OMEGA Pitch Detection", fallback_font,
                                       x, y, width, bg_color=(25, 20, 35),
                                       border_color=(110, 90, 140),
                                       text_color=(220, 200, 255))
        
        y_offset += int(10 * ui_scale)  # Small gap after header
        line_height = int(26 * ui_scale)
        
        # Main pitch info
        pitch = self.pitch_info.get('pitch', 0.0)
        confidence = self.pitch_info.get('confidence', 0.0)
        note = self.pitch_info.get('note', '')
        octave = self.pitch_info.get('octave', 0)
        cents = self.pitch_info.get('cents_offset', 0)
        stability = self.pitch_info.get('stability', 0.0)
        
        # Pitch detection status
        if pitch > 0 and confidence > 0.3 and font_small and font_medium:
            # Show detected pitch
            pitch_color = (
                (255, 150, 100) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            
            # Frequency
            freq_text = f"Frequency: {pitch:.1f} Hz"
            freq_surf = font_small.render(freq_text, True, pitch_color)
            screen.blit(freq_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += line_height
            
            # Musical note with cents
            if note:
                note_text = f"Note: {note}{octave}"
                if cents != 0:
                    note_text += f" {cents:+d}¢"
                note_surf = font_medium.render(note_text, True, (200, 220, 255))
                screen.blit(note_surf, (x + int(20 * ui_scale), y_offset))
                y_offset += line_height + int(5 * ui_scale)
            
            # Confidence bar
            conf_text = f"Confidence: {confidence:.0%}"
            conf_surf = font_small.render(conf_text, True, (180, 180, 200))
            screen.blit(conf_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += line_height
            
            # Draw confidence bar
            bar_width = width - int(40 * ui_scale)
            bar_height = int(8 * ui_scale)
            bar_x = x + int(20 * ui_scale)
            bar_y = y_offset
            
            # Background
            pygame.draw.rect(screen, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height))
            # Confidence fill
            fill_width = int(bar_width * confidence)
            conf_color = (
                (200, 100, 100) if confidence < 0.5 else
                (200, 200, 100) if confidence < 0.7 else
                (100, 200, 100)
            )
            pygame.draw.rect(screen, conf_color, (bar_x, bar_y, fill_width, bar_height))
            # Border
            pygame.draw.rect(screen, (100, 100, 120), (bar_x, bar_y, bar_width, bar_height), 1)
            y_offset += bar_height + int(10 * ui_scale)
            
            # Stability indicator
            stab_text = f"Stability: {stability:.0%}"
            stab_color = (
                (255, 150, 150) if stability < 0.3 else
                (255, 255, 150) if stability < 0.7 else
                (150, 255, 150)
            )
            stab_surf = font_small.render(stab_text, True, stab_color)
            screen.blit(stab_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += line_height + int(10 * ui_scale)
            
        else:
            # No pitch detected
            if font_small:
                no_pitch_text = "No pitch detected"
                no_pitch_surf = font_small.render(no_pitch_text, True, (150, 150, 150))
                screen.blit(no_pitch_surf, (x + int(20 * ui_scale), y_offset))
            else:
                # Fallback if no font
                fallback_font = pygame.font.Font(None, int(20 * ui_scale))
                no_pitch_text = "No pitch detected"
                no_pitch_surf = fallback_font.render(no_pitch_text, True, (150, 150, 150))
                screen.blit(no_pitch_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += line_height + int(10 * ui_scale)
        
        # Method breakdown
        methods = self.pitch_info.get('methods', {})
        if methods and font_small and font_tiny:
            # Separator line
            pygame.draw.line(
                screen, (80, 80, 100), 
                (x + int(20 * ui_scale), y_offset), 
                (x + width - int(20 * ui_scale), y_offset), 1
            )
            y_offset += int(10 * ui_scale)
            
            # Methods header
            methods_text = font_small.render("Detection Methods:", True, (180, 180, 200))
            screen.blit(methods_text, (x + int(20 * ui_scale), y_offset))
            y_offset += line_height
            
            # Individual method results
            for method_name, (method_pitch, method_conf) in methods.items():
                if method_conf > 0:
                    method_color = (
                        (150, 150, 150) if method_conf < 0.3 else
                        (200, 200, 150) if method_conf < 0.6 else
                        (150, 200, 150)
                    )
                    
                    method_text = f"  {method_name.capitalize()}: "
                    if method_pitch > 0:
                        method_text += f"{method_pitch:.1f}Hz ({method_conf:.0%})"
                    else:
                        method_text += "No detection"
                    
                    method_surf = font_tiny.render(method_text, True, method_color)
                    screen.blit(method_surf, (x + int(20 * ui_scale), y_offset))
                    y_offset += int(18 * ui_scale)
        
        # Pitch history visualization (mini graph) - anchored to bottom
        graph_height = int(80 * ui_scale)
        graph_y = y + height - graph_height - int(20 * ui_scale)
        graph_x = x + int(20 * ui_scale)
        graph_width = width - int(40 * ui_scale)
        
        # Graph background
        pygame.draw.rect(screen, (30, 25, 40), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(screen, (60, 50, 80), (graph_x, graph_y, graph_width, graph_height), 1)
        
        # Draw pitch history
        if len(self.pitch_graph_history) > 1:
            # Find min/max for scaling
            valid_pitches = [p for p in self.pitch_graph_history if p > 0]
            if valid_pitches:
                min_pitch = min(valid_pitches) * 0.9
                max_pitch = max(valid_pitches) * 1.1
                
                # Draw graph line
                points = []
                for i, pitch_val in enumerate(self.pitch_graph_history):
                    if pitch_val > 0:
                        x_pos = graph_x + int(i * graph_width / len(self.pitch_graph_history))
                        y_normalized = (pitch_val - min_pitch) / (max_pitch - min_pitch)
                        y_pos = graph_y + graph_height - int(y_normalized * graph_height)
                        points.append((x_pos, y_pos))
                
                if len(points) > 1:
                    pygame.draw.lines(screen, (150, 200, 255), False, points, 2)
        
        # Graph label
        if font_tiny:
            graph_label = font_tiny.render("Pitch History", True, (150, 150, 170))
            screen.blit(graph_label, (graph_x, graph_y - int(15 * ui_scale)))
    
    def get_results(self) -> Dict[str, Any]:
        """Get current pitch detection results"""
        return self.pitch_info.copy()