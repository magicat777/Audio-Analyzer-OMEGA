"""
Room Mode Analyzer for OMEGA-4 Audio Analyzer
Phase 4: Extract room acoustics analysis for studio applications
Enhanced with mathematical fixes, caching, and improved accuracy
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from scipy import signal as scipy_signal
import logging
from dataclasses import dataclass

from .room_mode_config import RoomModeConfig

logger = logging.getLogger(__name__)


class RoomModeAnalyzer:
    """Enhanced room acoustics analysis for studio applications"""

    def __init__(self, sample_rate: int = 48000, config: Optional[RoomModeConfig] = None):
        self.sample_rate = sample_rate
        self.config = config or RoomModeConfig()
        self.sustained_peaks_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        
        # Performance optimization caches
        self._peak_cache = {}
        self._last_fft_hash = None
        self._cache_age = 0
        
        # Audio history for RT60 calculation
        self._audio_history = deque(maxlen=self.config.min_audio_history * 2)
        
        logger.info(f"RoomModeAnalyzer initialized: {sample_rate}Hz, "
                   f"range {self.config.min_frequency}-{self.config.max_frequency}Hz")

    def detect_room_modes(self, fft_data: np.ndarray, freqs: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential room modes from sustained frequency peaks with validation"""
        # Input validation
        if fft_data is None or freqs is None:
            return []
        
        if len(fft_data) != len(freqs):
            logger.error("FFT data and frequency arrays must have same length")
            return []
        
        if len(fft_data) == 0:
            return []
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(fft_data)) or not np.all(np.isfinite(freqs)):
            logger.warning("Invalid values in input arrays")
            return []
        
        # Check cache if enabled
        if self.config.enable_caching:
            try:
                current_hash = hash(fft_data.tobytes())
                if (current_hash == self._last_fft_hash and 
                    self._cache_age < self.config.cache_max_age_frames):
                    self._cache_age += 1
                    return self._peak_cache.get('room_modes', [])
                
                # Reset cache
                self._last_fft_hash = current_hash
                self._cache_age = 0
            except Exception as e:
                logger.debug(f"Caching error: {e}")
        
        room_modes = []

        # Find room mode frequency range with configurable bounds
        room_mode_mask = ((freqs >= self.config.min_frequency) & 
                          (freqs <= self.config.max_frequency))
        
        if not np.any(room_mode_mask):
            if self.config.enable_caching:
                self._peak_cache['room_modes'] = []
            return []

        room_mode_freqs = freqs[room_mode_mask]
        room_mode_magnitudes = fft_data[room_mode_mask]
        
        # Calculate adaptive thresholds
        mean_magnitude = np.mean(room_mode_magnitudes)
        std_magnitude = np.std(room_mode_magnitudes)
        
        if mean_magnitude == 0 or std_magnitude == 0:
            if self.config.enable_caching:
                self._peak_cache['room_modes'] = []
            return []

        # Find sustained peaks with configurable thresholds
        try:
            peaks, properties = scipy_signal.find_peaks(
                room_mode_magnitudes,
                height=mean_magnitude * self.config.peak_threshold_multiplier,
                prominence=std_magnitude * self.config.minimum_prominence_std,
            )
        except Exception as e:
            logger.error(f"Peak detection error: {e}")
            return []

        # Process each peak
        for i, peak_idx in enumerate(peaks):
            if peak_idx >= len(room_mode_freqs):
                continue
                
            freq = room_mode_freqs[peak_idx]
            magnitude = room_mode_magnitudes[peak_idx]

            # Estimate Q factor with fixed calculation using frequencies
            q_factor = self.estimate_q_factor(room_mode_magnitudes, room_mode_freqs, peak_idx)

            # Enhanced room mode classification
            mode_type = self.classify_room_mode(freq, self._get_room_dimensions())

            # Calculate severity based on magnitude and Q factor
            relative_magnitude = magnitude / mean_magnitude
            normalized_q = min(q_factor / 10.0, 1.0)
            severity = min(1.0, relative_magnitude * normalized_q * 0.5)

            if severity > self.config.minimum_severity:
                room_modes.append({
                    "frequency": float(freq),
                    "magnitude": float(magnitude),
                    "q_factor": float(q_factor),
                    "severity": float(severity),
                    "type": mode_type,
                    "prominence": float(properties["prominences"][i]) if i < len(properties["prominences"]) else 0.0,
                    "bandwidth": float(freq / q_factor) if q_factor > 0 else 0.0
                })

        # Sort by severity and cache if enabled
        room_modes = sorted(room_modes, key=lambda x: x["severity"], reverse=True)
        
        if self.config.enable_caching:
            self._peak_cache['room_modes'] = room_modes
            
        return room_modes

    def estimate_q_factor(self, magnitudes: np.ndarray, freqs: np.ndarray, peak_idx: int) -> float:
        """Estimate Q factor using actual frequency values (FIXED)"""
        if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
            return 1.0

        peak_magnitude = magnitudes[peak_idx]
        half_power = peak_magnitude / np.sqrt(2)
        center_freq = freqs[peak_idx]

        # Initialize with center frequency
        left_freq = center_freq
        right_freq = center_freq

        # Search left for -3dB point
        for i in range(peak_idx - 1, -1, -1):
            if magnitudes[i] <= half_power:
                # Interpolate for more accurate frequency
                if i < len(freqs) - 1:
                    # Linear interpolation between points
                    y1, y2 = magnitudes[i], magnitudes[i + 1]
                    f1, f2 = freqs[i], freqs[i + 1]
                    left_freq = f1 + (half_power - y1) * (f2 - f1) / (y2 - y1)
                else:
                    left_freq = freqs[i]
                break

        # Search right for -3dB point
        for i in range(peak_idx + 1, len(magnitudes)):
            if magnitudes[i] <= half_power:
                # Interpolate for more accurate frequency
                if i > 0:
                    y1, y2 = magnitudes[i - 1], magnitudes[i]
                    f1, f2 = freqs[i - 1], freqs[i]
                    right_freq = f1 + (half_power - y1) * (f2 - f1) / (y2 - y1)
                else:
                    right_freq = freqs[i]
                break

        # Calculate Q factor: center_frequency / bandwidth
        bandwidth = right_freq - left_freq
        if bandwidth > 0:
            q_factor = center_freq / bandwidth
        else:
            q_factor = 50.0  # Very sharp peak

        # Clamp to configured limits
        return min(self.config.max_q_factor, max(self.config.min_q_factor, q_factor))

    def classify_room_mode(self, frequency: float, room_dimensions: Optional[Dict] = None) -> str:
        """Enhanced room mode classification with optional room dimension hints"""
        
        # If we have room dimensions, calculate expected mode frequencies
        if room_dimensions and all(d > 0 for d in room_dimensions.values()):
            speed_sound = self.get_speed_of_sound()
            
            # Calculate fundamental frequencies for each dimension
            length_fundamental = speed_sound / (2 * room_dimensions.get('length', 10))
            width_fundamental = speed_sound / (2 * room_dimensions.get('width', 8))
            height_fundamental = speed_sound / (2 * room_dimensions.get('height', 3))
            
            # Check harmonics up to 3rd order
            for harmonic in range(1, 4):
                # Length modes
                if abs(frequency - length_fundamental * harmonic) < 5:
                    return f"axial_length_H{harmonic}"
                # Width modes
                if abs(frequency - width_fundamental * harmonic) < 5:
                    return f"axial_width_H{harmonic}"
                # Height modes
                if abs(frequency - height_fundamental * harmonic) < 5:
                    return f"axial_height_H{harmonic}"
            
            # Check for tangential modes (combination of two dimensions)
            tangential_freqs = [
                speed_sound * np.sqrt((1/(2*room_dimensions['length']))**2 + 
                                     (1/(2*room_dimensions['width']))**2),
                speed_sound * np.sqrt((1/(2*room_dimensions['length']))**2 + 
                                     (1/(2*room_dimensions['height']))**2),
                speed_sound * np.sqrt((1/(2*room_dimensions['width']))**2 + 
                                     (1/(2*room_dimensions['height']))**2)
            ]
            
            for tang_freq in tangential_freqs:
                if abs(frequency - tang_freq) < 10:
                    return "tangential"
        
        # Fallback to frequency-based classification
        if 30 <= frequency <= 80:
            return "axial_length"
        elif 80 <= frequency <= 120:
            return "axial_width"
        elif 120 <= frequency <= 200:
            return "axial_height"
        elif 200 <= frequency <= 300:
            return "tangential"
        else:
            return "oblique"

    def get_speed_of_sound(self, temperature_c: Optional[float] = None, 
                          humidity_percent: Optional[float] = None) -> float:
        """Calculate speed of sound based on environmental conditions"""
        # Use config values if not provided
        temp = temperature_c if temperature_c is not None else self.config.temperature_celsius
        humidity = humidity_percent if humidity_percent is not None else self.config.humidity_percent
        
        # Temperature correction (more accurate formula)
        speed_base = 331.3 * np.sqrt(1 + temp / 273.15)
        
        # Humidity correction (simplified but more accurate)
        # At 20°C, speed increases about 0.2 m/s per 10% humidity
        humidity_correction = 0.02 * humidity
        
        return speed_base + humidity_correction
    
    def _get_room_dimensions(self) -> Optional[Dict[str, float]]:
        """Get room dimensions from config if available"""
        if (self.config.room_length_hint > 0 and 
            self.config.room_width_hint > 0 and 
            self.config.room_height_hint > 0):
            return {
                'length': self.config.room_length_hint,
                'width': self.config.room_width_hint,
                'height': self.config.room_height_hint
            }
        return None
    
    def estimate_room_dimensions(self, room_modes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate room dimensions from detected modes"""
        speed_of_sound = self.get_speed_of_sound()
        
        dimensions = {
            "length": 0.0,
            "width": 0.0,
            "height": 0.0,
            "confidence": 0.0
        }
        
        # Group modes by type
        length_modes = [m for m in room_modes if "axial_length" in m["type"]]
        width_modes = [m for m in room_modes if "axial_width" in m["type"]]
        height_modes = [m for m in room_modes if "axial_height" in m["type"]]
        
        # Estimate dimensions from fundamental modes
        if length_modes:
            # Find the fundamental (lowest frequency mode)
            fundamental = min(length_modes, key=lambda x: x["frequency"])
            # Extract harmonic number if present
            harmonic = 1
            if "_H" in fundamental["type"]:
                try:
                    harmonic = int(fundamental["type"].split("_H")[1])
                except:
                    harmonic = 1
            dimensions["length"] = speed_of_sound / (2 * fundamental["frequency"] / harmonic)
            
        if width_modes:
            fundamental = min(width_modes, key=lambda x: x["frequency"])
            harmonic = 1
            if "_H" in fundamental["type"]:
                try:
                    harmonic = int(fundamental["type"].split("_H")[1])
                except:
                    harmonic = 1
            dimensions["width"] = speed_of_sound / (2 * fundamental["frequency"] / harmonic)
            
        if height_modes:
            fundamental = min(height_modes, key=lambda x: x["frequency"])
            harmonic = 1
            if "_H" in fundamental["type"]:
                try:
                    harmonic = int(fundamental["type"].split("_H")[1])
                except:
                    harmonic = 1
            dimensions["height"] = speed_of_sound / (2 * fundamental["frequency"] / harmonic)
        
        # Calculate confidence based on number of detected modes and their Q factors
        total_modes = len(length_modes) + len(width_modes) + len(height_modes)
        if total_modes > 0:
            # Consider Q factors in confidence calculation
            avg_q = np.mean([m["q_factor"] for m in room_modes]) if room_modes else 1.0
            q_confidence = min(1.0, avg_q / 20.0)  # High Q indicates clear modes
            mode_confidence = min(1.0, total_modes / 6.0)
            dimensions["confidence"] = (mode_confidence + q_confidence) / 2
        
        return dimensions
    
    def update_audio_history(self, audio_chunk: np.ndarray):
        """Update audio history for RT60 calculation"""
        if audio_chunk is not None and len(audio_chunk) > 0:
            self._audio_history.append(audio_chunk.copy())
    
    def calculate_rt60_estimate(self, method: Optional[str] = None) -> Dict[str, float]:
        """Enhanced RT60 estimation with multiple methods"""
        calc_method = method if method is not None else self.config.rt60_method
        
        if len(self._audio_history) < self.config.min_audio_history:
            return {"rt60": 0.0, "confidence": 0.0, "method": calc_method}
        
        try:
            # Convert deque to list for processing
            audio_history = list(self._audio_history)
            
            if calc_method == "schroeder":
                return self._calculate_rt60_schroeder(audio_history)
            elif calc_method == "edt":
                return self._calculate_rt60_edt(audio_history)
            else:
                return self._calculate_rt60_simple(audio_history)
                
        except Exception as e:
            logger.error(f"RT60 calculation error: {e}")
            return {"rt60": 0.0, "confidence": 0.0, "error": str(e), "method": calc_method}

    def _calculate_rt60_schroeder(self, audio_history: List[np.ndarray]) -> Dict[str, float]:
        """Schroeder integration method for RT60"""
        # Concatenate audio history
        audio_signal = np.concatenate(audio_history)
        
        # Calculate energy decay curve
        energy = audio_signal ** 2
        
        # Backward integration (Schroeder method)
        schroeder_curve = np.cumsum(energy[::-1])[::-1]
        
        # Avoid log of zero
        schroeder_curve = np.maximum(schroeder_curve, 1e-10)
        
        # Convert to dB
        schroeder_db = 10 * np.log10(schroeder_curve / schroeder_curve[0])
        
        # Find -5dB and -35dB points for regression
        try:
            indices_5db = np.where(schroeder_db <= -5)[0]
            indices_35db = np.where(schroeder_db <= -35)[0]
            
            if len(indices_5db) == 0 or len(indices_35db) == 0:
                return {"rt60": 0.5, "confidence": 0.1, "method": "schroeder"}
            
            idx_5db = indices_5db[0]
            idx_35db = indices_35db[0]
            
            if idx_35db <= idx_5db:
                return {"rt60": 0.5, "confidence": 0.1, "method": "schroeder"}
            
            # Linear regression between -5dB and -35dB
            time_axis = np.arange(len(schroeder_db)) / self.sample_rate
            
            # Fit line to decay region
            coeffs = np.polyfit(time_axis[idx_5db:idx_35db], 
                               schroeder_db[idx_5db:idx_35db], 1)
            slope = coeffs[0]
            
            # Extrapolate to -60dB
            if slope < 0:
                rt60 = -60.0 / slope
            else:
                rt60 = 0.5  # Default if no decay
            
            # Calculate confidence based on linearity of decay
            fitted_line = np.polyval(coeffs, time_axis[idx_5db:idx_35db])
            residuals = schroeder_db[idx_5db:idx_35db] - fitted_line
            r_squared = 1 - (np.sum(residuals**2) / 
                           np.sum((schroeder_db[idx_5db:idx_35db] - 
                                  np.mean(schroeder_db[idx_5db:idx_35db]))**2))
            
            confidence = max(0.0, min(1.0, r_squared))
            
            return {
                "rt60": min(5.0, max(0.05, rt60)),
                "confidence": confidence,
                "method": "schroeder",
                "slope": slope,
                "r_squared": r_squared
            }
            
        except (IndexError, ValueError) as e:
            logger.debug(f"Schroeder method calculation error: {e}")
            return {"rt60": 0.5, "confidence": 0.0, "method": "schroeder"}

    def _calculate_rt60_edt(self, audio_history: List[np.ndarray]) -> Dict[str, float]:
        """Early Decay Time (EDT) method - uses 0 to -10dB"""
        # Similar to Schroeder but focuses on early decay
        audio_signal = np.concatenate(audio_history)
        energy = audio_signal ** 2
        
        # Calculate energy decay
        schroeder_curve = np.cumsum(energy[::-1])[::-1]
        schroeder_curve = np.maximum(schroeder_curve, 1e-10)
        schroeder_db = 10 * np.log10(schroeder_curve / schroeder_curve[0])
        
        try:
            # Find 0dB and -10dB points
            idx_0db = 0
            indices_10db = np.where(schroeder_db <= -10)[0]
            
            if len(indices_10db) == 0:
                return {"rt60": 0.5, "confidence": 0.1, "method": "edt"}
                
            idx_10db = indices_10db[0]
            
            # Calculate slope from 0 to -10dB
            time_axis = np.arange(len(schroeder_db)) / self.sample_rate
            slope = -10.0 / time_axis[idx_10db]
            
            # Extrapolate to -60dB
            edt = -60.0 / slope if slope < 0 else 0.5
            
            return {
                "rt60": min(5.0, max(0.05, edt)),
                "confidence": 0.7,  # EDT is less reliable than full Schroeder
                "method": "edt"
            }
            
        except Exception as e:
            logger.debug(f"EDT calculation error: {e}")
            return {"rt60": 0.5, "confidence": 0.0, "method": "edt"}

    def _calculate_rt60_simple(self, audio_history: List[np.ndarray]) -> Dict[str, float]:
        """Simple RT60 estimation (legacy method)"""
        # Calculate energy decay over time
        energies = [np.sum(frame**2) for frame in audio_history]
        
        if max(energies) == 0:
            return {"rt60": 0.0, "confidence": 0.0, "method": "simple"}
        
        # Normalize to dB scale
        energies_db = 10 * np.log10(np.array(energies) / max(energies) + 1e-10)
        
        # Find -60dB point (or extrapolate)
        for i, energy in enumerate(energies_db):
            if energy <= -60:
                # Calculate time to -60dB
                frame_duration = len(audio_history[0]) / self.sample_rate
                rt60 = i * frame_duration
                return {
                    "rt60": min(3.0, max(0.1, rt60)),
                    "confidence": 0.5,
                    "method": "simple"
                }
        
        # If we didn't reach -60dB, extrapolate
        if len(energies_db) > 2:
            # Linear fit to decay curve
            x = np.arange(len(energies_db))
            coeffs = np.polyfit(x, energies_db, 1)
            slope = coeffs[0]
            
            if slope < 0:
                frames_to_60db = -60 / slope
                frame_duration = len(audio_history[0]) / self.sample_rate
                rt60 = frames_to_60db * frame_duration
                return {
                    "rt60": min(3.0, max(0.1, rt60)),
                    "confidence": 0.3,
                    "method": "simple"
                }
        
        return {"rt60": 0.5, "confidence": 0.1, "method": "simple"}