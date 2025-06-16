"""
Room Mode Analyzer for OMEGA-4 Audio Analyzer
Phase 4: Extract room acoustics analysis for studio applications
"""

import numpy as np
from typing import Dict, List, Any
from collections import deque
from scipy import signal as scipy_signal


class RoomModeAnalyzer:
    """Room acoustics analysis for studio applications"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.sustained_peaks_history = deque(maxlen=300)  # 5 seconds at 60 FPS

    def detect_room_modes(self, fft_data: np.ndarray, freqs: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential room modes from sustained frequency peaks"""
        room_modes = []

        # Find peaks in room mode frequency range (30-300 Hz)
        room_mode_mask = (freqs >= 30) & (freqs <= 300)
        if not np.any(room_mode_mask):
            return room_modes

        room_mode_freqs = freqs[room_mode_mask]
        room_mode_magnitudes = fft_data[room_mode_mask]

        # Find sustained peaks
        peaks, properties = scipy_signal.find_peaks(
            room_mode_magnitudes,
            height=np.mean(room_mode_magnitudes) * 2,  # At least 2x average
            prominence=np.std(room_mode_magnitudes),  # Significant prominence
        )

        for peak_idx in peaks:
            freq = room_mode_freqs[peak_idx]
            magnitude = room_mode_magnitudes[peak_idx]

            # Estimate Q factor (sharpness of peak)
            q_factor = self.estimate_q_factor(room_mode_magnitudes, peak_idx)

            # Classify room mode type
            mode_type = self.classify_room_mode(freq)

            # Calculate severity based on magnitude and Q factor
            severity = min(1.0, (magnitude / np.mean(room_mode_magnitudes)) * (q_factor / 10.0))

            if severity > 0.3:  # Significant room mode
                room_modes.append(
                    {
                        "frequency": freq,
                        "magnitude": magnitude,
                        "q_factor": q_factor,
                        "severity": severity,
                        "type": mode_type,
                    }
                )

        return sorted(room_modes, key=lambda x: x["severity"], reverse=True)

    def estimate_q_factor(self, magnitudes: np.ndarray, peak_idx: int) -> float:
        """Estimate Q factor of a peak"""
        if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
            return 1.0

        peak_magnitude = magnitudes[peak_idx]
        half_power = peak_magnitude / np.sqrt(2)

        # Find -3dB points
        left_idx = peak_idx
        right_idx = peak_idx

        # Search left
        for i in range(peak_idx - 1, -1, -1):
            if magnitudes[i] <= half_power:
                left_idx = i
                break

        # Search right
        for i in range(peak_idx + 1, len(magnitudes)):
            if magnitudes[i] <= half_power:
                right_idx = i
                break

        # Calculate Q factor
        bandwidth = right_idx - left_idx
        if bandwidth > 0:
            q_factor = peak_idx / bandwidth
        else:
            q_factor = 10.0  # Very sharp peak

        return min(50.0, max(0.5, q_factor))

    def classify_room_mode(self, frequency: float) -> str:
        """Classify room mode type based on frequency"""
        if 30 <= frequency <= 80:
            return "axial_length"  # Room length modes
        elif 80 <= frequency <= 120:
            return "axial_width"  # Room width modes
        elif 120 <= frequency <= 200:
            return "axial_height"  # Room height modes
        elif 200 <= frequency <= 300:
            return "tangential"  # Tangential modes
        else:
            return "oblique"  # Oblique modes
    
    def estimate_room_dimensions(self, room_modes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate room dimensions from detected modes"""
        # Speed of sound in air at room temperature
        speed_of_sound = 343.0  # m/s
        
        dimensions = {
            "length": 0.0,
            "width": 0.0,
            "height": 0.0,
            "confidence": 0.0
        }
        
        # Group modes by type
        length_modes = [m for m in room_modes if m["type"] == "axial_length"]
        width_modes = [m for m in room_modes if m["type"] == "axial_width"]
        height_modes = [m for m in room_modes if m["type"] == "axial_height"]
        
        # Estimate dimensions from fundamental modes
        if length_modes:
            # Room length = speed_of_sound / (2 * fundamental_frequency)
            fundamental = min(length_modes, key=lambda x: x["frequency"])
            dimensions["length"] = speed_of_sound / (2 * fundamental["frequency"])
            
        if width_modes:
            fundamental = min(width_modes, key=lambda x: x["frequency"])
            dimensions["width"] = speed_of_sound / (2 * fundamental["frequency"])
            
        if height_modes:
            fundamental = min(height_modes, key=lambda x: x["frequency"])
            dimensions["height"] = speed_of_sound / (2 * fundamental["frequency"])
        
        # Calculate confidence based on number of detected modes
        total_modes = len(length_modes) + len(width_modes) + len(height_modes)
        if total_modes > 0:
            dimensions["confidence"] = min(1.0, total_modes / 6.0)
        
        return dimensions
    
    def calculate_rt60_estimate(self, audio_history: List[np.ndarray]) -> float:
        """Estimate RT60 (reverb time) from audio decay"""
        # This is a simplified estimation
        # Real RT60 measurement requires controlled acoustic conditions
        
        if len(audio_history) < 10:
            return 0.0
        
        # Calculate energy decay over time
        energies = [np.sum(frame**2) for frame in audio_history]
        
        if max(energies) == 0:
            return 0.0
        
        # Normalize to dB scale
        energies_db = 10 * np.log10(np.array(energies) / max(energies) + 1e-10)
        
        # Find -60dB point (or extrapolate)
        for i, energy in enumerate(energies_db):
            if energy <= -60:
                # Calculate time to -60dB
                time_samples = i
                rt60 = (time_samples / self.sample_rate) * (len(audio_history[0]) / self.sample_rate)
                return min(3.0, max(0.1, rt60))  # Clamp to reasonable range
        
        # If we didn't reach -60dB, extrapolate
        if len(energies_db) > 2:
            # Linear fit to decay curve
            decay_rate = (energies_db[-1] - energies_db[0]) / len(energies_db)
            if decay_rate < 0:
                samples_to_60db = -60 / decay_rate
                rt60 = (samples_to_60db / self.sample_rate) * (len(audio_history[0]) / self.sample_rate)
                return min(3.0, max(0.1, rt60))
        
        return 0.5  # Default estimate