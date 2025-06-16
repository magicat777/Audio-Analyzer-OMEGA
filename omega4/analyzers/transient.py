"""
Transient Analyzer for OMEGA-4 Audio Analyzer
Phase 4: Extract transient analysis for attack detection and dynamics
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque
from scipy import signal as scipy_signal


class TransientAnalyzer:
    """Transient analysis for attack detection and dynamics"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.envelope_history = deque(maxlen=int(0.5 * 60))  # 500ms at 60 FPS

    def analyze_transients(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect and analyze transients in audio"""
        if len(audio_data) < 64:
            return {"transients_detected": 0, "attack_time": 0.0, "punch_factor": 0.0}

        # Calculate envelope using Hilbert transform
        analytic_signal = scipy_signal.hilbert(audio_data)
        envelope = np.abs(analytic_signal)

        # Smooth envelope
        envelope_smooth = scipy_signal.savgol_filter(
            envelope, min(21, len(envelope) // 2 * 2 + 1), 3
        )

        self.envelope_history.append(np.mean(envelope_smooth))

        # Detect attacks using derivative
        if len(envelope_smooth) > 10:
            envelope_diff = np.diff(envelope_smooth)

            # Find attack points
            attack_threshold = np.std(envelope_diff) * 2.0
            attack_points = np.where(envelope_diff > attack_threshold)[0]

            # Calculate attack characteristics
            attack_time = self.calculate_attack_time(envelope_smooth, attack_points)
            punch_factor = self.calculate_punch_factor(envelope_smooth, attack_points)

            return {
                "transients_detected": len(attack_points),
                "attack_time": attack_time,
                "punch_factor": punch_factor,
                "envelope_peak": np.max(envelope_smooth),
                "envelope_rms": np.sqrt(np.mean(envelope_smooth**2)),
            }

        return {"transients_detected": 0, "attack_time": 0.0, "punch_factor": 0.0}

    def calculate_attack_time(self, envelope: np.ndarray, attack_points: np.ndarray) -> float:
        """Calculate average attack time in milliseconds"""
        if len(attack_points) == 0:
            return 0.0

        attack_times = []

        for attack_idx in attack_points:
            # Find 10% and 90% points of attack
            if attack_idx > 10 and attack_idx < len(envelope) - 10:
                start_idx = max(0, attack_idx - 10)
                peak_val = envelope[attack_idx]

                # Find 10% point
                ten_percent = peak_val * 0.1
                ten_percent_idx = start_idx
                for i in range(start_idx, attack_idx):
                    if envelope[i] >= ten_percent:
                        ten_percent_idx = i
                        break

                # Find 90% point
                ninety_percent = peak_val * 0.9
                ninety_percent_idx = attack_idx
                for i in range(ten_percent_idx, min(len(envelope), attack_idx + 10)):
                    if envelope[i] >= ninety_percent:
                        ninety_percent_idx = i
                        break

                # Calculate attack time
                attack_samples = ninety_percent_idx - ten_percent_idx
                attack_time_ms = (attack_samples / self.sample_rate) * 1000
                attack_times.append(attack_time_ms)

        return np.mean(attack_times) if attack_times else 0.0

    def calculate_punch_factor(self, envelope: np.ndarray, attack_points: np.ndarray) -> float:
        """Calculate punch factor (attack sharpness)"""
        if len(attack_points) == 0:
            return 0.0

        punch_scores = []

        for attack_idx in attack_points:
            if attack_idx > 5 and attack_idx < len(envelope) - 5:
                # Calculate slope at attack point
                before_vals = envelope[attack_idx - 5 : attack_idx]
                after_vals = envelope[attack_idx : attack_idx + 5]

                if len(before_vals) > 0 and len(after_vals) > 0:
                    slope = np.mean(after_vals) - np.mean(before_vals)
                    punch_scores.append(max(0.0, slope))

        return np.mean(punch_scores) if punch_scores else 0.0
    
    def get_envelope_history(self) -> List[float]:
        """Get envelope history for visualization"""
        return list(self.envelope_history)
    
    def detect_transient_type(self, audio_data: np.ndarray, attack_point: int) -> str:
        """Classify transient type (kick, snare, hi-hat, etc.)"""
        # Simple classification based on frequency content at attack point
        if attack_point < len(audio_data) - 256:
            segment = audio_data[attack_point:attack_point + 256]
            
            # Get frequency content
            fft = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
            
            # Calculate spectral centroid
            if np.sum(fft) > 0:
                centroid = np.sum(freqs * fft) / np.sum(fft)
                
                # Classify based on centroid
                if centroid < 200:
                    return "kick"
                elif centroid < 800:
                    return "snare"
                elif centroid < 3000:
                    return "mid_percussion"
                else:
                    return "hi_hat"
            
        return "unknown"