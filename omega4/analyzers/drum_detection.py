"""
Drum Detection Analyzers for OMEGA-4 Audio Analyzer
Phase 4: Extract drum detection classes with multi-band analysis
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from collections import deque


class EnhancedKickDetector:
    """Professional kick/bass drum detection with multi-band analysis and value persistence"""

    def __init__(self, sample_rate: int = 48000, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity

        # Multi-band frequency ranges (industry standard)
        self.sub_bass_range = (20, 60)  # Sub-bass punch
        self.kick_body_range = (60, 120)  # Kick body/thump
        self.kick_click_range = (2000, 5000)  # Kick click/beater

        # Spectral flux calculation for each band
        self.prev_magnitude = None
        self.sub_flux_history = deque(maxlen=21)
        self.body_flux_history = deque(maxlen=21)
        self.click_flux_history = deque(maxlen=21)

        # Adaptive thresholding
        self.onset_history = deque(maxlen=43)
        self.min_kick_interval = 0.1
        self.last_kick_time = 0

        # Detection state with persistence
        self.kick_detected = False
        self.kick_strength = 0.0
        self.kick_velocity = 0.0

        # Value persistence for display
        self.display_strength = 0.0
        self.display_velocity = 0
        self.decay_rate = 0.92
        self.hold_time = 0.2
        self.last_detection_time = 0

    def calculate_band_flux(
        self, magnitude: np.ndarray, start_bin: int, end_bin: int, band_type: str
    ) -> float:
        """Calculate spectral flux for specific frequency band"""
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude.copy()
            return 0.0

        # Calculate positive spectral differences in band
        diff = magnitude[start_bin:end_bin] - self.prev_magnitude[start_bin:end_bin]
        positive_flux = np.sum(np.maximum(diff, 0))

        # Store in appropriate history
        if band_type == "sub":
            self.sub_flux_history.append(positive_flux)
        elif band_type == "body":
            self.body_flux_history.append(positive_flux)
        elif band_type == "click":
            self.click_flux_history.append(positive_flux)

        return positive_flux

    def calculate_adaptive_threshold(self, flux_history: deque) -> float:
        """Calculate adaptive threshold using median + scaled MAD"""
        if len(flux_history) < 10:
            return 0.0

        flux_array = np.array(list(flux_history))
        median = np.median(flux_array)
        mad = np.median(np.abs(flux_array - median))

        return median + self.sensitivity * 2.8 * mad

    def detect_kick_onset(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Enhanced kick detection with multi-band analysis"""
        current_time = time.time()
        nyquist = self.sample_rate / 2

        # Calculate flux in each frequency band
        sub_start = int(self.sub_bass_range[0] * len(magnitude) / nyquist)
        sub_end = int(self.sub_bass_range[1] * len(magnitude) / nyquist)
        sub_flux = self.calculate_band_flux(magnitude, sub_start, sub_end, "sub")

        body_start = int(self.kick_body_range[0] * len(magnitude) / nyquist)
        body_end = int(self.kick_body_range[1] * len(magnitude) / nyquist)
        body_flux = self.calculate_band_flux(magnitude, body_start, body_end, "body")

        click_start = int(self.kick_click_range[0] * len(magnitude) / nyquist)
        click_end = int(self.kick_click_range[1] * len(magnitude) / nyquist)
        click_flux = self.calculate_band_flux(magnitude, click_start, click_end, "click")

        self.prev_magnitude = magnitude.copy()

        # Calculate adaptive thresholds for each band
        sub_threshold = self.calculate_adaptive_threshold(self.sub_flux_history)
        body_threshold = self.calculate_adaptive_threshold(self.body_flux_history)
        click_threshold = self.calculate_adaptive_threshold(self.click_flux_history)

        # Multi-criteria detection
        kick_detected = False
        kick_strength = 0.0
        kick_velocity = 0.0

        time_since_last = current_time - self.last_kick_time

        if (
            len(self.sub_flux_history) >= 10
            and len(self.body_flux_history) >= 10
            and time_since_last > self.min_kick_interval
        ):

            # Multi-band detection criteria
            sub_hit = sub_flux > sub_threshold
            body_hit = body_flux > body_threshold
            click_present = click_flux > click_threshold * 0.7

            # Kick requires strong sub-bass AND body presence
            if sub_hit and body_hit:
                kick_detected = True

                # Enhanced strength calculation combining all bands
                sub_strength = sub_flux / (sub_threshold + 1e-6)
                body_strength = body_flux / (body_threshold + 1e-6)
                click_strength = click_flux / (click_threshold + 1e-6) if click_threshold > 0 else 0

                # Weighted combination
                kick_strength = min(
                    1.0, (sub_strength * 0.4 + body_strength * 0.5 + click_strength * 0.1)
                )
                kick_velocity = min(127, int(kick_strength * 127))

                self.last_kick_time = current_time
                self.last_detection_time = current_time

        # Value persistence system
        if kick_detected and kick_strength > 0:
            self.display_strength = kick_strength
            self.display_velocity = kick_velocity
        else:
            # Hold values for specified time, then decay
            time_since_detection = current_time - self.last_detection_time
            if time_since_detection > self.hold_time:
                self.display_strength *= self.decay_rate
                self.display_velocity = int(self.display_velocity * self.decay_rate)
            # Values below threshold are zeroed
            if self.display_strength < 0.05:
                self.display_strength = 0.0
                self.display_velocity = 0

        self.kick_detected = kick_detected
        self.kick_strength = kick_strength
        self.kick_velocity = kick_velocity

        return {
            "kick_detected": kick_detected,
            "kick_strength": kick_strength,
            "kick_velocity": kick_velocity,
            "display_strength": self.display_strength,
            "display_velocity": self.display_velocity,
            "sub_flux": sub_flux,
            "body_flux": body_flux,
            "click_flux": click_flux,
            "sub_threshold": sub_threshold,
            "body_threshold": body_threshold,
            "multi_band_score": kick_strength,
        }


class EnhancedSnareDetector:
    """Professional snare detection with multi-band analysis and value persistence"""

    def __init__(self, sample_rate: int = 48000, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity

        # Enhanced frequency ranges for snare detection
        self.snare_fundamental_range = (150, 400)
        self.snare_body_range = (400, 1000)
        self.snare_snap_range = (2000, 8000)
        self.snare_rattle_range = (8000, 15000)

        # Multi-resolution analysis
        self.prev_magnitude = None
        self.fundamental_flux_history = deque(maxlen=21)
        self.body_flux_history = deque(maxlen=21)
        self.snap_flux_history = deque(maxlen=21)
        self.rattle_flux_history = deque(maxlen=21)
        self.centroid_history = deque(maxlen=21)

        # Adaptive thresholding
        self.min_snare_interval = 0.08
        self.last_snare_time = 0

        # Detection state with persistence
        self.snare_detected = False
        self.snare_strength = 0.0
        self.snare_velocity = 0.0

        # Value persistence for display
        self.display_strength = 0.0
        self.display_velocity = 0
        self.decay_rate = 0.90
        self.hold_time = 0.15
        self.last_detection_time = 0

    def calculate_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """Calculate spectral centroid for snare characterization"""
        freqs = np.fft.rfftfreq(len(magnitude) * 2 - 1, 1 / self.sample_rate)

        # Focus on snare-relevant frequencies
        nyquist = self.sample_rate / 2
        relevant_start = int(self.snare_fundamental_range[0] * len(magnitude) / nyquist)
        relevant_end = int(self.snare_rattle_range[1] * len(magnitude) / nyquist)

        relevant_freqs = freqs[relevant_start:relevant_end]
        relevant_magnitude = magnitude[relevant_start:relevant_end]

        if np.sum(relevant_magnitude) > 0:
            centroid = np.sum(relevant_freqs * relevant_magnitude) / np.sum(relevant_magnitude)
        else:
            centroid = 0

        return centroid

    def calculate_multi_band_flux(self, magnitude: np.ndarray) -> Tuple[float, float, float, float]:
        """Calculate spectral flux in all snare frequency ranges"""
        if self.prev_magnitude is None:
            self.prev_magnitude = magnitude.copy()
            return 0.0, 0.0, 0.0, 0.0

        nyquist = self.sample_rate / 2

        # Fundamental range flux
        fund_start = int(self.snare_fundamental_range[0] * len(magnitude) / nyquist)
        fund_end = int(self.snare_fundamental_range[1] * len(magnitude) / nyquist)
        fund_diff = magnitude[fund_start:fund_end] - self.prev_magnitude[fund_start:fund_end]
        fundamental_flux = np.sum(np.maximum(fund_diff, 0))

        # Body range flux
        body_start = int(self.snare_body_range[0] * len(magnitude) / nyquist)
        body_end = int(self.snare_body_range[1] * len(magnitude) / nyquist)
        body_diff = magnitude[body_start:body_end] - self.prev_magnitude[body_start:body_end]
        body_flux = np.sum(np.maximum(body_diff, 0))

        # Snap range flux
        snap_start = int(self.snare_snap_range[0] * len(magnitude) / nyquist)
        snap_end = int(self.snare_snap_range[1] * len(magnitude) / nyquist)
        snap_diff = magnitude[snap_start:snap_end] - self.prev_magnitude[snap_start:snap_end]
        snap_flux = np.sum(np.maximum(snap_diff, 0))

        # Rattle range flux
        rattle_start = int(self.snare_rattle_range[0] * len(magnitude) / nyquist)
        rattle_end = int(self.snare_rattle_range[1] * len(magnitude) / nyquist)
        rattle_diff = (
            magnitude[rattle_start:rattle_end] - self.prev_magnitude[rattle_start:rattle_end]
        )
        rattle_flux = np.sum(np.maximum(rattle_diff, 0))

        self.prev_magnitude = magnitude.copy()
        return fundamental_flux, body_flux, snap_flux, rattle_flux

    def detect_snare_onset(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Enhanced snare detection using multi-band analysis"""
        current_time = time.time()

        # Calculate spectral features
        fundamental_flux, body_flux, snap_flux, rattle_flux = self.calculate_multi_band_flux(
            magnitude
        )
        centroid = self.calculate_spectral_centroid(magnitude)

        # Store history
        self.fundamental_flux_history.append(fundamental_flux)
        self.body_flux_history.append(body_flux)
        self.snap_flux_history.append(snap_flux)
        self.rattle_flux_history.append(rattle_flux)
        self.centroid_history.append(centroid)

        # Calculate adaptive thresholds
        snare_detected = False
        snare_strength = 0.0
        snare_velocity = 0.0

        if len(self.fundamental_flux_history) >= 10:
            # Calculate thresholds for each band
            fund_array = np.array(list(self.fundamental_flux_history))
            fund_median = np.median(fund_array)
            fund_mad = np.median(np.abs(fund_array - fund_median))
            fund_threshold = fund_median + self.sensitivity * 2.5 * fund_mad

            body_array = np.array(list(self.body_flux_history))
            body_median = np.median(body_array)
            body_mad = np.median(np.abs(body_array - body_median))
            body_threshold = body_median + self.sensitivity * 2.3 * body_mad

            snap_array = np.array(list(self.snap_flux_history))
            snap_median = np.median(snap_array)
            snap_mad = np.median(np.abs(snap_array - snap_median))
            snap_threshold = snap_median + self.sensitivity * 2.0 * snap_mad

            # Spectral centroid in snare range
            centroid_in_range = 800 <= centroid <= 6000

            # Time-based gating
            time_since_last = current_time - self.last_snare_time

            # Multi-criteria detection
            fundamental_hit = fundamental_flux > fund_threshold
            body_hit = body_flux > body_threshold
            snap_hit = snap_flux > snap_threshold
            timing_ok = time_since_last > self.min_snare_interval

            # Enhanced snare detection (requires fundamental + body + snap)
            if fundamental_hit and body_hit and snap_hit and timing_ok and centroid_in_range:
                snare_detected = True

                # Multi-band strength calculation
                fund_strength = fundamental_flux / (fund_threshold + 1e-6)
                body_strength = body_flux / (body_threshold + 1e-6)
                snap_strength = snap_flux / (snap_threshold + 1e-6)

                # Weighted combination (snap is most characteristic of snare)
                snare_strength = min(
                    1.0, (fund_strength * 0.2 + body_strength * 0.3 + snap_strength * 0.5)
                )

                snare_velocity = min(127, int(snare_strength * 127))
                self.last_snare_time = current_time
                self.last_detection_time = current_time

        # Value persistence system
        if snare_detected and snare_strength > 0:
            self.display_strength = snare_strength
            self.display_velocity = snare_velocity
        else:
            # Hold values for specified time, then decay
            time_since_detection = current_time - self.last_detection_time
            if time_since_detection > self.hold_time:
                self.display_strength *= self.decay_rate
                self.display_velocity = int(self.display_velocity * self.decay_rate)
            # Values below threshold are zeroed
            if self.display_strength < 0.05:
                self.display_strength = 0.0
                self.display_velocity = 0

        self.snare_detected = snare_detected
        self.snare_strength = snare_strength
        self.snare_velocity = snare_velocity

        return {
            "snare_detected": snare_detected,
            "snare_strength": snare_strength,
            "snare_velocity": snare_velocity,
            "display_strength": self.display_strength,
            "display_velocity": self.display_velocity,
            "fundamental_flux": fundamental_flux,
            "body_flux": body_flux,
            "snap_flux": snap_flux,
            "rattle_flux": rattle_flux,
            "spectral_centroid": centroid,
            "multi_band_score": snare_strength,
        }


class GrooveAnalyzer:
    """Industry-grade groove pattern recognition and tempo analysis"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.beat_grid = deque(maxlen=64)
        self.tempo_candidates = deque(maxlen=16)
        self.stable_bpm = 0
        self.beat_confidence = 0.0
        self.groove_stability = 0.0

        # Extended groove pattern library
        self.groove_patterns = {
            "four_four_basic": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "backbeat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "shuffle": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            "latin_clave": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            "breakbeat": [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "drum_and_bass": [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            "reggae": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "rock_basic": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }

        # Tempo analysis
        self.kick_intervals = deque(maxlen=8)
        self.snare_intervals = deque(maxlen=8)
        self.last_kick_time = 0
        self.last_snare_time = 0

        # Pattern matching state
        self.current_pattern = "unknown"
        self.pattern_confidence = 0.0
        self.pattern_lock_time = 0
        self.pattern_lock_duration = 8.0

    def estimate_tempo_from_intervals(self, intervals: List[float]) -> float:
        """Estimate tempo using autocorrelation of inter-onset intervals"""
        if len(intervals) < 3:
            return 0

        # Filter reasonable intervals (0.2-2.0 seconds = 30-300 BPM)
        valid_intervals = [i for i in intervals if 0.2 <= i <= 2.0]

        if len(valid_intervals) < 2:
            return 0

        # Use median for stability
        median_interval = np.median(valid_intervals)

        # Convert to BPM
        bpm = 60.0 / median_interval

        # Quantize to common BPM values for stability
        common_bpms = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        closest_bpm = min(common_bpms, key=lambda x: abs(x - bpm))

        # Only use if close enough to a common BPM
        if abs(bpm - closest_bpm) < 8:
            return closest_bpm

        return bpm

    def analyze_groove(
        self, kick_detected: bool, snare_detected: bool, kick_strength: float, snare_strength: float
    ) -> Dict[str, Any]:
        """Analyze musical groove pattern in real-time"""
        current_time = time.time()

        # Track kick timing
        if kick_detected:
            if self.last_kick_time > 0:
                interval = current_time - self.last_kick_time
                self.kick_intervals.append(interval)
            self.last_kick_time = current_time
            self.beat_grid.append((current_time, "kick", kick_strength))

        # Track snare timing
        if snare_detected:
            if self.last_snare_time > 0:
                interval = current_time - self.last_snare_time
                self.snare_intervals.append(interval)
            self.last_snare_time = current_time
            self.beat_grid.append((current_time, "snare", snare_strength))

        # Tempo estimation
        all_intervals = list(self.kick_intervals) + list(self.snare_intervals)

        if len(all_intervals) >= 3:
            tempo_estimate = self.estimate_tempo_from_intervals(all_intervals)

            if tempo_estimate > 0:
                self.tempo_candidates.append(tempo_estimate)

            # Stable BPM using weighted average of recent estimates
            if len(self.tempo_candidates) >= 4:
                recent_tempos = list(self.tempo_candidates)
                weights = np.exp(np.linspace(-1, 0, len(recent_tempos)))
                self.stable_bpm = np.average(recent_tempos, weights=weights)

                # Calculate tempo stability
                tempo_std = np.std(recent_tempos)
                self.groove_stability = max(0, 1.0 - (tempo_std / 20.0))

        # Beat confidence based on tempo stability
        self.beat_confidence = self.groove_stability * 0.8

        return {
            "stable_bpm": round(self.stable_bpm, 1),
            "groove_pattern": self.current_pattern,
            "pattern_confidence": self.pattern_confidence,
            "beat_confidence": self.beat_confidence,
            "groove_stability": self.groove_stability,
            "tempo_std": (
                np.std(list(self.tempo_candidates)) if len(self.tempo_candidates) > 1 else 0
            ),
            "active_beats": len(self.beat_grid),
        }


class EnhancedDrumDetector:
    """Enhanced drum detection system with all industry improvements"""

    def __init__(self, sample_rate: int = 48000, sensitivity: float = 1.0):
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity

        # Initialize enhanced detectors
        self.kick_detector = EnhancedKickDetector(sample_rate, sensitivity)
        self.snare_detector = EnhancedSnareDetector(sample_rate, sensitivity)
        self.groove_analyzer = GrooveAnalyzer(sample_rate)

        # Legacy BPM calculation (maintained for compatibility)
        self.kick_times = deque(maxlen=8)
        self.current_bpm = 0

        # Enhanced pattern detection
        self.pattern_history = deque(maxlen=32)

    def process_audio(self, fft_data: np.ndarray, band_values: np.ndarray) -> Dict[str, Any]:
        """Process audio for enhanced drum detection"""
        current_time = time.time()

        # Detect kicks and snares with enhancements
        kick_info = self.kick_detector.detect_kick_onset(fft_data)
        snare_info = self.snare_detector.detect_snare_onset(fft_data)

        # Groove analysis
        groove_info = self.groove_analyzer.analyze_groove(
            kick_info["kick_detected"],
            snare_info["snare_detected"],
            kick_info["kick_strength"],
            snare_info["snare_strength"],
        )

        # Legacy BPM calculation (for compatibility)
        if kick_info["kick_detected"]:
            self.kick_times.append(current_time)
            self.pattern_history.append(("kick", current_time, kick_info["kick_velocity"]))

            if len(self.kick_times) >= 2:
                recent_intervals = []
                for i in range(1, len(self.kick_times)):
                    interval = self.kick_times[i] - self.kick_times[i - 1]
                    if 0.3 < interval < 2.0:
                        recent_intervals.append(interval)

                if recent_intervals:
                    avg_interval = np.mean(recent_intervals)
                    self.current_bpm = 60.0 / avg_interval

        # Update pattern history for snares
        if snare_info["snare_detected"]:
            self.pattern_history.append(("snare", current_time, snare_info["snare_velocity"]))

        # Detect simultaneous hits
        simultaneous_hit = kick_info["kick_detected"] and snare_info["snare_detected"]

        return {
            "kick": kick_info,
            "snare": snare_info,
            "groove": groove_info,
            "bpm": max(self.current_bpm, groove_info["stable_bpm"]),
            "simultaneous_hit": simultaneous_hit,
            "pattern_length": len(self.pattern_history),
            "beat_detected": kick_info["kick_detected"] or snare_info["snare_detected"],
            "beat_strength": max(kick_info["kick_strength"], snare_info["snare_strength"]),
        }

    def set_sensitivity(self, sensitivity: float):
        """Update sensitivity for both detectors"""
        self.sensitivity = sensitivity
        self.kick_detector.sensitivity = sensitivity
        self.snare_detector.sensitivity = sensitivity