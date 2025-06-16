#!/usr/bin/env python3
"""
Live Audio Analyzer v5 - Professional Studio-Grade Analysis
Advanced real-time audio analysis with enhanced low-end detail and professional features

Version 5 Features:
- Multi-resolution FFT for enhanced bass detail
- Professional metering (LUFS, K-weighting, True Peak)
- Advanced harmonic analysis and instrument identification
- Phase coherence analysis for stereo imaging
- Transient analysis and room mode detection
- Psychoacoustic bass enhancement
- Studio-grade visualization tools
"""

import numpy as np
import pygame
import sys
import threading
import queue
from queue import Queue, Empty
import time
import os
import subprocess
from collections import deque
from typing import Dict, List, Tuple, Any
import argparse
from scipy import signal as scipy_signal

# Add path for industry voice detection module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "voice_detection"))

# Import after path is set
from industry_voice_detection import IndustryVoiceDetector

# Audio analysis constants
SAMPLE_RATE = 48000
CHUNK_SIZE = 512  # Reduced for lower latency (10.7ms instead of 21.3ms)
BARS_DEFAULT = 1024  # Optimized for vocal clarity while maintaining >45 FPS
BARS_MAX = 1536
MAX_FREQ = 20000
FFT_SIZE_BASE = 4096  # Reduced from 8192 for better latency (85ms window vs 170ms)


class MultiResolutionFFT:
    """Multi-resolution FFT analysis for enhanced low-end detail"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
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

            if pos + chunk_len <= len(buffer):
                buffer[pos : pos + chunk_len] = audio_chunk
            else:
                first_part = len(buffer) - pos
                buffer[pos:] = audio_chunk[:first_part]
                buffer[: chunk_len - first_part] = audio_chunk[first_part:]

            self.buffer_positions[i] = (pos + chunk_len) % len(buffer)

            # Extract latest samples for FFT
            audio_data = np.zeros(fft_size, dtype=np.float32)
            if pos >= fft_size:
                audio_data[:] = buffer[pos - fft_size : pos]
            elif pos > 0:
                audio_data[-pos:] = buffer[:pos]
                audio_data[:-pos] = buffer[-(fft_size - pos) :]

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


class ProfessionalMetering:
    """Professional audio metering standards (LUFS, K-weighting, True Peak)"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

        # LUFS measurement history
        self.lufs_momentary_history = deque(maxlen=int(0.4 * 60))  # 400ms at 60 FPS
        self.lufs_short_term_history = deque(maxlen=int(3.0 * 60))  # 3s at 60 FPS
        self.lufs_integrated_history = deque(maxlen=int(60 * 60))  # 60s max integration

        # True peak detection
        self.peak_history = deque(maxlen=int(1.0 * 60))  # 1s peak hold

        # K-weighting filter coefficients (simplified implementation)
        self.k_weighting_filter = self.create_k_weighting_filter()

        # Gate threshold for integrated LUFS
        self.gate_threshold = -70.0  # LUFS

        # Current values
        self.current_lufs = {
            "momentary": -100.0,
            "short_term": -100.0,
            "integrated": -100.0,
            "range": 0.0,
        }
        self.current_true_peak = -100.0

    def create_k_weighting_filter(self):
        """Create K-weighting filter (simplified for real-time use)"""
        # High-shelf filter at 1500 Hz (+4 dB)
        # High-pass filter at 38 Hz

        # Simplified biquad coefficients for K-weighting approximation
        nyquist = self.sample_rate / 2

        # High-pass at 38 Hz
        hp_freq = 38 / nyquist
        hp_b, hp_a = scipy_signal.butter(2, hp_freq, btype="high")

        # High-shelf at 1500 Hz (+4 dB)
        shelf_freq = 1500 / nyquist
        shelf_gain_db = 4.0
        shelf_gain_linear = 10 ** (shelf_gain_db / 20)

        # Simplified shelf filter
        shelf_b, shelf_a = scipy_signal.iirfilter(
            2, shelf_freq, btype="high", ftype="butter", output="ba"
        )

        return {
            "hp_b": hp_b,
            "hp_a": hp_a,
            "shelf_b": shelf_b,
            "shelf_a": shelf_a,
            "shelf_gain": shelf_gain_linear,
        }

    def apply_k_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply K-weighting filter to audio data"""
        # Check if we have very low signal
        input_rms = np.sqrt(np.mean(audio_data**2))
        if input_rms < 1e-6:
            return np.zeros_like(audio_data)  # Return zeros for very quiet signal
            
        # Apply high-pass filter
        filtered = scipy_signal.filtfilt(
            self.k_weighting_filter["hp_b"], self.k_weighting_filter["hp_a"], audio_data
        )

        # Apply simplified high-shelf boost
        shelf_filtered = scipy_signal.filtfilt(
            self.k_weighting_filter["shelf_b"], self.k_weighting_filter["shelf_a"], filtered
        )

        # Combine with original for shelf effect
        result = filtered + (shelf_filtered - filtered) * 0.3  # Simplified shelf

        return result

    def calculate_lufs(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate LUFS measurements"""
        if len(audio_data) == 0:
            return self.current_lufs

        # Apply K-weighting
        weighted = self.apply_k_weighting(audio_data)

        # Calculate mean square power
        mean_square = np.mean(weighted**2)

        # Convert to LUFS
        if mean_square > 1e-10:
            lufs_instantaneous = -0.691 + 10 * np.log10(mean_square)
        else:
            lufs_instantaneous = -100.0

        # Update histories with blocks of audio
        # Each audio chunk represents a time block, not a single frame
        block_duration_ms = (len(audio_data) / self.sample_rate) * 1000
        
        # Add the instantaneous value to histories
        self.lufs_momentary_history.append(lufs_instantaneous)
        self.lufs_short_term_history.append(lufs_instantaneous)
        self.lufs_integrated_history.append(lufs_instantaneous)

        # Calculate momentary (400ms window)
        # Remove old values to maintain 400ms window
        while len(self.lufs_momentary_history) > 0:
            total_duration = len(self.lufs_momentary_history) * block_duration_ms
            if total_duration > 400:  # 400ms window
                self.lufs_momentary_history.popleft()
            else:
                break
        
        if len(self.lufs_momentary_history) > 0:
            momentary_values = list(self.lufs_momentary_history)
            self.current_lufs["momentary"] = self.calculate_gated_loudness(momentary_values)

        # Calculate short-term (3s window)
        while len(self.lufs_short_term_history) > 0:
            total_duration = len(self.lufs_short_term_history) * block_duration_ms
            if total_duration > 3000:  # 3 second window
                self.lufs_short_term_history.popleft()
            else:
                break
                
        if len(self.lufs_short_term_history) > 0:
            short_term_values = list(self.lufs_short_term_history)
            self.current_lufs["short_term"] = self.calculate_gated_loudness(short_term_values)

        # Calculate integrated (entire duration with gating)
        if len(self.lufs_integrated_history) * block_duration_ms >= 1000:  # At least 1s
            integrated_values = list(self.lufs_integrated_history)
            self.current_lufs["integrated"] = self.calculate_gated_loudness(integrated_values)
            self.current_lufs["range"] = self.calculate_loudness_range(integrated_values)

        return self.current_lufs

    def calculate_gated_loudness(self, values: List[float]) -> float:
        """Calculate gated loudness per ITU-R BS.1770-4"""
        if not values:
            return -100.0

        # First pass: calculate ungated loudness
        valid_values = [v for v in values if v > -100.0]
        if not valid_values:
            return -100.0

        ungated_loudness = 10 * np.log10(np.mean(10 ** (np.array(valid_values) / 10)))

        # Second pass: apply absolute gate at -70 LUFS
        absolute_gate = -70.0
        gated_values = [v for v in valid_values if v >= absolute_gate]

        if not gated_values:
            return -100.0

        # Third pass: apply relative gate at ungated_loudness - 10 LU
        relative_gate = ungated_loudness - 10.0
        final_values = [v for v in gated_values if v >= relative_gate]

        if not final_values:
            return ungated_loudness

        return 10 * np.log10(np.mean(10 ** (np.array(final_values) / 10)))

    def calculate_loudness_range(self, values: List[float]) -> float:
        """Calculate loudness range (LRA) per EBU R128"""
        if len(values) < 60:  # Need at least 1s of data
            return 0.0

        # Apply gating
        gated_values = [v for v in values if v >= -70.0]  # Absolute gate

        if len(gated_values) < 10:
            return 0.0

        # Calculate 10th and 95th percentiles
        percentiles = np.percentile(gated_values, [10, 95])
        lra = percentiles[1] - percentiles[0]

        return max(0.0, lra)

    def calculate_true_peak(self, audio_data: np.ndarray) -> float:
        """Calculate True Peak level per ITU-R BS.1770-4"""
        if len(audio_data) == 0:
            return -100.0

        # Upsample by 4x for true peak detection
        upsampled = scipy_signal.resample(audio_data, len(audio_data) * 4)

        # Find peak
        peak_linear = np.max(np.abs(upsampled))

        if peak_linear > 1e-10:
            peak_db = 20 * np.log10(peak_linear)
        else:
            peak_db = -100.0

        # Update peak history
        self.peak_history.append(peak_db)

        # Return peak hold value
        self.current_true_peak = max(self.peak_history) if self.peak_history else -100.0

        return self.current_true_peak


class HarmonicAnalyzer:
    """Advanced harmonic analysis and instrument identification"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Instrument harmonic profiles
        self.instrument_profiles = {
            "kick_drum": {
                "fundamental_range": (40, 120),
                "harmonic_pattern": [1.0, 0.3, 0.1, 0.05],  # Decreasing harmonics
                "characteristics": "low_fundamental_strong",
            },
            "snare_drum": {
                "fundamental_range": (150, 400),
                "harmonic_pattern": [0.8, 1.0, 0.6, 0.4, 0.2],  # Complex pattern
                "characteristics": "broadband_with_peaks",
            },
            "bass_guitar": {
                "fundamental_range": (41, 200),
                "harmonic_pattern": [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
                "characteristics": "strong_harmonics",
            },
            "electric_guitar": {
                "fundamental_range": (82, 880),
                "harmonic_pattern": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
                "characteristics": "rich_harmonics",
            },
            "piano": {
                "fundamental_range": (27, 4186),
                "harmonic_pattern": [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
                "characteristics": "clean_harmonics",
            },
            "voice_female": {
                "fundamental_range": (150, 500),
                "harmonic_pattern": [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
                "characteristics": "formant_enhanced",
            },
            "voice_male": {
                "fundamental_range": (85, 300),
                "harmonic_pattern": [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
                "characteristics": "formant_enhanced",
            },
        }

        # Harmonic detection history
        self.harmonic_history = deque(maxlen=30)
        self.instrument_confidence = {}

    def detect_harmonic_series(self, fft_data: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """Detect harmonic series and identify potential instruments"""
        harmonics_found = []

        # Find peaks in the spectrum
        peaks, properties = scipy_signal.find_peaks(
            fft_data,
            height=np.max(fft_data) * 0.1,  # At least 10% of maximum
            distance=10,  # Minimum separation
        )

        # Analyze potential fundamentals
        for peak_idx in peaks:
            fundamental_freq = freqs[peak_idx]
            fundamental_magnitude = fft_data[peak_idx]

            if 20 <= fundamental_freq <= 2000:  # Reasonable fundamental range
                harmonic_info = self.analyze_harmonics_from_fundamental(
                    fft_data, freqs, fundamental_freq, fundamental_magnitude
                )

                if harmonic_info["strength"] > 0.3:  # Significant harmonic content
                    harmonics_found.append(harmonic_info)

        # Sort by harmonic strength
        harmonics_found.sort(key=lambda x: x["strength"], reverse=True)

        # Identify instruments
        instrument_matches = self.identify_instruments(harmonics_found[:3])  # Top 3 harmonic series

        return {
            "harmonic_series": harmonics_found,
            "instrument_matches": instrument_matches,
            "dominant_fundamental": harmonics_found[0]["fundamental"] if harmonics_found else 0,
        }

    def analyze_harmonics_from_fundamental(
        self,
        fft_data: np.ndarray,
        freqs: np.ndarray,
        fundamental_freq: float,
        fundamental_magnitude: float,
    ) -> Dict[str, Any]:
        """Analyze harmonic series from a given fundamental frequency"""
        harmonics = []
        max_harmonic = 16

        for n in range(1, max_harmonic + 1):
            harmonic_freq = fundamental_freq * n

            if harmonic_freq > self.nyquist:
                break

            # Find nearest frequency bin
            freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
            actual_freq = freqs[freq_idx]

            # Check if frequency is close enough to expected harmonic
            freq_tolerance = fundamental_freq * 0.05  # 5% tolerance
            if abs(actual_freq - harmonic_freq) <= freq_tolerance:
                magnitude = fft_data[freq_idx]
                relative_strength = magnitude / (fundamental_magnitude + 1e-10)

                harmonics.append(
                    {
                        "number": n,
                        "expected_freq": harmonic_freq,
                        "actual_freq": actual_freq,
                        "magnitude": magnitude,
                        "relative_strength": relative_strength,
                    }
                )

        # Calculate harmonic strength score
        harmonic_strength = 0.0
        if len(harmonics) >= 2:
            # Weight by harmonic number (lower harmonics more important)
            for harmonic in harmonics[:8]:  # First 8 harmonics
                weight = 1.0 / harmonic["number"]
                harmonic_strength += harmonic["relative_strength"] * weight

            harmonic_strength /= sum(1.0 / n for n in range(1, min(9, len(harmonics) + 1)))

        return {
            "fundamental": fundamental_freq,
            "harmonics": harmonics,
            "strength": harmonic_strength,
            "harmonic_count": len(harmonics),
        }

    def identify_instruments(self, harmonic_series_list: List[Dict]) -> List[Dict[str, Any]]:
        """Identify instruments based on harmonic content"""
        instrument_scores = {}

        for series in harmonic_series_list:
            fundamental = series["fundamental"]
            harmonics = series["harmonics"]

            # Test against each instrument profile
            for instrument, profile in self.instrument_profiles.items():
                score = self.calculate_instrument_match_score(fundamental, harmonics, profile)

                if instrument not in instrument_scores:
                    instrument_scores[instrument] = []
                instrument_scores[instrument].append(score)

        # Calculate average scores and sort
        final_scores = []
        for instrument, scores in instrument_scores.items():
            avg_score = np.mean(scores) if scores else 0.0
            if avg_score > 0.3:  # Minimum confidence threshold
                final_scores.append(
                    {"instrument": instrument, "confidence": avg_score, "match_count": len(scores)}
                )

        final_scores.sort(key=lambda x: x["confidence"], reverse=True)
        return final_scores

    def calculate_instrument_match_score(
        self, fundamental: float, harmonics: List[Dict], profile: Dict
    ) -> float:
        """Calculate how well harmonics match an instrument profile"""
        score = 0.0

        # Check if fundamental is in expected range
        fund_min, fund_max = profile["fundamental_range"]
        if not (fund_min <= fundamental <= fund_max):
            return 0.0

        # Check harmonic pattern matching
        expected_pattern = profile["harmonic_pattern"]

        if len(harmonics) >= 2:
            pattern_score = 0.0
            pattern_matches = 0

            for i, expected_strength in enumerate(expected_pattern):
                harmonic_num = i + 1

                # Find corresponding harmonic
                actual_harmonic = None
                for h in harmonics:
                    if h["number"] == harmonic_num:
                        actual_harmonic = h
                        break

                if actual_harmonic:
                    # Compare relative strengths
                    actual_strength = actual_harmonic["relative_strength"]
                    difference = abs(actual_strength - expected_strength)
                    match_score = max(0.0, 1.0 - difference)
                    pattern_score += match_score
                    pattern_matches += 1

            if pattern_matches > 0:
                score = pattern_score / pattern_matches

        return min(1.0, score)


class PhaseCoherenceAnalyzer:
    """Phase coherence analysis for stereo imaging (placeholder for future stereo support)"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.correlation_history = deque(maxlen=30)

    def analyze_phase_coherence(
        self, left_fft: np.ndarray, right_fft: np.ndarray
    ) -> Dict[str, float]:
        """Analyze phase relationships between stereo channels"""
        # For now, return placeholder values since we're using mono input
        # This will be implemented when stereo input is added

        return {
            "phase_correlation": 1.0,  # Perfect correlation for mono
            "stereo_width": 0.0,  # No width for mono
            "mono_compatibility": 1.0,  # Perfect mono compatibility
            "center_energy": 1.0,  # All energy in center for mono
        }


class TransientAnalyzer:
    """Transient analysis for attack detection and dynamics"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
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


class RoomModeAnalyzer:
    """Room acoustics analysis for studio applications"""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
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


class CepstralAnalyzer:
    """Advanced pitch detection using cepstrum, autocorrelation, and YIN algorithm"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.pitch_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        # Pitch detection parameters
        self.min_pitch = 50  # Hz (roughly G1)
        self.max_pitch = 2000  # Hz (roughly B6)
        self.min_period = int(sample_rate / self.max_pitch)
        self.max_period = int(sample_rate / self.min_pitch)
        
        # YIN algorithm parameters
        self.yin_threshold = 0.15
        self.yin_window_size = 1024  # Reduced for lower latency
        
        # Confidence thresholds
        self.high_confidence = 0.8
        self.medium_confidence = 0.5
        
    def compute_cepstrum(self, signal: np.ndarray) -> np.ndarray:
        """Compute real cepstrum for pitch detection"""
        # Apply window to reduce spectral leakage
        windowed = signal * np.hanning(len(signal))
        
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
        """YIN algorithm for robust pitch detection"""
        n = len(signal)
        if n < self.yin_window_size:
            return 0.0, 0.0
            
        # Use a window of the signal
        signal = signal[:self.yin_window_size]
        n = self.yin_window_size
        
        # Step 1: Calculate difference function (vectorized for speed)
        diff_function = np.zeros(n // 2)
        # Vectorized computation is much faster than nested loops
        for tau in range(1, min(self.max_period, n // 2)):
            # Compute squared differences for all j at once
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
                # Check if this is a local minimum
                if tau + 1 < n // 2 and cumulative_mean[tau] < cumulative_mean[tau - 1] and cumulative_mean[tau] < cumulative_mean[tau + 1]:
                    break
            tau += 1
        
        # Step 4: Refine using parabolic interpolation if valid minimum found
        if tau < min(self.max_period, n // 2 - 1):
            # Parabolic interpolation for more accurate period
            if tau > 0 and tau < n // 2 - 1:
                x0 = cumulative_mean[tau - 1]
                x1 = cumulative_mean[tau]
                x2 = cumulative_mean[tau + 1]
                
                a = (x0 - 2 * x1 + x2) / 2
                b = (x2 - x0) / 2
                
                if a != 0:
                    x_offset = -b / (2 * a)
                    tau = tau + x_offset
            
            pitch = self.sample_rate / tau
            confidence = 1.0 - cumulative_mean[int(tau)]
            
            return pitch, confidence
        
        return 0.0, 0.0
    
    def combine_pitch_estimates(self, signal: np.ndarray, cepstrum: np.ndarray) -> Dict[str, Any]:
        """Combine multiple pitch detection methods with confidence weighting"""
        # Method 1: Cepstral analysis
        cepstral_pitch, cepstral_conf = self.detect_pitch_cepstral(cepstrum)
        
        # Method 2: Autocorrelation
        autocorr_pitch, autocorr_conf = self.detect_pitch_autocorrelation(signal)
        
        # Method 3: YIN algorithm
        yin_pitch, yin_conf = self.yin_pitch_detection(signal)
        
        # Collect valid estimates
        estimates = []
        if cepstral_conf > 0.2 and self.min_pitch <= cepstral_pitch <= self.max_pitch:
            estimates.append((cepstral_pitch, cepstral_conf * 0.8))  # Slightly lower weight for cepstral
            
        if autocorr_conf > 0.3 and self.min_pitch <= autocorr_pitch <= self.max_pitch:
            estimates.append((autocorr_pitch, autocorr_conf * 0.9))
            
        if yin_conf > 0.4 and self.min_pitch <= yin_pitch <= self.max_pitch:
            estimates.append((yin_pitch, yin_conf * 1.0))  # YIN typically most reliable
        
        # Combine estimates
        if not estimates:
            return {
                'pitch': 0.0,
                'confidence': 0.0,
                'note': '',
                'cents_offset': 0,
                'methods': {
                    'cepstral': (cepstral_pitch, cepstral_conf),
                    'autocorr': (autocorr_pitch, autocorr_conf),
                    'yin': (yin_pitch, yin_conf)
                }
            }
        
        # Weighted average of estimates
        total_weight = sum(conf for _, conf in estimates)
        weighted_pitch = sum(pitch * conf for pitch, conf in estimates) / total_weight
        combined_confidence = total_weight / len(estimates)
        
        # Convert to musical note
        note_info = self.pitch_to_note(weighted_pitch)
        
        # Update history
        self.pitch_history.append(weighted_pitch)
        self.confidence_history.append(combined_confidence)
        
        return {
            'pitch': weighted_pitch,
            'confidence': combined_confidence,
            'note': note_info['note'],
            'cents_offset': note_info['cents_offset'],
            'octave': note_info['octave'],
            'methods': {
                'cepstral': (cepstral_pitch, cepstral_conf),
                'autocorr': (autocorr_pitch, autocorr_conf),
                'yin': (yin_pitch, yin_conf)
            },
            'stability': self.calculate_pitch_stability()
        }
    
    def pitch_to_note(self, pitch: float) -> Dict[str, Any]:
        """Convert frequency to musical note with cents offset"""
        if pitch <= 0:
            return {'note': '', 'cents_offset': 0, 'octave': 0}
            
        # A4 = 440 Hz standard
        A4 = 440.0
        C0 = A4 * (2 ** (-4.75))  # C0 frequency
        
        # Calculate semitones from C0
        semitones = 12 * np.log2(pitch / C0)
        rounded_semitones = int(round(semitones))
        cents_offset = int((semitones - rounded_semitones) * 100)
        
        # Get note name
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = rounded_semitones % 12
        octave = rounded_semitones // 12
        
        return {
            'note': note_names[note_index],
            'cents_offset': cents_offset,
            'octave': octave
        }
    
    def calculate_pitch_stability(self) -> float:
        """Calculate pitch stability over recent history"""
        if len(self.pitch_history) < 10:
            return 0.0
            
        recent_pitches = list(self.pitch_history)[-20:]
        recent_confidences = list(self.confidence_history)[-20:]
        
        # Filter out low confidence estimates
        valid_pitches = [p for p, c in zip(recent_pitches, recent_confidences) if c > 0.5 and p > 0]
        
        if len(valid_pitches) < 5:
            return 0.0
            
        # Calculate stability as inverse of pitch variance
        pitch_std = np.std(valid_pitches)
        mean_pitch = np.mean(valid_pitches)
        
        if mean_pitch > 0:
            relative_std = pitch_std / mean_pitch
            stability = max(0.0, 1.0 - relative_std * 10)  # Scale for display
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
            
        # Compute cepstrum once for efficiency
        cepstrum = self.compute_cepstrum(signal)
        
        # Combine all methods
        return self.combine_pitch_estimates(signal, cepstrum)


class ChromagramAnalyzer:
    """Real-time chromagram and key detection for musical analysis"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.chroma_bins = 12
        
        # Krumhansl-Kessler key profiles for major and minor keys
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Note names for display
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # History for temporal smoothing
        self.chroma_history = deque(maxlen=30)  # 0.5 seconds at 60 FPS
        self.key_history = deque(maxlen=60)  # 1 second for stability
        
    def compute_chromagram(self, fft_data: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Extract 12-bin chromagram from FFT data"""
        chroma = np.zeros(12)
        A4 = 440.0
        
        # Only process frequencies above 80Hz to avoid octave errors
        for i, freq in enumerate(freqs):
            if freq > 80 and freq < 8000:  # Focus on musical range
                # Map frequency to MIDI pitch
                midi_pitch = 69 + 12 * np.log2(freq / A4)
                
                # Get chroma bin (0-11)
                chroma_bin = int(round(midi_pitch) % 12)
                
                # Weight by magnitude
                chroma[chroma_bin] += fft_data[i]
        
        # Normalize to sum to 1
        chroma_sum = np.sum(chroma)
        if chroma_sum > 0:
            chroma = chroma / chroma_sum
            
        # Add to history for smoothing
        self.chroma_history.append(chroma)
        
        # Return smoothed chromagram
        if len(self.chroma_history) > 5:
            return np.mean(self.chroma_history, axis=0)
        else:
            return chroma
    
    def detect_key(self, chroma: np.ndarray) -> Tuple[str, float]:
        """Detect musical key using correlation with Krumhansl-Kessler profiles"""
        best_key = None
        best_correlation = -1
        best_is_major = True
        
        # Test all 24 possible keys (12 major + 12 minor)
        for i in range(12):
            # Rotate profiles to test each key
            major_profile_rotated = np.roll(self.major_profile, i)
            minor_profile_rotated = np.roll(self.minor_profile, i)
            
            # Calculate correlation coefficients
            if np.std(chroma) > 0:  # Avoid division by zero
                major_corr = np.corrcoef(chroma, major_profile_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma, minor_profile_rotated)[0, 1]
            else:
                major_corr = minor_corr = 0
            
            # Check if this is the best correlation so far
            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = f"{self.note_names[i]} Major"
                best_is_major = True
                
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = f"{self.note_names[i]} Minor"
                best_is_major = False
        
        # Add to history
        if best_key:
            self.key_history.append((best_key, best_correlation))
        
        return best_key, best_correlation
    
    def get_key_stability(self) -> float:
        """Calculate how stable the key detection has been"""
        if len(self.key_history) < 10:
            return 0.0
            
        # Count occurrences of each key
        key_counts = {}
        for key, _ in self.key_history:
            key_counts[key] = key_counts.get(key, 0) + 1
            
        # Find most common key
        if key_counts:
            max_count = max(key_counts.values())
            stability = max_count / len(self.key_history)
            return stability
        return 0.0
    
    def get_most_likely_key(self) -> Tuple[str, float]:
        """Get the most likely key based on recent history"""
        if not self.key_history:
            return "Unknown", 0.0
            
        # Weight recent detections more heavily
        weighted_keys = {}
        for i, (key, correlation) in enumerate(self.key_history):
            weight = (i + 1) / len(self.key_history)  # Linear weighting
            if key not in weighted_keys:
                weighted_keys[key] = 0
            weighted_keys[key] += correlation * weight
            
        # Find key with highest weighted score
        best_key = max(weighted_keys, key=weighted_keys.get)
        confidence = weighted_keys[best_key] / sum(weighted_keys.values())
        
        return best_key, confidence


# Enhanced drum detection classes (imported from v4 functionality)
class EnhancedKickDetector:
    """Professional kick/bass drum detection with multi-band analysis and value persistence"""

    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
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

    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
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

    def __init__(self, sample_rate: int = SAMPLE_RATE):
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

    def __init__(self, sample_rate: int = SAMPLE_RATE, sensitivity: float = 1.0):
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


class PipeWireMonitorCapture:
    """Reuse the capture class from output monitor"""

    def __init__(
        self, source_name: str, sample_rate: int = SAMPLE_RATE, chunk_size: int = CHUNK_SIZE
    ):
        self.source_name = source_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=5)
        self.capture_process = None
        self.capture_thread = None
        self.running = False

        # Noise gating
        self.noise_floor = 0.001
        self.silence_samples = 0
        self.silence_threshold = sample_rate // 4
        self.background_level = 0.0
        self.background_alpha = 0.001

    def list_monitor_sources(self):
        """List available monitor sources"""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"], capture_output=True, text=True
            )
            sources = []

            print("\n🔊 AVAILABLE OUTPUT MONITOR SOURCES:")
            print("=" * 70)

            for line in result.stdout.strip().split("\n"):
                if "monitor" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        source_id = parts[0]
                        source_name = parts[1]

                        # Categorize sources
                        quality_indicators = []
                        name_lower = source_name.lower()

                        if "focusrite" in name_lower or "scarlett" in name_lower:
                            quality_indicators.append("🎧 FOCUSRITE-OUTPUT")
                        elif "gsx" in name_lower or "sennheiser" in name_lower:
                            quality_indicators.append("🎮 GSX-OUTPUT")
                        elif "hdmi" in name_lower:
                            quality_indicators.append("📺 HDMI-OUTPUT")
                        else:
                            quality_indicators.append("🔊 SYSTEM-OUTPUT")

                        # Check if currently active
                        if len(parts) >= 4:
                            state = parts[3] if len(parts) > 3 else "UNKNOWN"
                            if state == "RUNNING":
                                quality_indicators.append("✅ ACTIVE")
                            elif state == "IDLE":
                                quality_indicators.append("💤 IDLE")
                            else:
                                quality_indicators.append("⚪ SUSPENDED")

                        indicators_str = " ".join(quality_indicators)

                        print(f"ID {source_id:3s}: {source_name}")
                        print(f"       {indicators_str}")
                        print()

                        sources.append((source_id, source_name))

            print("=" * 70)
            return sources

        except Exception as e:
            print(f"❌ Error listing sources: {e}")
            return []

    def select_monitor_source(self):
        """Interactive monitor source selection"""
        sources = self.list_monitor_sources()

        if not sources:
            print("❌ No monitor sources found!")
            return None

        # Auto-select Focusrite if available
        focusrite_sources = [
            s for s in sources if "focusrite" in s[1].lower() or "scarlett" in s[1].lower()
        ]

        if len(focusrite_sources) == 1:
            selected = focusrite_sources[0]
            print(f"🎯 Auto-selected Focusrite: {selected[1]}")
            return selected[1]

        # Interactive selection
        print("📋 SELECT MONITOR SOURCE:")
        print("💡 Choose the output device you want to analyze")

        if focusrite_sources:
            print(f"🌟 RECOMMENDED: ID {focusrite_sources[0][0]} (Focusrite)")

        print(f"\nEnter source ID or press Enter for auto-select:")

        try:
            user_input = input("Source ID: ").strip()

            if user_input == "":
                if focusrite_sources:
                    selected = focusrite_sources[0]
                    print(f"🎯 Auto-selected: {selected[1]}")
                    return selected[1]
                else:
                    selected = sources[0]
                    print(f"🎯 Using first available: {selected[1]}")
                    return selected[1]
            else:
                source_id = user_input
                for sid, sname in sources:
                    if sid == source_id:
                        print(f"✅ Selected: {sname}")
                        return sname

                print(f"❌ Invalid source ID: {source_id}")
                return None

        except KeyboardInterrupt:
            print("❌ Selection cancelled")
            return None

    def start_capture(self):
        """Start audio capture from monitor source"""
        if not self.source_name:
            self.source_name = self.select_monitor_source()

        if not self.source_name:
            print("❌ No monitor source selected")
            return False

        print(f"\n🎵 STARTING PROFESSIONAL ANALYSIS V5:")
        print(f"   Source: {self.source_name}")
        print(f"   Sample Rate: {self.sample_rate}Hz")
        print(f"   Chunk Size: {self.chunk_size} samples")

        # Start parec process
        try:
            self.capture_process = subprocess.Popen(
                [
                    "parec",
                    "--device=" + self.source_name,
                    "--format=float32le",
                    f"--rate={self.sample_rate}",
                    "--channels=1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            print("✅ Professional analysis v5 started!")
            return True

        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            return False

    def _capture_loop(self):
        """Audio capture loop"""
        bytes_per_sample = 4  # float32
        chunk_bytes = self.chunk_size * bytes_per_sample

        while self.running and self.capture_process:
            try:
                # Read audio data
                data = self.capture_process.stdout.read(chunk_bytes)
                if not data:
                    break

                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)

                if len(audio_data) == self.chunk_size:
                    # Apply noise gating
                    rms_level = np.sqrt(np.mean(audio_data**2))

                    # Update background noise estimate
                    if rms_level < self.noise_floor * 2:
                        self.background_level = (
                            1 - self.background_alpha
                        ) * self.background_level + self.background_alpha * rms_level

                    # Noise gate
                    if rms_level < max(self.noise_floor, self.background_level * 3):
                        self.silence_samples += self.chunk_size
                        if self.silence_samples > self.silence_threshold:
                            # Send silence
                            audio_data = np.zeros_like(audio_data)
                    else:
                        self.silence_samples = 0

                    # Put in queue
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        # Drop oldest sample
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(audio_data)
                        except:
                            pass

            except Exception as e:
                print(f"❌ Capture error: {e}")
                break

    def get_audio_data(self):
        """Get latest audio data"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_capture(self):
        """Stop audio capture"""
        self.running = False

        if self.capture_process:
            try:
                self.capture_process.terminate()
                self.capture_process.wait(timeout=2)
            except:
                try:
                    self.capture_process.kill()
                except:
                    pass
            self.capture_process = None

        if self.capture_thread:
            self.capture_thread.join(timeout=1)


class ContentTypeDetector:
    """Detects content type for adaptive frequency allocation"""
    
    def __init__(self):
        self.history_size = 30  # 30 frames ~0.5 seconds
        self.voice_history = []
        self.energy_history = []
        self.spectral_history = []
        
    def analyze_content(self, voice_info: Dict, band_values: np.ndarray, freq_starts: np.ndarray, freq_ends: np.ndarray) -> str:
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


class ProfessionalLiveAudioAnalyzer:
    """Professional Live Audio Analyzer v5 with enhanced low-end detail and studio features"""

    def __init__(self, width=2000, height=1080, bars=BARS_DEFAULT, source_name=None):
        self.width = width
        self.height = height
        self.bars = bars

        # Initialize core components
        self.capture = PipeWireMonitorCapture(source_name)
        self.drum_detector = EnhancedDrumDetector()
        self.voice_detector = IndustryVoiceDetector(SAMPLE_RATE)
        self.content_detector = ContentTypeDetector()

        # Initialize new v5 components
        self.multi_fft = MultiResolutionFFT(SAMPLE_RATE)
        self.professional_metering = ProfessionalMetering(SAMPLE_RATE)
        self.harmonic_analyzer = HarmonicAnalyzer(SAMPLE_RATE)
        self.phase_analyzer = PhaseCoherenceAnalyzer(SAMPLE_RATE)
        self.transient_analyzer = TransientAnalyzer(SAMPLE_RATE)
        self.room_analyzer = RoomModeAnalyzer(SAMPLE_RATE)
        self.cepstral_analyzer = CepstralAnalyzer(SAMPLE_RATE)  # OMEGA: Advanced pitch detection
        self.chromagram_analyzer = ChromagramAnalyzer(SAMPLE_RATE)  # OMEGA-1: Musical key detection
        
        # Input gain control for better signal levels
        self.input_gain = 4.0  # Default 12dB boost for typical music sources
        self.auto_gain_enabled = True
        self.gain_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.target_lufs = -16.0  # Target for good visualization
        
        # Compensation toggles for debugging
        self.freq_compensation_enabled = True  # Toggle with 'Q'
        self.psychoacoustic_enabled = True     # Toggle with 'W'
        self.normalization_enabled = True      # Toggle with 'E'
        self.smoothing_enabled = True          # Toggle with 'R'

        # Audio analysis with enhanced resolution
        self.ring_buffer = np.zeros(FFT_SIZE_BASE * 4, dtype=np.float32)  # Larger buffer
        self.buffer_pos = 0

        # Enhanced frequency mapping with 768 bars
        self.freqs = np.fft.rfftfreq(FFT_SIZE_BASE, 1 / SAMPLE_RATE)
        
        # Adaptive content detection
        self.adaptive_allocation_enabled = False  # Disabled for perceptual mapping
        self.current_content_type = 'instrumental'
        self.current_allocation = 0.75
        self.band_indices = self._create_enhanced_band_mapping()

        # Pygame setup with professional UI
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Professional Audio Analyzer v4 OMEGA-1 - Musical Key Detection")
        self.clock = pygame.time.Clock()

        # Dynamic font sizing system
        self.base_width = 2000  # Reference width for font scaling
        self.update_fonts(width)

        # Enhanced visualization state - NOW using updated self.bars from band mapping
        self.bar_heights = np.zeros(self.bars, dtype=np.float32)
        self.colors = self._generate_professional_colors()

        # Midrange boost for 1k-6k enhancement - Initialize before frequency compensation
        self.midrange_boost_enabled = True
        self.midrange_boost_factor = 2.0  # 2x boost for 1k-6k range

        # Pre-calculate frequency compensation gains
        self.freq_compensation_gains = self._precalculate_freq_compensation()
        
        # Pre-calculate band mapping for performance
        self._setup_optimized_band_mapping()
        
        # Pre-calculate frequency ranges for smoothing
        self._setup_frequency_ranges()
        
        # Ensure bar_heights matches the actual bar count after all initialization
        if len(self.bar_heights) != self.bars:
            self.bar_heights = np.zeros(self.bars, dtype=np.float32)
        
        # Enhanced bass detail panel (64 independent bars for 20-200Hz)
        self.bass_detail_bars = 64
        self.bass_bar_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_timestamps = np.zeros(self.bass_detail_bars, dtype=np.float64)
        self._setup_bass_frequency_mapping()
        
        # Pre-calculate frequency to bar mapping for fast lookups
        self._build_frequency_lookup()

        # Performance tracking
        self.last_process_time = 0
        self.peak_latency = 0.0  # Track highest latency
        self.fps_counter = deque(maxlen=30)
        self.frame_counter = 0  # For skipping expensive operations
        
        # Detailed performance profiling
        self.performance_profiling = False
        self.profile_times = {
            'audio_capture': 0,
            'fft_processing': 0,
            'band_mapping': 0,
            'analysis': 0,
            'smoothing': 0,
            'bass_detail': 0,
            'total': 0
        }
        
        # Dynamic quality mode for performance optimization
        self.quality_mode = "quality"  # "quality" or "performance"
        self.dynamic_quality_enabled = True
        self.performance_threshold = 30.0  # ms - switch to performance mode above this
        self.quality_threshold = 25.0  # ms - switch back to quality mode below this
        self.quality_mode_hold_time = 10.0  # seconds - minimum time before switching modes
        self.last_quality_switch_time = 0.0  # timestamp of last mode switch
        
        # Async bass processing for reduced latency
        self.bass_processing_queue = Queue(maxsize=2)
        self.bass_thread_running = True
        self.latest_bass_result = None
        self.bass_processing_thread = threading.Thread(target=self._bass_processor_worker)
        self.bass_processing_thread.daemon = True
        self.bass_processing_thread.start()
        
        # VU Meter initialization
        self.show_vu_meters = True  # Toggle with 'J'
        self.vu_integration_time = 0.3  # 300ms VU standard
        # Adjust reference level for typical music with consideration for input gain
        # Standard: 0 VU = -20 dBFS, but we need to compensate for any input gain
        self.vu_reference_level = 0.1  # -20 dBFS = 0 VU (will be adjusted for gain)
        
        # VU meter buffers
        self.vu_buffer_samples = int(SAMPLE_RATE * self.vu_integration_time)
        self.vu_left_buffer = deque(maxlen=self.vu_buffer_samples)
        self.vu_right_buffer = deque(maxlen=self.vu_buffer_samples)
        
        # Initialize with silence
        for _ in range(self.vu_buffer_samples):
            self.vu_left_buffer.append(0.0)
            self.vu_right_buffer.append(0.0)
            
        # VU meter display values
        self.vu_left_db = -20.0
        self.vu_right_db = -20.0
        self.vu_left_display = -20.0
        self.vu_right_display = -20.0
        self.vu_damping = 0.85  # Needle damping
        
        # Peak hold for VU meters
        self.vu_left_peak_db = -60.0
        self.vu_right_peak_db = -60.0
        self.vu_peak_hold_time = 2.0  # seconds
        self.vu_left_peak_time = 0.0
        self.vu_right_peak_time = 0.0
        
        # Initialize analysis results for frame skipping
        self.drum_info = {
            'kick': {'kick_detected': False, 'magnitude': 0.0},
            'snare': {'snare_detected': False, 'magnitude': 0.0}
        }
        self.voice_info = {
            'has_voice': False,
            'voice_confidence': 0.0,
            'is_singing': False,
            'voice_type': 'none'
        }
        self.lufs_info = {
            'momentary': -100.0,
            'short_term': -100.0,
            'integrated': -100.0,
            'range': 0.0
        }
        self.true_peak = -100.0
        self.harmonic_info = {
            'harmonic_series': [],
            'instrument_matches': [],
            'dominant_fundamental': 0
        }
        self.transient_info = {'transients_detected': 0, 'attack_time': 0.0, 'punch_factor': 0.0}
        self.room_modes = []
        self.pitch_info = {  # OMEGA: Initialize pitch detection results
            'pitch': 0.0,
            'confidence': 0.0,
            'note': '',
            'cents_offset': 0,
            'octave': 0,
            'stability': 0.0,
            'methods': {}
        }
        self.chromagram_info = {  # OMEGA-1: Initialize chromagram results
            'chromagram': np.zeros(12),
            'key': 'C',
            'is_major': True,
            'confidence': 0.0,
            'stability': 0.0,
            'alternative_keys': []
        }

        # Enhanced visual effects
        self.kick_flash_time = 0
        self.snare_flash_time = 0
        self.voice_flash_time = 0
        self.singing_flash_time = 0

        # Professional studio enhancements - use self.bars instead of bars parameter
        self.peak_hold_bars = np.zeros(self.bars, dtype=np.float32)
        self.peak_decay_rate = 0.85  # More aggressive decay - drops to ~0.4% after 20 frames
        self.peak_hold_time = 3.0  # Hold peaks for 3 seconds (reduced from 5)
        # Initialize timestamps to 0 (will trigger immediate decay for any non-zero peaks)
        self.peak_timestamps = np.zeros(self.bars, dtype=np.float64)  # Use float64 for timestamps
        
        # Peak hold for bass zoom window
        self.bass_peak_hold = {}  # Will store peak values for bass frequencies
        self.bass_peak_timestamps = {}  # Timestamps for bass peaks

        # Sub-bass monitoring (0-60Hz)
        self.sub_bass_energy = 0.0
        self.sub_bass_history = deque(maxlen=100)
        self.sub_bass_warning_active = False
        self.sub_bass_warning_time = 0
        
        # Ultra sub-bass monitoring (0-20Hz)
        self.ultra_sub_bass_energy = 0.0
        self.ultra_sub_bass_history = deque(maxlen=100)

        # Room mode detection
        self.sustained_peaks = {}  # freq -> (magnitude, start_time, x_pos)
        self.room_mode_candidates = []

        # A/B comparison system
        self.reference_spectrum = None
        self.comparison_mode = False
        self.reference_stored = False

        # Technical overlay state
        self.show_technical_overlay = False
        self.last_technical_analysis = {}

        # Professional grid and separators
        self.show_frequency_grid = True
        self.show_band_separators = True
        self.show_note_labels = True

        # UI state with new features
        self.show_professional_meters = True
        self.show_harmonic_analysis = True
        self.show_room_analysis = False
        self.show_bass_zoom = True
        self.show_advanced_info = False
        self.show_voice_info = True
        self.show_formants = False
        self.show_pitch_detection = False  # OMEGA: Disabled by default for performance
        self.show_chromagram = False  # OMEGA-1: Disabled by default for performance

        # Professional meters layout - increased for better spacing
        self.meter_panel_width = 280  # Increased from 200 to prevent title cutoff
        self.bass_zoom_height = 200  # Increased for 64-bar display
        self.left_border_width = 80  # Left border for SUB level indicator and spacing - increased for more separation
        self.vu_meter_width = 150  # Width for VU meters on the right

        print(f"\n🎵 Professional Audio Analyzer v4 OMEGA-1 - Musical Key Detection")
        print(f"Resolution: {width}x{height}, Bars: {bars}")
        print("Studio-Grade Features:")
        print("  🎚️ Perceptual frequency mapping using Bark scale")
        print("  👂 Natural frequency representation matching human hearing")
        print("  📊 Professional metering (LUFS, K-weighting, True Peak)")
        print("  🎼 Advanced harmonic analysis and instrument ID")
        print("  🔊 Phase coherence and transient analysis")
        print("  🏠 Room mode detection")
        print("  🎤 Industry voice detection + beat analysis")
        print("  🎵 OMEGA: Advanced pitch detection with cepstral analysis")
        print("Enhanced Controls:")
        print("  M: Toggle professional meters")
        print("  H: Toggle harmonic analysis")
        print("  R: Toggle room analysis")
        print("  Z: Toggle bass zoom window")
        print("  G: Toggle analysis grid")
        print("  B: Toggle band separators")
        print("  T: Toggle technical overlay")
        print("  P: Toggle pitch detection (OMEGA)")
        print("  C: Store reference / Toggle A/B comparison")
        print("  X: Clear reference spectrum")
        print("  ESC/S/V/F/A: Standard controls")
        print("Window Presets:")
        print("  1: 1400x900 (Compact)     2: 1800x1000 (Standard)")
        print("  3: 2200x1200 (Wide)       4: 2600x1400 (Ultra-Wide)")
        print("  5: 3000x1600 (Cinema)     6: 3400x1800 (Studio)")
        print("  7: 1920x1080 (Full HD)    8: 2560x1440 (2K)")
        print("  9: 3840x2160 (4K)         0: Toggle Fullscreen")

    def update_fonts(self, current_width: int):
        """Update fonts based on current window width for optimal readability"""
        # Scale fonts based on window size
        scale_factor = current_width / self.base_width
        scale_factor = max(0.6, min(1.4, scale_factor))  # Clamp between 60% and 140%

        # Professional font sizes with scaling
        self.font_large = pygame.font.Font(None, max(24, int(36 * scale_factor)))
        self.font_medium = pygame.font.Font(None, max(20, int(28 * scale_factor)))
        self.font_small = pygame.font.Font(None, max(18, int(24 * scale_factor)))
        self.font_tiny = pygame.font.Font(None, max(16, int(20 * scale_factor)))

        # Special fonts for critical information
        self.font_meters = pygame.font.Font(None, max(18, int(22 * scale_factor)))
        self.font_grid = pygame.font.Font(None, max(14, int(18 * scale_factor)))
        self.font_mono = pygame.font.Font(None, max(16, int(20 * scale_factor)))

        # Store current scale for other UI elements
        self.ui_scale = scale_factor

    def draw_organized_header(self, header_height: int):
        """Draw organized header layout to prevent text overlap"""
        # Create columns for better organization
        col1_x = 20
        col2_x = int(self.width * 0.35)
        col3_x = int(self.width * 0.65)

        row_height = int(25 * self.ui_scale)
        y_start = 15

        # Column 1: Basic Info
        y = y_start
        title = self.font_large.render("Professional Audio Analyzer v4 OMEGA", True, (255, 255, 255))
        self.screen.blit(title, (col1_x, y))
        y += int(40 * self.ui_scale)

        # Studio subtitle - check if it fits in one line
        subtitle_text = (
            "Musical Perceptual Mapping • Professional Metering • Harmonic Analysis • Room Acoustics"
        )
        subtitle = self.font_small.render(subtitle_text, True, (180, 200, 220))
        if subtitle.get_width() > self.width - col1_x - 100:
            # Split into two lines if too long
            subtitle1 = self.font_small.render(
                "Musical Perceptual Mapping • Professional Metering", True, (180, 200, 220)
            )
            subtitle2 = self.font_small.render(
                "Harmonic Analysis • Room Acoustics", True, (180, 200, 220)
            )
            self.screen.blit(subtitle1, (col1_x, y))
            y += row_height
            self.screen.blit(subtitle2, (col1_x, y))
        else:
            self.screen.blit(subtitle, (col1_x, y))
        y += int(35 * self.ui_scale)

        # Feature highlights
        features1 = self.font_tiny.render(
            "🎯 Peak Hold • ⚠️ Sub-Bass Monitor • 📊 Band Separators", True, (150, 170, 190)
        )
        self.screen.blit(features1, (col1_x, y))
        y += row_height

        features2 = self.font_tiny.render(
            "🔬 Technical Overlay • ⚖️ A/B Comparison • 📏 Analysis Grid", True, (150, 170, 190)
        )
        self.screen.blit(features2, (col1_x, y))
        y += row_height

        # Gain controls hint
        controls_hint = self.font_tiny.render(
            "🎛️ Gain: +/- keys • 0: Toggle Auto-gain • ESC: Exit", True, (120, 140, 160)
        )
        self.screen.blit(controls_hint, (col1_x, y))

        # Column 2: System Status
        if col2_x < self.width - 200:  # Only show if there's space
            y = y_start + int(60 * self.ui_scale)

            # Audio source info
            if hasattr(self, "capture") and self.capture:
                source_text = "🎤 Audio Source: Professional Monitor"
                source_surf = self.font_small.render(source_text, True, (120, 200, 120))
                self.screen.blit(source_surf, (col2_x, y))
                y += row_height

                # Technical specifications
                tech_info = f"📊 {SAMPLE_RATE}Hz • {self.bars} bars"
                tech_surf = self.font_tiny.render(tech_info, True, (140, 160, 180))
                self.screen.blit(tech_surf, (col2_x, y))
                y += row_height

                # Multi-FFT info
                fft_info = "Multi-FFT: 8192/4096/2048/1024"
                fft_surf = self.font_tiny.render(fft_info, True, (140, 160, 180))
                self.screen.blit(fft_surf, (col2_x, y))

        # Column 3: Performance & Active Features
        if col3_x < self.width - 150:  # Only show if there's space
            y = y_start + int(60 * self.ui_scale)

            # Performance metrics
            if hasattr(self, "last_process_time"):
                latency_ms = self.last_process_time * 1000
                # Update peak latency
                if latency_ms > self.peak_latency:
                    self.peak_latency = latency_ms
                    
                perf_color = (
                    (120, 200, 120)
                    if self.peak_latency < 10
                    else (200, 200, 120) if self.peak_latency < 20 else (200, 120, 120)
                )
                perf_text = f"⚡ Peak Latency: {self.peak_latency:.1f}ms"
                perf_surf = self.font_small.render(perf_text, True, perf_color)
                self.screen.blit(perf_surf, (col3_x, y))
                y += row_height

                # FPS info
                if hasattr(self, "fps_counter") and len(self.fps_counter) > 0:
                    avg_fps = sum(self.fps_counter) / len(self.fps_counter)
                    fps_color = (120, 200, 120) if avg_fps > 55 else (200, 200, 120)
                    fps_text = f"📈 FPS: {avg_fps:.1f}"
                    fps_surf = self.font_tiny.render(fps_text, True, fps_color)
                    self.screen.blit(fps_surf, (col3_x, y))
                    y += row_height
                
                # Quality mode indicator
                if hasattr(self, 'quality_mode'):
                    if self.quality_mode == "performance":
                        mode_color = (255, 150, 100)  # Orange for performance
                        mode_text = "🚀 Performance Mode"
                    else:
                        mode_color = (100, 255, 100)  # Green for quality
                        mode_text = "✨ Quality Mode"
                    mode_surf = self.font_tiny.render(mode_text, True, mode_color)
                    self.screen.blit(mode_surf, (col3_x, y))
                    y += row_height

            # Input gain display
            gain_db = 20 * np.log10(self.input_gain)
            gain_color = (120, 200, 120) if self.auto_gain_enabled else (200, 200, 120)
            gain_text = f"🎚️ Gain: +{gain_db:.1f}dB"
            if self.auto_gain_enabled:
                gain_text += " (Auto)"
            gain_surf = self.font_small.render(gain_text, True, gain_color)
            self.screen.blit(gain_surf, (col3_x, y))
            y += row_height

            # Active features status
            active_features = []
            if self.show_frequency_grid:
                active_features.append("Grid")
            if self.show_band_separators:
                active_features.append("Bands")
            if self.show_technical_overlay:
                active_features.append("Tech")
            if self.comparison_mode:
                active_features.append("A/B")
            if self.show_room_analysis:
                active_features.append("Room")

            if active_features:
                features_text = f"🔧 Active: {' • '.join(active_features[:3])}"  # Limit to 3
                if len(active_features) > 3:
                    features_text += f" (+{len(active_features) - 3})"
                features_surf = self.font_tiny.render(features_text, True, (100, 150, 200))
                self.screen.blit(features_surf, (col3_x, y))

    def draw_enhanced_frequency_scale(self):
        """Enhanced frequency scale with band-aligned tick marks"""
        # Calculate spectrum area bounds
        header_height = 280
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        
        # The spectrum extends from header to where we want the scale
        # We need space for: spectrum + scale (40px) + gap (10px) + bass zoom
        spectrum_bottom = self.height - bass_zoom_height - 50  # Leave 50px for scale + gap
        
        # Position scale right after the spectrum with a small gap
        scale_y = spectrum_bottom + 5

        # Key frequencies including band boundaries
        # Primary marks (with labels): band boundaries + key frequencies
        primary_frequencies = [20, 60, 120, 250, 500, 1000, 2000, 4000, 6000, 10000, 20000]
        # Secondary marks (ticks only): additional reference points
        secondary_frequencies = [30, 40, 80, 150, 300, 700, 1500, 3000, 5000, 8000, 15000]

        # Draw scale line
        spectrum_left = self.left_border_width
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_right = self.width - meter_width - vu_width - 10
        
        # Draw background for the scale area to ensure visibility
        scale_bg_rect = pygame.Rect(spectrum_left - 5, scale_y - 5, spectrum_right - spectrum_left + 10, 45)
        pygame.draw.rect(self.screen, (15, 20, 30), scale_bg_rect)
        
        # Draw a more visible scale line
        pygame.draw.line(
            self.screen, (120, 120, 140), (spectrum_left, scale_y), (spectrum_right, scale_y), 3
        )
        
        # Draw secondary tick marks first (smaller, no labels)
        for freq in secondary_frequencies:
            x_pos = int(self.freq_to_x_position(freq))
            if spectrum_left <= x_pos <= spectrum_right:
                tick_height = int(5 * self.ui_scale)
                pygame.draw.line(
                    self.screen, (140, 140, 150), (x_pos, scale_y), (x_pos, scale_y + tick_height), 2
                )
        
        # Draw primary marks with labels
        self.last_label_x = -100
        for freq in primary_frequencies:
            x_pos = int(self.freq_to_x_position(freq))

            # Skip if outside spectrum area
            if x_pos < spectrum_left or x_pos > spectrum_right:
                continue

            # Draw larger tick mark for primary frequencies
            tick_height = int(10 * self.ui_scale)
            # Band boundaries get thicker ticks
            if freq in [60, 250, 500, 2000, 6000]:
                pygame.draw.line(
                    self.screen, (200, 200, 220), (x_pos, scale_y - 3), (x_pos, scale_y + tick_height + 3), 3
                )
            else:
                pygame.draw.line(
                    self.screen, (170, 170, 180), (x_pos, scale_y), (x_pos, scale_y + tick_height), 2
                )

            # Skip label if too close to previous
            min_spacing = max(35, int(40 * self.ui_scale))
            if abs(x_pos - self.last_label_x) < min_spacing:
                continue

            # Format frequency text
            if freq >= 10000:
                freq_text = f"{freq/1000:.0f}k"
            elif freq >= 1000:
                freq_text = f"{freq/1000:.0f}k" if freq % 1000 == 0 else f"{freq/1000:.1f}k"
            else:
                freq_text = f"{freq}"

            # Draw label with background for clarity
            label = self.font_grid.render(freq_text, True, (220, 220, 230))
            label_rect = label.get_rect(centerx=x_pos, top=scale_y + tick_height + 3)

            # Dark background for readability
            bg_padding = int(2 * self.ui_scale)
            pygame.draw.rect(
                self.screen, (20, 25, 35), label_rect.inflate(bg_padding * 2, bg_padding)
            )
            self.screen.blit(label, label_rect)

            self.last_label_x = x_pos

    def draw_smart_grid_labels(self):
        """Grid with dB labels for mirrored spectrum (0dB at center)"""
        # Amplitude grid - show fewer labels if window is cramped
        if self.height < 900:
            db_levels = [0, -20, -40, -60]  # Minimal set for small windows
        else:
            db_levels = [0, -10, -20, -30, -40, -50, -60]  # Full set for large windows

        spectrum_left = self.left_border_width
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_right = self.width - meter_width - vu_width - 10
        spectrum_top = 280 + 10  # Header height + gap
        # Account for frequency scale and bass zoom
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50
        
        # Calculate center line position
        center_y = spectrum_top + (spectrum_bottom - spectrum_top) // 2
        spectrum_height = spectrum_bottom - spectrum_top
        
        # Draw center line (0dB) more prominently
        pygame.draw.line(
            self.screen, (60, 70, 90), (spectrum_left, center_y), (spectrum_right, center_y), 2
        )
        
        # Draw 0dB label at center
        label_text = "  0"
        label = self.font_grid.render(label_text, True, (150, 160, 170))
        label_rect = label.get_rect(right=spectrum_left - 5, centery=center_y)
        if label_rect.left >= 5:
            bg_padding = int(2 * self.ui_scale)
            pygame.draw.rect(
                self.screen, (15, 20, 30), label_rect.inflate(bg_padding * 2, bg_padding)
            )
            self.screen.blit(label, label_rect)

        # Draw other dB levels both above and below center
        for db in db_levels[1:]:  # Skip 0dB as we already drew it
            # Calculate distance from center (0 to 1)
            normalized_pos = abs(db) / 60.0
            
            # Upper position (above center line)
            upper_y = int(center_y - (spectrum_height / 2) * normalized_pos)
            # Lower position (below center line)
            lower_y = int(center_y + (spectrum_height / 2) * normalized_pos)
            
            # Draw grid lines
            if spectrum_top <= upper_y <= center_y:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), (spectrum_left, upper_y), (spectrum_right, upper_y), 1
                )
            if center_y <= lower_y <= spectrum_bottom:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), (spectrum_left, lower_y), (spectrum_right, lower_y), 1
                )
            
            # Draw labels
            label_text = f"{db:3d}"
            label = self.font_grid.render(label_text, True, (120, 120, 130))
            
            # Upper label
            if spectrum_top <= upper_y <= center_y:
                label_rect = label.get_rect(right=spectrum_left - 5, centery=upper_y)
                if label_rect.left >= 5:
                    bg_padding = int(2 * self.ui_scale)
                    pygame.draw.rect(
                        self.screen, (15, 20, 30), label_rect.inflate(bg_padding * 2, bg_padding)
                    )
                    self.screen.blit(label, label_rect)
            
            # Lower label
            if center_y <= lower_y <= spectrum_bottom:
                label_rect = label.get_rect(right=spectrum_left - 5, centery=lower_y)
                if label_rect.left >= 5:
                    bg_padding = int(2 * self.ui_scale)
                    pygame.draw.rect(
                        self.screen, (15, 20, 30), label_rect.inflate(bg_padding * 2, bg_padding)
                    )
                    self.screen.blit(label, label_rect)

    def _create_enhanced_band_mapping(self):
        """Create perceptual frequency band mapping using modified Bark scale for musical representation"""
        # Calculate FFT bin resolution
        fft_bin_width = self.freqs[1] - self.freqs[0]  # Hz per bin
        min_fft_freq = fft_bin_width  # First non-DC bin
        
        freq_min = max(20, min_fft_freq)
        freq_max = min(MAX_FREQ, self.freqs[-1])

        # Modified perceptual scale that balances Bark scale with musical needs
        # This hybrid approach ensures good coverage across all frequency ranges
        
        # Define key frequency ranges for music
        # Sub-bass: 20-60 Hz (fundamental frequencies of bass instruments)
        # Bass: 60-250 Hz (fundamental frequencies of most instruments)
        # Low-mid: 250-500 Hz (body of instruments)
        # Mid: 500-2000 Hz (critical for intelligibility and presence)
        # High-mid: 2000-6000 Hz (brilliance and clarity)
        # High: 6000-20000 Hz (air and sparkle)
        
        # Allocate bars based on musical importance and perceptual sensitivity
        total_bars = self.bars
        
        # Distribution that ensures good mid-range coverage
        sub_bass_bars = int(total_bars * 0.08)    # 8% for 20-60 Hz
        bass_bars = int(total_bars * 0.17)        # 17% for 60-250 Hz
        low_mid_bars = int(total_bars * 0.15)     # 15% for 250-500 Hz
        mid_bars = int(total_bars * 0.25)         # 25% for 500-2000 Hz (critical range)
        high_mid_bars = int(total_bars * 0.20)    # 20% for 2000-6000 Hz
        high_bars = total_bars - (sub_bass_bars + bass_bars + low_mid_bars + mid_bars + high_mid_bars)  # 15% for 6000-20000 Hz
        
        # Create band indices directly from available FFT bins
        band_indices = []
        freq_edges = [freq_min]  # Start with minimum frequency
        
        # Get indices of FFT bins within our frequency range
        valid_bin_indices = np.where((self.freqs >= freq_min) & (self.freqs <= freq_max))[0]
        num_valid_bins = len(valid_bin_indices)
        
        if num_valid_bins < total_bars:
            # If we have fewer FFT bins than bars, reduce bar count
            print(f"⚠️  Reducing bars from {total_bars} to {num_valid_bins} due to FFT resolution")
            self.bars = num_valid_bins
            total_bars = num_valid_bins
            
            # Recalculate distribution
            sub_bass_bars = max(1, int(total_bars * 0.08))
            bass_bars = max(1, int(total_bars * 0.17))
            low_mid_bars = max(1, int(total_bars * 0.15))
            mid_bars = max(1, int(total_bars * 0.25))
            high_mid_bars = max(1, int(total_bars * 0.20))
            high_bars = total_bars - (sub_bass_bars + bass_bars + low_mid_bars + mid_bars + high_mid_bars)
        
        # Map frequency ranges to FFT bins
        current_bin = 0
        ranges = [
            (sub_bass_bars, 20, 60),
            (bass_bars, 60, 250),
            (low_mid_bars, 250, 500),
            (mid_bars, 500, 2000),
            (high_mid_bars, 2000, 6000),
            (high_bars, 6000, 20000)
        ]
        
        for num_bars, range_start, range_end in ranges:
            # Find FFT bins in this frequency range
            range_bins = np.where((self.freqs >= range_start) & (self.freqs < range_end))[0]
            
            if len(range_bins) > 0:
                # Distribute available bins among bars for this range
                bins_per_bar = max(1, len(range_bins) // num_bars)
                
                for i in range(num_bars):
                    start_idx = i * bins_per_bar
                    end_idx = (i + 1) * bins_per_bar if i < num_bars - 1 else len(range_bins)
                    
                    if start_idx < len(range_bins):
                        bar_bin_indices = range_bins[start_idx:end_idx]
                        if len(bar_bin_indices) > 0:
                            band_indices.append(bar_bin_indices)
                            # Store frequency edges for display
                            freq_edges.append(self.freqs[bar_bin_indices[-1]])
        
        # Ensure we have exactly the right number of bars
        if len(band_indices) > total_bars:
            band_indices = band_indices[:total_bars]
            freq_edges = freq_edges[:total_bars + 1]
        
        # Update bars count to match actual band indices
        self.bars = len(band_indices)
        
        # Store frequency edges for use in freq_to_x_position
        self.freq_edges = np.array(freq_edges)

        # Debug: Print perceptual band analysis
        print(f"\n🎵 Musical perceptual frequency mapping created: {len(band_indices)} bars")
        print(f"Frequency range: {freq_min:.1f}Hz - {freq_max:.1f}Hz")
        print(f"FFT bin resolution: {fft_bin_width:.1f}Hz per bin")
        
        # Count actual bars per range
        actual_distribution = {}
        for i, indices in enumerate(band_indices):
            if len(indices) > 0:
                center_freq = self.freqs[indices[len(indices)//2]]
                if center_freq < 60:
                    key = "sub_bass"
                elif center_freq < 250:
                    key = "bass"
                elif center_freq < 500:
                    key = "low_mid"
                elif center_freq < 2000:
                    key = "mid"
                elif center_freq < 6000:
                    key = "high_mid"
                else:
                    key = "high"
                actual_distribution[key] = actual_distribution.get(key, 0) + 1
        
        print(f"\nFrequency distribution:")
        print(f"  Sub-bass (20-60Hz): {actual_distribution.get('sub_bass', 0)} bars")
        print(f"  Bass (60-250Hz): {actual_distribution.get('bass', 0)} bars")
        print(f"  Low-mid (250-500Hz): {actual_distribution.get('low_mid', 0)} bars")
        print(f"  Mid (500-2kHz): {actual_distribution.get('mid', 0)} bars")
        print(f"  High-mid (2k-6kHz): {actual_distribution.get('high_mid', 0)} bars")
        print(f"  High (6k-20kHz): {actual_distribution.get('high', 0)} bars")

        return band_indices

    def _generate_professional_colors(self):
        """Generate professional color gradient for enhanced resolution"""
        colors = []
        for i in range(self.bars):
            hue = i / self.bars

            # Smooth gradient using cosine interpolation for seamless transitions
            # Purple -> Red -> Orange -> Yellow -> Green -> Cyan -> Blue
            
            # Create smooth transitions using sine/cosine functions
            # This creates natural, continuous color transitions
            
            if hue < 0.167:  # Purple to Red (0 to 1/6)
                t = hue / 0.167
                r = int(150 + 105 * t)
                g = int(50 * (1 - t))
                b = int(200 * (1 - t))
            elif hue < 0.333:  # Red to Orange to Yellow (1/6 to 1/3)
                t = (hue - 0.167) / 0.166
                r = 255
                g = int(255 * t)
                b = 0
            elif hue < 0.5:  # Yellow to Green (1/3 to 1/2)
                t = (hue - 0.333) / 0.167
                r = int(255 * (1 - t))
                g = 255
                b = int(100 * t)
            elif hue < 0.667:  # Green to Cyan (1/2 to 2/3)
                t = (hue - 0.5) / 0.167
                r = 0
                g = int(255 - 55 * t)
                b = int(100 + 155 * t)
            elif hue < 0.833:  # Cyan to Blue (2/3 to 5/6)
                t = (hue - 0.667) / 0.166
                r = int(100 * t)
                g = int(200 * (1 - t))
                b = 255
            else:  # Blue to Light Blue/White (5/6 to 1)
                t = (hue - 0.833) / 0.167
                r = int(100 + 155 * t)
                g = int(200 * t)
                b = 255

            colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

        return colors

    def process_audio_chunk(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Process audio chunk with multi-resolution FFT"""
        chunk_len = len(audio_data)

        # Update ring buffer
        if self.buffer_pos + chunk_len <= len(self.ring_buffer):
            self.ring_buffer[self.buffer_pos : self.buffer_pos + chunk_len] = audio_data
        else:
            first_part = len(self.ring_buffer) - self.buffer_pos
            self.ring_buffer[self.buffer_pos :] = audio_data[:first_part]
            self.ring_buffer[: chunk_len - first_part] = audio_data[first_part:]

        self.buffer_pos = (self.buffer_pos + chunk_len) % len(self.ring_buffer)

        # Extract latest samples for analysis
        audio_buffer = np.zeros(FFT_SIZE_BASE, dtype=np.float32)
        if self.buffer_pos >= FFT_SIZE_BASE:
            audio_buffer[:] = self.ring_buffer[self.buffer_pos - FFT_SIZE_BASE : self.buffer_pos]
        elif self.buffer_pos > 0:
            audio_buffer[-self.buffer_pos :] = self.ring_buffer[: self.buffer_pos]
            audio_buffer[: -self.buffer_pos] = self.ring_buffer[
                -(FFT_SIZE_BASE - self.buffer_pos) :
            ]

        # Multi-resolution FFT processing (cache for performance)
        if self.frame_counter % 2 == 0:  # Process every other frame
            multi_fft_results = self.multi_fft.process_multi_resolution(audio_data, self.psychoacoustic_enabled)
            # Combine results into unified spectrum
            combined_magnitude = self.combine_multi_resolution_results(multi_fft_results)
            self.cached_multi_fft_results = multi_fft_results
            self.cached_combined_magnitude = combined_magnitude
        else:
            # Use cached results
            multi_fft_results = self.cached_multi_fft_results if hasattr(self, 'cached_multi_fft_results') else self.multi_fft.process_multi_resolution(audio_data, self.psychoacoustic_enabled)
            combined_magnitude = self.cached_combined_magnitude if hasattr(self, 'cached_combined_magnitude') else self.combine_multi_resolution_results(multi_fft_results)

        return {
            "combined": combined_magnitude,
            "multi_resolution": multi_fft_results,
            "base_fft": np.abs(np.fft.rfft(audio_buffer * np.blackman(len(audio_buffer)))),
        }

    def combine_multi_resolution_results(
        self, multi_fft_results: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Combine multi-resolution FFT results into unified spectrum"""
        # Create unified frequency array
        unified_freqs = np.fft.rfftfreq(FFT_SIZE_BASE, 1 / SAMPLE_RATE)
        unified_magnitude = np.zeros_like(unified_freqs)

        freq_arrays = self.multi_fft.get_frequency_arrays()

        # Combine results with frequency-dependent weighting
        for config_idx, magnitude in multi_fft_results.items():
            config_freqs = freq_arrays[config_idx]
            config = self.multi_fft.fft_configs[config_idx]

            freq_range = config["range"]

            # Interpolate to unified frequency grid
            for i, freq in enumerate(unified_freqs):
                if freq_range[0] <= freq <= freq_range[1]:
                    # Find nearest frequency in this config
                    nearest_idx = np.argmin(np.abs(config_freqs - freq))
                    if nearest_idx < len(magnitude):
                        # Weight by resolution advantage
                        weight = config["weight"]
                        unified_magnitude[i] = max(
                            unified_magnitude[i], magnitude[nearest_idx] * weight
                        )

        return unified_magnitude

    def process_frame(self):
        """Process one frame with enhanced professional analysis"""
        audio_data = self.capture.get_audio_data()
        if audio_data is None:
            # Even with no audio, ensure bar heights decay to allow peak hold to work
            self.bar_heights *= 0.9  # Quick decay when no audio
            return False

        start = time.perf_counter()
        self.frame_counter += 1
        
        # Save raw audio for VU meters (before gain)
        raw_audio_data = audio_data.copy()
        
        # Apply input gain
        audio_data = audio_data * self.input_gain
        
        # Auto-gain adjustment based on LUFS
        if self.auto_gain_enabled and hasattr(self, 'lufs_info'):
            current_lufs = self.lufs_info.get('short_term', -100)
            if current_lufs > -100:  # Valid LUFS reading
                # Calculate gain adjustment needed
                gain_adjustment = 10 ** ((self.target_lufs - current_lufs) / 20)
                # Smooth gain changes to avoid pumping
                self.gain_history.append(gain_adjustment)
                if len(self.gain_history) > 30:  # 0.5 seconds of history
                    smooth_gain = np.median(list(self.gain_history))
                    # Limit gain changes to ±6dB per adjustment
                    smooth_gain = np.clip(smooth_gain, 0.5, 2.0)
                    self.input_gain *= smooth_gain
                    # Keep total gain reasonable (0dB to +24dB)
                    self.input_gain = np.clip(self.input_gain, 1.0, 16.0)

        # Multi-resolution audio analysis
        fft_results = self.process_audio_chunk(audio_data)
        combined_fft = fft_results["combined"]

        # Map to enhanced bands using vectorized operations
        band_values = self._vectorized_band_mapping(combined_fft)

        # Apply frequency-dependent gain compensation for better high frequency response
        if self.freq_compensation_enabled:
            band_values = self.apply_frequency_compensation(band_values)

        # Normalize
        if self.normalization_enabled:
            max_val = np.max(band_values)
            if max_val > 0:
                band_values = band_values / max_val

        # Run expensive analysis only every few frames - spread across different frames
        analysis_skip = getattr(self, 'analysis_skip_frames', 2)
        
        # Spread expensive operations across different frames for better latency
        # Voice detection - ONLY when voice info displayed or formants shown
        if (self.show_voice_info or self.show_formants) and self.frame_counter % analysis_skip == 0:
            voice_info = self.voice_detector.detect_voice_realtime(audio_data)
            self.voice_info = voice_info
        else:
            voice_info = getattr(self, 'voice_info', self.voice_info)  # Use cached result
            
        # Drum detection - always run for visual beat reactivity
        if self.frame_counter % (analysis_skip + 1) == 0:
            drum_info = self.drum_detector.process_audio(combined_fft, band_values)
            self.drum_info = drum_info
        else:
            drum_info = getattr(self, 'drum_info', self.drum_info)  # Use cached result
        
        # Adaptive content detection (if enabled) - run less frequently
        if self.adaptive_allocation_enabled and self.frame_counter % 10 == 0:
            content_type = self.content_detector.analyze_content(voice_info, band_values, self.freq_starts, self.freq_ends)
            new_allocation = self.content_detector.get_allocation_for_content(content_type)
            
            # Update allocation if content type changed significantly
            if content_type != self.current_content_type or abs(new_allocation - self.current_allocation) > 0.05:
                self.current_content_type = content_type
                self.current_allocation = new_allocation
                # Note: Band remapping happens on next initialization cycle to avoid performance hit

        # Run very expensive analysis less frequently and spread across frames
        # LUFS metering every 4 frames - ONLY when meters shown
        # For debug output, we'll calculate on demand
        if self.show_professional_meters and self.frame_counter % 4 == 0:
            lufs_info = self.professional_metering.calculate_lufs(audio_data)
            true_peak = self.professional_metering.calculate_true_peak(audio_data)
            self.lufs_info = lufs_info
            self.true_peak = true_peak
        else:
            lufs_info = getattr(self, 'lufs_info', self.lufs_info)
            true_peak = getattr(self, 'true_peak', self.true_peak)
            
        # Harmonic analysis every 5 frames - ONLY when displayed
        if self.show_harmonic_analysis and self.frame_counter % 5 == 0:
            harmonic_info = self.harmonic_analyzer.detect_harmonic_series(combined_fft, self.freqs)
            self.harmonic_info = harmonic_info
        else:
            harmonic_info = getattr(self, 'harmonic_info', self.harmonic_info)
            
        # Transient analysis every 6 frames - ONLY when meters shown (used in meters)
        if self.show_professional_meters and self.frame_counter % 6 == 0:
            transient_info = self.transient_analyzer.analyze_transients(audio_data)
            self.transient_info = transient_info
        else:
            transient_info = getattr(self, 'transient_info', self.transient_info)
            
        # Room modes every 8 frames - ONLY when room analysis shown
        if self.show_room_analysis and self.frame_counter % 8 == 0:
            room_modes = self.room_analyzer.detect_room_modes(combined_fft, self.freqs)
            self.room_modes = room_modes
        else:
            room_modes = getattr(self, 'room_modes', self.room_modes)
            
        # OMEGA: Cepstral pitch detection every 3 frames - ONLY when displayed
        if self.show_pitch_detection and self.frame_counter % 3 == 0:
            # Use a larger window from the ring buffer for pitch detection
            pitch_window_size = min(self.cepstral_analyzer.yin_window_size * 2, len(self.ring_buffer))
            if self.buffer_pos >= pitch_window_size:
                pitch_audio = self.ring_buffer[self.buffer_pos - pitch_window_size : self.buffer_pos].copy()
            else:
                # Wrap around ring buffer
                pitch_audio = np.zeros(pitch_window_size, dtype=np.float32)
                if self.buffer_pos > 0:
                    pitch_audio[-self.buffer_pos:] = self.ring_buffer[:self.buffer_pos]
                    pitch_audio[:-self.buffer_pos] = self.ring_buffer[-(pitch_window_size - self.buffer_pos):]
                else:
                    pitch_audio = self.ring_buffer[-pitch_window_size:]
            
            pitch_info = self.cepstral_analyzer.detect_pitch_advanced(pitch_audio)
            self.pitch_info = pitch_info
        else:
            pitch_info = getattr(self, 'pitch_info', {
                'pitch': 0.0,
                'confidence': 0.0,
                'note': '',
                'cents_offset': 0,
                'octave': 0,
                'stability': 0.0
            })
            
        # OMEGA-1: Chromagram and key detection every 4 frames - ONLY when displayed
        if self.show_chromagram and self.frame_counter % 4 == 0:
            # Process chromagram from FFT data
            chromagram = self.chromagram_analyzer.compute_chromagram(combined_fft, self.freqs)
            
            # Detect musical key
            key_name, key_confidence = self.chromagram_analyzer.detect_key(chromagram)
            
            # Get key stability
            stability = self.chromagram_analyzer.get_key_stability()
            
            # Update chromagram info
            chromagram_info = {
                'chromagram': chromagram,
                'key': key_name,
                'is_major': 'major' in key_name.lower(),
                'confidence': key_confidence,
                'stability': stability,
                'alternative_keys': []  # ChromagramAnalyzer doesn't have multi-key detection yet
            }
        else:
            # Use cached chromagram info
            chromagram_info = self.chromagram_info

        # Update flash times
        if drum_info["kick"]["kick_detected"]:
            self.kick_flash_time = time.time()
        if drum_info["snare"]["snare_detected"]:
            self.snare_flash_time = time.time()
        if voice_info["has_voice"] and voice_info["voice_confidence"] > 0.7:
            self.voice_flash_time = time.time()
        if voice_info.get("is_singing", False):
            self.singing_flash_time = time.time()

        # Store all analysis results
        self.drum_info = drum_info
        self.voice_info = voice_info
        self.lufs_info = lufs_info
        self.true_peak = true_peak
        self.harmonic_info = harmonic_info
        self.transient_info = transient_info
        self.room_modes = room_modes
        self.pitch_info = pitch_info  # OMEGA: Store pitch detection results
        self.chromagram_info = chromagram_info  # OMEGA-1: Store chromagram results

        # Enhanced voice-reactive spectrum smoothing using vectorized operations
        if self.smoothing_enabled:
            self._vectorized_smoothing(band_values, drum_info, voice_info)
        else:
            # Direct assignment without smoothing
            self.bar_heights = band_values
        
        # Process bass detail every 4th frame for better performance
        # Since it's async, we can afford to run it less frequently
        if self.frame_counter % 4 == 0:
            self.process_bass_detail(audio_data)
        
        # Apply bass results from async thread
        self.apply_bass_results()
        
        # Calculate processing time
        self.last_process_time = time.perf_counter() - start
        
        # Process VU meters with raw audio data (before gain)
        if self.show_vu_meters:
            self.process_vu_meters(raw_audio_data, self.last_process_time)
        
        # Dynamic quality adjustment based on latency
        if self.dynamic_quality_enabled:
            latency_ms = self.last_process_time * 1000
            current_time = time.time()
            time_since_switch = current_time - self.last_quality_switch_time
            
            # Only allow mode switches if hold time has elapsed
            if time_since_switch >= self.quality_mode_hold_time:
                # Switch to performance mode if latency is high
                if self.quality_mode == "quality" and latency_ms > self.performance_threshold:
                    self.quality_mode = "performance"
                    self._switch_to_performance_mode()
                    self.last_quality_switch_time = current_time
                    print(f"⚡ Switched to performance mode (latency: {latency_ms:.1f}ms)")
                
                # Switch back to quality mode if latency is low
                elif self.quality_mode == "performance" and latency_ms < self.quality_threshold:
                    self.quality_mode = "quality"
                    self._switch_to_quality_mode()
                    self.last_quality_switch_time = current_time
                    print(f"✨ Switched to quality mode (latency: {latency_ms:.1f}ms)")
        
        return True

    def _precalculate_freq_compensation(self) -> np.ndarray:
        """Pre-calculate frequency compensation gains for balanced visual response"""
        gains = np.ones(self.bars, dtype=np.float32)

        for i in range(self.bars):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2

            if center_freq > 0:
                # More aggressive compensation with fixed bass boundary
                if center_freq < 60:
                    gain = 0.15   # Aggressive reduction for sub-bass
                elif center_freq < 100:
                    gain = 0.2    # Strong reduction for bass
                elif center_freq < 250:
                    gain = 0.3    # Moderate reduction for ALL bass (fixed boundary)
                elif center_freq < 500:
                    gain = 3.0    # Strong boost for low-mids
                elif center_freq < 1000:
                    gain = 4.5    # Very strong boost for mids
                elif center_freq < 2000:
                    gain = 5.0    # Maximum boost for upper mids
                elif center_freq < 4000:
                    gain = 4.0    # Strong boost for presence
                elif center_freq < 8000:
                    gain = 2.0    # Moderate boost for highs
                elif center_freq < 12000:
                    gain = 0.8    # Slight reduction for upper highs
                else:
                    gain = 0.5    # Moderate reduction for air

                # Apply additional midrange boost if enabled (1k-6k Hz enhancement)
                if self.midrange_boost_enabled and 1000 <= center_freq <= 6000:
                    gain *= self.midrange_boost_factor

                gains[i] = gain

        return gains

    def _setup_optimized_band_mapping(self):
        """Pre-compute band mapping arrays for faster processing"""
        # Create lists to store band mapping info
        self.band_starts = []
        self.band_ends = []
        self.band_counts = []
        
        for indices in self.band_indices:
            if len(indices) > 0:
                self.band_starts.append(indices[0])
                self.band_ends.append(indices[-1] + 1)
                self.band_counts.append(len(indices))
            else:
                self.band_starts.append(0)
                self.band_ends.append(0)
                self.band_counts.append(0)
        
        # Convert to numpy arrays for faster access
        self.band_starts = np.array(self.band_starts, dtype=np.int32)
        self.band_ends = np.array(self.band_ends, dtype=np.int32)
        self.band_counts = np.array(self.band_counts, dtype=np.float32)
    
    def _vectorized_band_mapping(self, fft_data: np.ndarray) -> np.ndarray:
        """Optimized band mapping using pre-computed indices"""
        band_values = np.zeros(self.bars, dtype=np.float32)
        
        # Vectorized computation where possible
        for i in range(min(self.bars, len(self.band_counts))):
            if self.band_counts[i] > 0:
                start = self.band_starts[i]
                end = self.band_ends[i]
                band_values[i] = np.mean(fft_data[start:end])
        
        return band_values
    
    def _setup_frequency_ranges(self):
        """Pre-calculate frequency ranges for each bar"""
        self.freq_starts = np.zeros(self.bars, dtype=np.float32)
        self.freq_ends = np.zeros(self.bars, dtype=np.float32)
        
        for i in range(self.bars):
            start, end = self.get_frequency_range_for_bar(i)
            self.freq_starts[i] = start
            self.freq_ends[i] = end
    
    def _build_frequency_lookup(self):
        """Build frequency to bar index lookup for O(1) access"""
        self.freq_to_bar = {}
        
        # Build lookup table for main spectrum bars
        for i in range(self.bars):
            freq_range = self.get_frequency_range_for_bar(i)
            if freq_range[0] > 0:
                # Map every 10Hz increment to nearest bar
                for freq in range(int(freq_range[0]), int(freq_range[1]) + 1, 10):
                    self.freq_to_bar[freq] = i
                    
        # Add specific lookups for common formant frequencies
        formant_freqs = [700, 800, 1220, 1400, 2600, 2800, 3400, 3600]
        for freq in formant_freqs:
            # Find closest bar
            best_bar = 0
            best_dist = float('inf')
            for i in range(self.bars):
                f_range = self.get_frequency_range_for_bar(i)
                if f_range[0] <= freq <= f_range[1]:
                    self.freq_to_bar[freq] = i
                    break
                center = (f_range[0] + f_range[1]) / 2
                dist = abs(freq - center)
                if dist < best_dist:
                    best_dist = dist
                    best_bar = i
            if freq not in self.freq_to_bar:
                self.freq_to_bar[freq] = best_bar
    
    def get_bar_for_frequency(self, freq: float) -> int:
        """Fast O(1) frequency to bar lookup"""
        # Round to nearest 10Hz for lookup
        lookup_freq = int(round(freq / 10) * 10)
        return self.freq_to_bar.get(lookup_freq, 0)
    
    def _setup_bass_frequency_mapping(self):
        """Create frequency mapping for bass detail panel using actual FFT resolution"""
        bass_fft_size = 8192  # Larger FFT for better bass resolution
        bass_freqs = np.fft.rfftfreq(bass_fft_size, 1 / SAMPLE_RATE)
        
        # Find FFT bins that cover our bass range (20-200Hz)
        bass_min_freq = 20.0
        bass_max_freq = 200.0
        valid_bins = np.where((bass_freqs >= bass_min_freq) & (bass_freqs <= bass_max_freq))[0]
        
        if len(valid_bins) == 0:
            # Fallback if no valid bins
            self.bass_freq_ranges = [(20, 200)]
            self.bass_bin_mapping = [[]]
            self.bass_detail_bars = 1
            return
        
        # Create frequency ranges based on actual FFT bins for complete coverage
        self.bass_freq_ranges = []
        self.bass_bin_mapping = []
        
        # Group bins to create smooth visualization
        bins_per_bar = max(1, len(valid_bins) // 31)  # Target ~31 bars for good visual density
        
        for i in range(0, len(valid_bins), bins_per_bar):
            end_idx = min(i + bins_per_bar, len(valid_bins))
            bin_group = valid_bins[i:end_idx]
            
            if len(bin_group) > 0:
                f_start = bass_freqs[bin_group[0]]
                f_end = bass_freqs[bin_group[-1]]
                
                # Ensure ranges don't overlap and cover gaps
                if len(self.bass_freq_ranges) > 0:
                    f_start = max(f_start, self.bass_freq_ranges[-1][1])
                
                self.bass_freq_ranges.append((f_start, f_end))
                self.bass_bin_mapping.append(bin_group)
        
        # Update the actual number of bass bars based on what we created
        self.bass_detail_bars = len(self.bass_freq_ranges)
        
        # Initialize arrays with correct size
        self.bass_bar_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_timestamps = np.zeros(self.bass_detail_bars, dtype=np.float64)
    
    def _vectorized_smoothing(self, band_values: np.ndarray, drum_info: Dict, voice_info: Dict):
        """Vectorized smoothing for better performance"""
        # Pre-calculate smoothing parameters based on frequency ranges
        attack = np.zeros(self.bars, dtype=np.float32)
        release = np.zeros(self.bars, dtype=np.float32)
        
        # Determine attack/release rates based on frequency ranges
        ultra_low_mask = self.freq_starts <= 100
        low_mask = (self.freq_starts > 100) & (self.freq_starts <= 500)
        mid_mask = (self.freq_starts > 500) & (self.freq_starts <= 3000)
        high_mask = self.freq_starts > 3000
        
        # Set base attack/release rates
        kick_detected = drum_info["kick"]["kick_detected"]
        snare_detected = drum_info["snare"]["snare_detected"]
        has_voice = voice_info["has_voice"]
        
        # Ultra-low frequencies
        attack[ultra_low_mask] = 0.98 if kick_detected else 0.75
        release[ultra_low_mask] = 0.06
        
        # Low frequencies
        attack[low_mask] = 0.95 if (kick_detected or snare_detected) else 0.80
        release[low_mask] = 0.10
        
        # Mid frequencies - faster attack for better vocal response
        voice_attack = 0.95 if has_voice else 0.85  # Faster when voice detected
        attack[mid_mask] = voice_attack
        release[mid_mask] = 0.15
        
        # High frequencies
        attack[high_mask] = 0.75
        release[high_mask] = 0.25
        
        # Vectorized smoothing calculation
        rising_mask = band_values > self.bar_heights
        falling_mask = ~rising_mask
        
        # Apply smoothing
        self.bar_heights[rising_mask] = (
            self.bar_heights[rising_mask] + 
            (band_values[rising_mask] - self.bar_heights[rising_mask]) * attack[rising_mask]
        )
        self.bar_heights[falling_mask] = (
            self.bar_heights[falling_mask] + 
            (band_values[falling_mask] - self.bar_heights[falling_mask]) * release[falling_mask]
        )
    
    def apply_frequency_compensation(self, bar_values: np.ndarray) -> np.ndarray:
        """Apply pre-calculated frequency-dependent gain compensation"""
        return bar_values * self.freq_compensation_gains
    
    def _bass_processor_worker(self):
        """Worker thread for async bass processing"""
        while self.bass_thread_running:
            try:
                audio_data = self.bass_processing_queue.get(timeout=0.1)
                if audio_data is None:
                    break
                    
                # Process bass detail in background thread
                result = self._process_bass_detail_internal(audio_data)
                self.latest_bass_result = result
                
            except Empty:
                continue
            except Exception as e:
                import traceback
                print(f"Bass processing error: {type(e).__name__}: {e}")
                traceback.print_exc()
                
    def submit_bass_processing(self, audio_data: np.ndarray):
        """Submit audio data for async bass processing"""
        try:
            # Non-blocking put - skip if queue is full
            self.bass_processing_queue.put_nowait(audio_data.copy())
        except:
            pass  # Skip this frame if queue is full
            
    def apply_bass_results(self):
        """Apply the latest bass processing results if available"""
        if self.latest_bass_result is not None:
            self.bass_bar_values = self.latest_bass_result['bar_values']
            self.bass_peak_values = self.latest_bass_result['peak_values']
            self.bass_peak_timestamps = self.latest_bass_result['peak_timestamps']
            self.latest_bass_result = None

    def _process_bass_detail_internal(self, audio_data: np.ndarray):
        """Internal bass processing logic (runs on separate thread)"""
        # This is the original process_bass_detail logic, but returns results instead of setting them directly
        bass_fft_size = 8192
        
        # Apply window
        window = np.hanning(min(len(audio_data), bass_fft_size))
        if len(audio_data) < bass_fft_size:
            padded_audio = np.zeros(bass_fft_size)
            padded_audio[:len(audio_data)] = audio_data * window
            windowed_audio = padded_audio
        else:
            windowed_audio = audio_data[:bass_fft_size] * window
        
        # Compute FFT
        bass_fft = np.fft.rfft(windowed_audio)
        bass_magnitude = np.abs(bass_fft)
        
        # Create copies for thread safety
        bar_values = self.bass_bar_values.copy()
        peak_values = self.bass_peak_values.copy()
        peak_timestamps = self.bass_peak_timestamps.copy()
        
        # Calculate dynamic range for proper scaling
        bass_max = 0.0
        raw_values = {}
        
        # First pass: collect raw values
        for i, bin_indices in enumerate(self.bass_bin_mapping):
            if i >= len(bar_values):
                break  # Prevent index out of bounds
            if len(bin_indices) > 0:
                bar_value = np.mean(bass_magnitude[bin_indices])
                if i >= len(self.bass_freq_ranges):
                    continue  # Skip if freq_ranges doesn't have this index
                center_freq = (self.bass_freq_ranges[i][0] + self.bass_freq_ranges[i][1]) / 2
                
                # Apply frequency compensation
                if center_freq < 60:
                    bar_value *= 0.3
                elif center_freq < 100:
                    bar_value *= 0.6
                elif center_freq < 150:
                    bar_value *= 1.0
                else:
                    bar_value *= 0.8
                
                raw_values[i] = bar_value
                bass_max = max(bass_max, bar_value)
        
        # Dynamic scaling
        if bass_max > 0:
            scale_factor = 0.85 / bass_max
            log_scale = np.log10(max(1.0, bass_max * 10)) / 2.0
            scale_factor *= log_scale
        else:
            scale_factor = 1.0
        
        # Second pass: apply scaling and update bars
        current_time = time.time()
        for i, bin_indices in enumerate(self.bass_bin_mapping):
            if i >= len(bar_values):
                break  # Prevent index out of bounds
            if len(bin_indices) > 0 and i in raw_values:
                scaled_value = raw_values[i] * scale_factor
                
                # Apply compression curve
                if scaled_value > 0.7:
                    compressed_value = 0.7 + (scaled_value - 0.7) * 0.3
                else:
                    compressed_value = scaled_value
                
                # Smooth the bass values
                if compressed_value > bar_values[i]:
                    bar_values[i] = bar_values[i] * 0.1 + compressed_value * 0.9
                else:
                    bar_values[i] = bar_values[i] * 0.6 + compressed_value * 0.4
                
                # Clamp to 0-1 range
                bar_values[i] = max(0.0, min(1.0, bar_values[i]))
                
                # Update peak hold
                if bar_values[i] > peak_values[i]:
                    peak_values[i] = bar_values[i]
                    peak_timestamps[i] = current_time
                elif current_time - peak_timestamps[i] > 3.0:
                    peak_values[i] *= 0.95
        
        return {
            'bar_values': bar_values,
            'peak_values': peak_values,
            'peak_timestamps': peak_timestamps
        }

    def process_bass_detail(self, audio_data: np.ndarray):
        """Submit bass processing to async thread"""
        # Submit to async processing thread
        self.submit_bass_processing(audio_data)
        # Bass processing now happens asynchronously
    
    def _switch_to_performance_mode(self):
        """Switch to performance mode for lower latency"""
        # Reduce quality settings for better performance
        self.target_bars = 512  # Reduce from 698
        self.skip_expensive_analysis = True
        # Process expensive analysis less frequently
        self.analysis_skip_frames = 4  # Was 2
        
    def _switch_to_quality_mode(self):
        """Switch back to quality mode"""
        # Restore quality settings
        self.target_bars = 698
        self.skip_expensive_analysis = False
        self.analysis_skip_frames = 2  # Back to normal

    def show_help_controls(self):
        """Display all available keyboard controls"""
        print("\n" + "="*90)
        print("🎛️  PROFESSIONAL AUDIO ANALYZER v4 - COMPLETE KEYBOARD CONTROLS")
        print("="*90)
        
        print("\n📊 DISPLAY TOGGLES:")
        print("M: Toggle professional meters              H: Toggle harmonic analysis")
        print("R: Toggle room analysis                    Z: Toggle bass zoom window")
        print("G: Toggle analysis grid                    B: Toggle band separators")
        print("T: Toggle technical overlay                V: Toggle voice info")
        print("F: Toggle formants                         A: Toggle advanced info")
        print("J: Toggle VU meters                        (Professional VU metering)")
        
        print("\n🎚️  AUDIO PROCESSING:")
        print("Q: Toggle frequency compensation          W: Toggle psychoacoustic weighting")
        print("E: Toggle normalization                   U: Toggle smoothing")
        print("P: Toggle dynamic quality mode            (Auto performance optimization)")
        print("+/=: Increase input gain                  -/_: Decrease input gain")
        print("I: Toggle midrange boost (1k-6k)          O/L: Increase/Decrease boost factor")
        
        print("\n⚖️  A/B COMPARISON:")
        print("C: Store reference / Toggle comparison    X: Clear reference spectrum")
        
        print("\n🖼️  WINDOW & DISPLAY:")
        print("S: Save screenshot                        ESC: Exit analyzer")
        print("0: Toggle fullscreen                      D: Print debug output")
        print("1: 1400x900 (Compact)                    2: 1600x1000 (Standard)")
        print("3: 1920x1200 (Large)                     4: 2200x1400 (XL)")
        print("5: 2560x1600 (2K)                        6: 3000x1800 (Ultra)")
        print("7: 2048x1152 (Wide)                      8: 2560x1440 (2K Monitor)")
        print("9: 3840x2160 (4K Monitor)")
        
        print("\n🆘 HELP:")
        print("?: Show this help display                 /: Show this help display")
        
        print("\n🎵 CURRENT STATUS:")
        status_items = [
            f"Professional meters: {'ON' if self.show_professional_meters else 'OFF'}",
            f"Harmonic analysis: {'ON' if self.show_harmonic_analysis else 'OFF'}",
            f"Room analysis: {'ON' if self.show_room_analysis else 'OFF'}",
            f"Bass zoom: {'ON' if self.show_bass_zoom else 'OFF'}",
            f"Analysis grid: {'ON' if self.show_frequency_grid else 'OFF'}",
            f"Band separators: {'ON' if self.show_band_separators else 'OFF'}",
            f"Technical overlay: {'ON' if self.show_technical_overlay else 'OFF'}",
            f"Voice info: {'ON' if self.show_voice_info else 'OFF'}",
            f"Formants: {'ON' if self.show_formants else 'OFF'}",
            f"Advanced info: {'ON' if self.show_advanced_info else 'OFF'}",
            f"Frequency compensation: {'ON' if self.freq_compensation_enabled else 'OFF'}",
            f"Psychoacoustic weighting: {'ON' if self.psychoacoustic_enabled else 'OFF'}",
            f"Normalization: {'ON' if self.normalization_enabled else 'OFF'}",
            f"Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}",
            f"A/B comparison: {'ON' if hasattr(self, 'comparison_mode') and self.comparison_mode else 'OFF'}",
            f"Input gain: {20 * np.log10(self.input_gain):.1f}dB",
            f"Midrange boost: {'ON' if self.midrange_boost_enabled else 'OFF'} ({self.midrange_boost_factor:.1f}x)",
            f"VU meters: {'ON' if self.show_vu_meters else 'OFF'}"
        ]
        
        # Display in two columns
        for i in range(0, len(status_items), 2):
            left = status_items[i].ljust(45)
            right = status_items[i+1] if i+1 < len(status_items) else ""
            print(f"{left} {right}")
            
        print("="*90)
        print("💡 TIP: Press '?' or '/' anytime to show this help again")
        print("="*90 + "\n")

    def get_frequency_range_for_bar(self, bar_index: int) -> Tuple[float, float]:
        """Get frequency range for a specific bar - ensures continuous coverage"""
        if bar_index < len(self.band_indices):
            indices = self.band_indices[bar_index]
            if len(indices) > 0:
                # Get actual frequency range covered by this bar
                freq_start = self.freqs[indices[0]]
                freq_end = self.freqs[indices[-1]]
                
                # Special case for first bar - ensure it starts at 20Hz minimum
                if bar_index == 0:
                    freq_start = max(20.0, freq_start)

                # For single-bin bars, extend the range slightly
                if len(indices) == 1:
                    # Use half the distance to neighboring bins
                    if indices[0] > 0 and bar_index > 0:  # Don't adjust start of first bar
                        prev_freq = self.freqs[indices[0] - 1]
                        freq_start = (prev_freq + freq_start) / 2
                    if indices[0] < len(self.freqs) - 1:
                        next_freq = self.freqs[indices[0] + 1]
                        freq_end = (freq_end + next_freq) / 2

                return (freq_start, freq_end)
        return (0, 0)

    def freq_to_x_position(self, frequency: float) -> float:
        """Convert frequency to x position in spectrum display using frequency edges"""
        # Calculate spectrum display width
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_width = self.width - meter_width - vu_width - self.left_border_width - 10
        bar_width = spectrum_width / self.bars
        
        # Handle edge cases
        if frequency <= 20:
            return self.left_border_width
        if frequency >= 20000:
            return self.left_border_width + spectrum_width
        
        # Find which bar contains this frequency using freq_edges
        if hasattr(self, 'freq_edges'):
            bar_idx = -1
            for i in range(len(self.freq_edges) - 1):
                if self.freq_edges[i] <= frequency < self.freq_edges[i + 1]:
                    bar_idx = i
                    break
            
            if bar_idx == -1:
                # Frequency is above the highest edge
                if frequency >= self.freq_edges[-1]:
                    bar_idx = len(self.freq_edges) - 2
                else:
                    # Find nearest bar
                    differences = np.abs(self.freq_edges[:-1] - frequency)
                    bar_idx = np.argmin(differences)
            
            # Calculate position within the bar
            freq_start = self.freq_edges[bar_idx]
            freq_end = self.freq_edges[bar_idx + 1]
            
            # Use logarithmic interpolation for better visual distribution
            if freq_end > freq_start:
                log_position = (np.log10(frequency) - np.log10(freq_start)) / (np.log10(freq_end) - np.log10(freq_start))
                log_position = np.clip(log_position, 0, 1)
            else:
                log_position = 0.5
            
            # Calculate bar position
            bar_start_x = self.left_border_width + bar_idx * bar_width
            
            # Return interpolated position within the bar
            return bar_start_x + log_position * bar_width
        
        # Fallback: if freq_edges not available, use logarithmic distribution
        log_freq = np.log10(frequency)
        log_min = np.log10(20)
        log_max = np.log10(20000)
        normalized_pos = (log_freq - log_min) / (log_max - log_min)
        return self.left_border_width + normalized_pos * spectrum_width

    def update_peak_holds(self):
        """Update peak hold indicators with time-based decay"""
        current_time = time.time()
        # Ensure we don't exceed array bounds
        num_bars = min(len(self.bar_heights), len(self.peak_hold_bars), len(self.peak_timestamps))
        
        for i in range(num_bars):
            # Update peak hold if current bar is significantly higher (with small threshold)
            if self.bar_heights[i] > self.peak_hold_bars[i] + 0.02:  # Small threshold to prevent jitter
                self.peak_hold_bars[i] = self.bar_heights[i]
                self.peak_timestamps[i] = current_time
            
            # Always check for decay, regardless of current bar height
            if self.peak_timestamps[i] > 0:  # Has a valid timestamp
                time_since_peak = current_time - self.peak_timestamps[i]
                
                if time_since_peak > self.peak_hold_time:
                    # Calculate time-based decay
                    decay_time = time_since_peak - self.peak_hold_time
                    # Exponential decay: reduce by 50% every 0.3 seconds after hold time
                    decay_factor = 0.5 ** (decay_time / 0.3)
                    
                    new_value = self.peak_hold_bars[i] * decay_factor
                    
                    # If decayed below threshold, reset completely
                    if new_value < 0.01:
                        self.peak_hold_bars[i] = 0.0
                        self.peak_timestamps[i] = 0.0  # Mark as cleared
                    else:
                        self.peak_hold_bars[i] = new_value
            
            # Force clear very old peaks that might be stuck
            elif self.peak_hold_bars[i] > 0 and self.peak_timestamps[i] == 0:
                # Peak exists but no timestamp - force decay
                self.peak_hold_bars[i] *= 0.9

    def calculate_sub_bass_energy(self) -> float:
        """Calculate sub-bass energy (20-60Hz)"""
        sub_bass_sum = 0.0
        sub_bass_count = 0

        for i, (freq_start, freq_end) in enumerate(
            [self.get_frequency_range_for_bar(j) for j in range(min(50, self.bars))]
        ):
            if 20 <= freq_start <= 60 or 20 <= freq_end <= 60:
                sub_bass_sum += self.bar_heights[i]
                sub_bass_count += 1

        if sub_bass_count > 0:
            energy = sub_bass_sum / sub_bass_count
            self.sub_bass_history.append(energy)
            self.sub_bass_energy = energy

            # Check for sub-bass warning
            if energy > 0.8:
                self.sub_bass_warning_active = True
                self.sub_bass_warning_time = time.time()
            elif time.time() - self.sub_bass_warning_time > 2.0:
                self.sub_bass_warning_active = False

            return energy
        return 0.0

    def draw_sub_bass_indicator(self):
        """Draw sub-bass energy indicator and warning"""
        sub_bass_energy = self.calculate_sub_bass_energy()

        # Sub-bass meter positioned in left border area
        meter_height = 200
        meter_width = 25  # Slightly wider for better visibility
        x_pos = (self.left_border_width - meter_width) // 2  # Center in left border
        y_pos = self.height // 2 - meter_height // 2

        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (x_pos, y_pos, meter_width, meter_height))
        pygame.draw.rect(self.screen, (100, 100, 120), (x_pos, y_pos, meter_width, meter_height), 1)

        # Energy level
        level_height = int(sub_bass_energy * meter_height)
        if level_height > 0:
            if sub_bass_energy > 0.8:
                color = (255, 100, 100)  # Red warning
            elif sub_bass_energy > 0.6:
                color = (255, 200, 100)  # Orange caution
            else:
                color = (100, 255, 150)  # Green normal

            pygame.draw.rect(
                self.screen,
                color,
                (x_pos, y_pos + meter_height - level_height, meter_width, level_height),
            )

        # Labels
        label_text = self.font_tiny.render("SUB", True, (150, 150, 170))
        self.screen.blit(label_text, (x_pos - 5, y_pos - 20))

        # Warning if too much sub-bass
        if self.sub_bass_warning_active:
            warning_text = self.font_small.render("SUB!", True, (255, 100, 100))
            self.screen.blit(warning_text, (x_pos - 10, y_pos - 45))

    def draw_adaptive_allocation_indicator(self):
        """Draw indicator for adaptive frequency allocation"""
        # Position in top-right corner
        indicator_width = 150
        indicator_height = 60
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        x_pos = self.width - meter_width - vu_width - indicator_width - 20
        y_pos = 80
        
        # Background
        bg_color = (20, 25, 35) if self.adaptive_allocation_enabled else (35, 25, 20)
        pygame.draw.rect(self.screen, bg_color, (x_pos, y_pos, indicator_width, indicator_height))
        pygame.draw.rect(self.screen, (80, 80, 100), (x_pos, y_pos, indicator_width, indicator_height), 1)
        
        # Status text
        status_color = (100, 255, 150) if self.adaptive_allocation_enabled else (150, 150, 150)
        status_text = "ADAPTIVE" if self.adaptive_allocation_enabled else "FIXED"
        status_surface = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surface, (x_pos + 5, y_pos + 5))
        
        # Content type and allocation
        if self.adaptive_allocation_enabled:
            content_color = {
                'music': (255, 150, 100),
                'speech': (150, 200, 255), 
                'mixed': (255, 255, 150),
                'instrumental': (200, 150, 255)
            }.get(self.current_content_type, (200, 200, 200))
            
            content_surface = self.font_tiny.render(f"{self.current_content_type.upper()}", True, content_color)
            self.screen.blit(content_surface, (x_pos + 5, y_pos + 25))
            
            allocation_text = f"{self.current_allocation*100:.0f}% LOW"
            allocation_surface = self.font_tiny.render(allocation_text, True, (200, 200, 200))
            self.screen.blit(allocation_surface, (x_pos + 5, y_pos + 40))
        else:
            fixed_surface = self.font_tiny.render("75% LOW", True, (150, 150, 150))
            self.screen.blit(fixed_surface, (x_pos + 5, y_pos + 25))

    def draw_frequency_band_separators(self):
        """Draw vertical lines to separate frequency bands"""
        if not self.show_band_separators:
            return

        band_boundaries = [
            (60, "SUB", (120, 80, 140)),
            (250, "BASS", (140, 100, 120)),
            (500, "L-MID", (120, 120, 100)),
            (2000, "MID", (100, 140, 120)),
            (6000, "H-MID", (100, 120, 140)),
        ]

        spectrum_top = 280 + 10  # Header height + gap
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50  # Match main spectrum area

        for freq, label, color in band_boundaries:
            x_pos = int(self.freq_to_x_position(freq))

            # Draw separator line
            pygame.draw.line(
                self.screen, color + (128,), (x_pos, spectrum_top), (x_pos, spectrum_bottom), 2
            )

            # Draw label
            label_surface = self.font_tiny.render(label, True, color)
            self.screen.blit(label_surface, (x_pos + 5, spectrum_top + 5))

    def detect_and_display_room_modes(self):
        """Detect potential room modes in the bass region"""
        current_time = time.time()
        room_mode_threshold = 0.6
        min_duration = 0.3  # 300ms minimum

        # Check bass region for sustained peaks
        for i in range(min(150, self.bars)):  # Check first 150 bars (roughly 20-200Hz)
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2

            if 30 <= center_freq <= 300 and self.bar_heights[i] > room_mode_threshold:
                # Check if this is a new peak or continuation
                if center_freq not in self.sustained_peaks:
                    self.sustained_peaks[center_freq] = (
                        self.bar_heights[i],
                        current_time,
                        int(self.freq_to_x_position(center_freq)),
                    )
                else:
                    # Update existing peak
                    magnitude, start_time, x_pos = self.sustained_peaks[center_freq]
                    self.sustained_peaks[center_freq] = (
                        max(magnitude, self.bar_heights[i]),
                        start_time,
                        x_pos,
                    )

        # Clean up old peaks and identify room modes
        self.room_mode_candidates = []
        to_remove = []

        for freq, (magnitude, start_time, x_pos) in self.sustained_peaks.items():
            duration = current_time - start_time

            # Remove if no longer sustained
            if duration > 0.1:  # Allow 100ms gap
                current_magnitude = 0
                for i in range(self.bars):
                    freq_start, freq_end = self.get_frequency_range_for_bar(i)
                    if freq_start <= freq <= freq_end:
                        current_magnitude = self.bar_heights[i]
                        break

                if current_magnitude < room_mode_threshold * 0.5:
                    to_remove.append(freq)
                    continue

            # If sustained long enough, mark as room mode
            if duration > min_duration:
                self.room_mode_candidates.append((freq, magnitude, duration, x_pos))

        # Remove expired peaks
        for freq in to_remove:
            del self.sustained_peaks[freq]

        # Draw room mode indicators
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50
        for freq, magnitude, duration, x_pos in self.room_mode_candidates:
            # Draw room mode indicator
            pygame.draw.circle(self.screen, (255, 100, 100, 180), (x_pos, spectrum_bottom + 20), 8)

            # Label with frequency
            mode_text = self.font_tiny.render(f"{freq:.1f}Hz", True, (255, 150, 150))
            self.screen.blit(mode_text, (x_pos - 25, spectrum_bottom + 35))

    def draw_enhanced_analysis_grid(self):
        """Draw enhanced grid lines for easier reading"""
        if not self.show_frequency_grid:
            return

        # Match the dimensions used in main draw method
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_left = self.left_border_width
        spectrum_right = self.width - meter_width - vu_width - 10
        spectrum_top = 280 + 10  # Header height + gap
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50

        # Skip horizontal dB lines - already drawn by draw_smart_grid_labels()
        # Only draw vertical frequency lines at musical intervals
        octave_frequencies = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in octave_frequencies:
            x_pos = int(self.freq_to_x_position(freq))
            if spectrum_left <= x_pos <= spectrum_right:
                pygame.draw.line(
                    self.screen,
                    (40, 40, 50, 80),
                    (x_pos, spectrum_top),
                    (x_pos, spectrum_bottom),
                    1,
                )
                
                # Skip frequency labels - already shown in frequency scale

    def draw_peak_hold_indicators(self):
        """Draw peak hold lines for all frequency bars (mirrored for spectrum)"""
        # Use same dimensions as main spectrum visualization
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        vis_width = self.width - meter_width - vu_width - self.left_border_width - 10
        
        # Match spectrum area calculations
        header_height = 280
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        vis_height = self.height - header_height - bass_zoom_height - 50
        vis_start_x = self.left_border_width
        vis_start_y = header_height + 10
        
        center_y = vis_start_y + vis_height // 2
        max_bar_height = (vis_height // 2) - 10
        bar_width = vis_width / self.bars

        # Ensure we don't draw more peaks than we have bars
        num_bars_to_draw = min(len(self.peak_hold_bars), self.bars)
        
        for i in range(num_bars_to_draw):
            peak = self.peak_hold_bars[i]
            if peak > 0.05:  # Only draw significant peaks
                x = vis_start_x + i * bar_width
                # Clamp peak height to prevent overflow
                clamped_peak = min(peak, 1.0)
                peak_height = int(clamped_peak * max_bar_height)
                
                # Draw peak hold line above center (upper spectrum)
                upper_y = center_y - peak_height
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255, 180),
                    (int(x), upper_y),
                    (int(x + bar_width - 1), upper_y),
                    1,
                )
                
                # Draw peak hold line below center (lower spectrum - mirrored)
                lower_y = center_y + peak_height
                pygame.draw.line(
                    self.screen,
                    (200, 200, 200, 140),  # Slightly dimmer for lower mirror
                    (int(x), lower_y),
                    (int(x + bar_width - 1), lower_y),
                    1,
                )

    def draw_formant_overlays(self, vis_start_x, vis_start_y, vis_width, vis_height, center_y):
        """Draw formant frequency overlays on the spectrum"""
        if not self.show_formants or not hasattr(self, "voice_info") or not self.voice_info.get("has_voice", False):
            return
        
        # Common formant frequencies for male and female voices
        # F1 (openness), F2 (frontness), F3 (lip rounding), F4 (timbre)
        male_formants = {
            "F1": (700, "Openness"),     # 400-800 Hz typical range
            "F2": (1220, "Frontness"),   # 800-1600 Hz typical range  
            "F3": (2600, "Rounding"),    # 2000-3200 Hz typical range
            "F4": (3400, "Timbre")       # 3000-4000 Hz typical range
        }
        
        female_formants = {
            "F1": (800, "Openness"),     # 500-1000 Hz typical range
            "F2": (1400, "Frontness"),   # 1000-1800 Hz typical range
            "F3": (2800, "Rounding"),    # 2200-3400 Hz typical range  
            "F4": (3600, "Timbre")       # 3200-4200 Hz typical range
        }
        
        # Use detected pitch to guess gender (rough approximation)
        pitch = self.voice_info.get("pitch", 150)
        formants = female_formants if pitch > 200 else male_formants
        
        # Draw formant frequency lines
        colors = {
            "F1": (255, 100, 100),  # Red for F1
            "F2": (100, 255, 100),  # Green for F2  
            "F3": (100, 100, 255),  # Blue for F3
            "F4": (255, 255, 100)   # Yellow for F4
        }
        
        for formant_name, (freq, description) in formants.items():
            # Find bar position for this frequency using fast lookup
            bar_pos = self.get_bar_for_frequency(freq)
            
            if bar_pos >= 0:  # Valid bar position (0 is valid)
                bar_width = vis_width / self.bars
                x = vis_start_x + int(bar_pos * bar_width)
                color = colors[formant_name]
                
                # Draw vertical line
                pygame.draw.line(
                    self.screen, color,
                    (x, vis_start_y), 
                    (x, vis_start_y + vis_height),
                    2
                )
                
                # Draw label at top
                label_text = f"{formant_name}: {freq}Hz"
                label_surface = self.font_tiny.render(label_text, True, color)
                label_y = vis_start_y + 10 + (list(formants.keys()).index(formant_name) * 15)
                self.screen.blit(label_surface, (x + 5, label_y))
                
                # Draw marker at spectrum level - make more visible
                current_bar_height = self.bar_heights[bar_pos] if bar_pos < len(self.bar_heights) else 0
                if current_bar_height > 0.005:  # Lower threshold
                    max_bar_height = (vis_height // 2) - 10
                    # Clamp bar height to prevent overflow
                    clamped_height = min(current_bar_height, 1.0)
                    height = int(clamped_height * max_bar_height)
                    marker_y = center_y - height - 8
                    
                    # Draw larger, more visible formant marker
                    pygame.draw.circle(self.screen, color, (x + 3, marker_y), 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + 3, marker_y), 6, 2)
                    # Add a small inner dot for better visibility
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + 3, marker_y), 2)

    def draw_technical_overlay(self):
        """Draw technical analysis overlay"""
        if not self.show_technical_overlay:
            return

        # Calculate position - stack below harmonic analysis on the right
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 250  # Match harmonic panel width
        overlay_height = 280
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
        # Position below pitch detection (if shown) or harmonic analysis
        # Header (280) + gap (10) + harmonic (200) + gap (10) + pitch (320 if shown) + gap (10)
        if self.show_pitch_detection:
            overlay_y = 280 + 10 + 200 + 10 + 320 + 10  # Updated for new pitch panel height
        else:
            overlay_y = 280 + 10 + 200 + 10

        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (overlay_x, overlay_y))

        # Border
        pygame.draw.rect(
            self.screen, (80, 90, 110), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )

        y_offset = overlay_y + 20
        line_height = 25

        # Title
        title_text = self.font_medium.render("Technical Analysis", True, (200, 200, 220))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 40

        # Tonal balance analysis
        bass_energy = self.calculate_band_energy(20, 250)
        mid_energy = self.calculate_band_energy(250, 2000)
        high_energy = self.calculate_band_energy(2000, 20000)

        self.draw_overlay_text(f"Bass (20-250Hz): {bass_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height
        self.draw_overlay_text(f"Mids (250-2kHz): {mid_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height
        self.draw_overlay_text(f"Highs (2k-20kHz): {high_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height * 1.5

        # Spectral tilt
        tilt = self.calculate_spectral_tilt()
        tilt_description = "Bright" if tilt > 0 else "Dark" if tilt < -3 else "Balanced"
        self.draw_overlay_text(
            f"Spectral Tilt: {tilt:.1f}dB/oct ({tilt_description})", overlay_x + 20, y_offset
        )
        y_offset += line_height * 1.5

        # Crest factor (dynamics)
        crest_factor = self.calculate_crest_factor()
        self.draw_overlay_text(f"Crest Factor: {crest_factor:.1f}dB", overlay_x + 20, y_offset)
        y_offset += line_height

        # Dynamic range
        dynamic_range = max(self.bar_heights) - np.mean(self.bar_heights[self.bar_heights > 0.1])
        self.draw_overlay_text(f"Dynamic Range: {dynamic_range:.1f}dB", overlay_x + 20, y_offset)
        y_offset += line_height * 1.5

        # Room mode summary
        if len(self.room_mode_candidates) > 0:
            self.draw_overlay_text(
                f"Room Modes Detected: {len(self.room_mode_candidates)}",
                overlay_x + 20,
                y_offset,
                (255, 150, 150),
            )
            y_offset += line_height
            for freq, _, duration, _ in self.room_mode_candidates[:3]:  # Show first 3
                self.draw_overlay_text(
                    f"  {freq:.1f}Hz ({duration:.1f}s)", overlay_x + 40, y_offset, (200, 120, 120)
                )
                y_offset += line_height

    def draw_overlay_text(self, text: str, x: int, y: int, color=(180, 180, 200)):
        """Helper to draw text in overlay"""
        text_surface = self.font_small.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def draw_voice_info(self):
        """Draw voice detection information overlay"""
        if not hasattr(self, "voice_info") or not self.voice_info:
            return
            
        # Calculate position - stack below technical overlay on the right
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 250  # Match harmonic panel width
        overlay_height = 320  # Slightly taller for gender detection info
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
        # Position below pitch detection (if shown) or technical overlay
        # Header (280) + gap (10) + harmonic (200) + gap (10) + pitch (320 if shown) + gap (10) + tech (280) + gap (10)
        if self.show_pitch_detection:
            overlay_y = 280 + 10 + 200 + 10 + 320 + 10 + 280 + 10  # Updated for new pitch panel height
        else:
            overlay_y = 280 + 10 + 200 + 10 + 280 + 10
        
        # Semi-transparent background
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        # Border
        pygame.draw.rect(
            self.screen, (80, 90, 110), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )
        
        y_offset = overlay_y + 20
        line_height = 25
        
        # Title
        title_text = self.font_medium.render("Voice Detection", True, (200, 200, 220))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 40
        
        # Voice detection status
        has_voice = self.voice_info.get("has_voice", False)
        pitch = self.voice_info.get("pitch", 0)
        
        # Calculate confidence based on voice energy in speech frequencies
        # Check energy in typical speech frequency range (80Hz - 4kHz)
        speech_energy = 0.0
        total_energy = 0.0
        voice_band_count = 0
        
        for i in range(min(self.bars, len(self.bar_heights))):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2
            
            if 80 <= center_freq <= 4000:
                speech_energy += self.bar_heights[i]
                voice_band_count += 1
            total_energy += self.bar_heights[i]
        
        # Calculate confidence based on speech energy ratio and presence of pitch
        if voice_band_count > 0 and total_energy > 0:
            speech_ratio = speech_energy / total_energy
            pitch_confidence = 1.0 if pitch > 80 else 0.0
            # Weighted confidence: 60% from speech energy ratio, 40% from pitch detection
            confidence = (speech_ratio * 0.6 + pitch_confidence * 0.4)
            # Apply threshold - if has_voice is True but confidence is low, set minimum
            if has_voice and confidence < 0.3:
                confidence = 0.3
        else:
            confidence = 0.0
        
        # Update voice_info with calculated confidence
        self.voice_info["confidence"] = confidence
        
        # Status with color coding
        if has_voice:
            status_color = (100, 255, 100)
            status_text = "VOICE DETECTED"
        else:
            status_color = (150, 150, 150)
            status_text = "No Voice"
            
        status_surf = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surf, (overlay_x + 20, y_offset))
        y_offset += line_height
        
        # Confidence level
        conf_text = f"Confidence: {confidence:.1%}"
        conf_color = (
            (255, 100, 100) if confidence < 0.3 else
            (255, 200, 100) if confidence < 0.7 else
            (100, 255, 100)
        )
        conf_surf = self.font_small.render(conf_text, True, conf_color)
        self.screen.blit(conf_surf, (overlay_x + 20, y_offset))
        y_offset += line_height
        
        # Pitch if detected
        if has_voice:
            pitch = self.voice_info.get("pitch", 0)
            if pitch > 0:
                pitch_text = f"Pitch: {pitch:.1f} Hz"
                note = self.freq_to_note(pitch)
                if note:
                    pitch_text += f" ({note})"
                pitch_surf = self.font_small.render(pitch_text, True, (200, 220, 255))
                self.screen.blit(pitch_surf, (overlay_x + 20, y_offset))
                y_offset += line_height
                
                # Gender detection based on pitch AND spectral characteristics
                y_offset += 5  # Small spacing
                if pitch > 0:
                    # Improved pitch ranges based on research:
                    # Male speaking: 85-180 Hz (average ~125 Hz)
                    # Female speaking: 165-255 Hz (average ~210 Hz)
                    # Male singing can go higher, female singing can go lower
                    
                    # Also check spectral tilt and formant ratios for better accuracy
                    # Calculate energy ratio between low and high frequencies
                    low_energy = 0.0  # 80-250 Hz
                    high_energy = 0.0  # 1000-4000 Hz
                    
                    for i in range(min(self.bars, len(self.bar_heights))):
                        freq_start, freq_end = self.get_frequency_range_for_bar(i)
                        center_freq = (freq_start + freq_end) / 2
                        
                        if 80 <= center_freq <= 250:
                            low_energy += self.bar_heights[i]
                        elif 1000 <= center_freq <= 4000:
                            high_energy += self.bar_heights[i]
                    
                    # Spectral tilt: male voices have more low frequency energy
                    spectral_tilt = low_energy / (high_energy + 0.001)
                    
                    # Adjust thresholds for better male voice detection
                    # Ivan Moody's voice is around 100-140 Hz typically
                    if pitch < 160:  # Raised threshold from 150
                        gender = "Male"
                        # Factor in spectral tilt - male voices have more low frequency energy
                        tilt_factor = min(1.0, spectral_tilt / 2.0)  # Males typically have tilt > 2
                        gender_confidence = min(1.0, (160 - pitch) / 75 * 0.7 + tilt_factor * 0.3)
                        gender_color = (100, 150, 255)  # Blue
                    elif pitch > 190:  # Lowered threshold from 200
                        gender = "Female"
                        # Females have less spectral tilt
                        tilt_factor = min(1.0, (2.0 - spectral_tilt) / 2.0)
                        gender_confidence = min(1.0, (pitch - 190) / 65 * 0.7 + tilt_factor * 0.3)
                        gender_color = (255, 150, 200)  # Pink
                    else:
                        # Ambiguous range (160-190 Hz) - use spectral tilt as tiebreaker
                        if spectral_tilt > 1.5:  # More low frequency energy suggests male
                            gender = "Likely Male"
                            gender_confidence = 0.5 + (spectral_tilt - 1.5) * 0.3
                            gender_color = (150, 175, 255)
                        else:
                            gender = "Likely Female"
                            gender_confidence = 0.5 + (1.5 - spectral_tilt) * 0.3
                            gender_color = (255, 175, 225)
                    
                    gender_text = f"Gender: {gender} ({gender_confidence:.0%})"
                    gender_surf = self.font_small.render(gender_text, True, gender_color)
                    self.screen.blit(gender_surf, (overlay_x + 20, y_offset))
                    y_offset += line_height
                    
                    # Debug: show spectral tilt
                    if self.show_advanced_info:
                        tilt_text = f"Spectral tilt: {spectral_tilt:.2f}"
                        tilt_surf = self.font_tiny.render(tilt_text, True, (150, 150, 170))
                        self.screen.blit(tilt_surf, (overlay_x + 40, y_offset))
                        y_offset += 20
        
        # Voice type classification
        classification = self.voice_info.get("classification", {})
        if classification:
            y_offset += 10
            class_text = self.font_small.render("Classification:", True, (180, 180, 200))
            self.screen.blit(class_text, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            for voice_type, prob in sorted(classification.items(), key=lambda x: x[1], reverse=True):
                if prob > 0.1:  # Only show significant probabilities
                    type_text = f"  {voice_type}: {prob:.1%}"
                    type_surf = self.font_small.render(type_text, True, (150, 170, 190))
                    self.screen.blit(type_surf, (overlay_x + 20, y_offset))
                    y_offset += line_height
        
        # Additional voice characteristics
        characteristics = self.voice_info.get("characteristics", {})
        if characteristics:
            y_offset += 10
            char_text = self.font_small.render("Characteristics:", True, (180, 180, 200))
            self.screen.blit(char_text, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            clarity = characteristics.get("clarity", 0)
            if clarity > 0:
                clarity_text = f"  Clarity: {clarity:.1%}"
                clarity_surf = self.font_small.render(clarity_text, True, (150, 170, 190))
                self.screen.blit(clarity_surf, (overlay_x + 20, y_offset))
    
    def draw_pitch_detection_info(self):
        """OMEGA: Draw advanced pitch detection information overlay"""
        if not hasattr(self, "pitch_info") or not self.pitch_info:
            return
            
        # Calculate position - below voice info or in its place if voice info is hidden
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 290  # Increased by 10px for better layout
        overlay_height = 320  # Increased by 20px for fixed graph position
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
        
        # Position below harmonic analysis panel
        # Header (280) + gap (10) + harmonic panel (200) + gap (10)
        overlay_y = 280 + 10 + 200 + 10
        
        # Semi-transparent background
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((25, 20, 35))  # Slightly purple tint
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        # Border
        pygame.draw.rect(
            self.screen, (110, 90, 140), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )
        
        y_offset = overlay_y + 20
        line_height = 22
        
        # Title
        title_text = self.font_medium.render("OMEGA Pitch Detection", True, (220, 200, 255))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 35
        
        # Main pitch info
        pitch = self.pitch_info.get('pitch', 0.0)
        confidence = self.pitch_info.get('confidence', 0.0)
        note = self.pitch_info.get('note', '')
        octave = self.pitch_info.get('octave', 0)
        cents = self.pitch_info.get('cents_offset', 0)
        stability = self.pitch_info.get('stability', 0.0)
        
        # Pitch detection status
        if pitch > 0 and confidence > 0.3:
            # Show detected pitch
            pitch_color = (
                (255, 150, 100) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            
            # Frequency
            freq_text = f"Frequency: {pitch:.1f} Hz"
            freq_surf = self.font_small.render(freq_text, True, pitch_color)
            self.screen.blit(freq_surf, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            # Musical note with cents
            if note:
                note_text = f"Note: {note}{octave}"
                if cents != 0:
                    note_text += f" {cents:+d}¢"
                note_surf = self.font_medium.render(note_text, True, (200, 220, 255))
                self.screen.blit(note_surf, (overlay_x + 20, y_offset))
                y_offset += line_height + 5
            
            # Confidence bar
            conf_text = f"Confidence: {confidence:.0%}"
            conf_surf = self.font_small.render(conf_text, True, (180, 180, 200))
            self.screen.blit(conf_surf, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            # Draw confidence bar
            bar_width = overlay_width - 40
            bar_height = 8
            bar_x = overlay_x + 20
            bar_y = y_offset
            
            # Background
            pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height))
            # Confidence fill
            fill_width = int(bar_width * confidence)
            conf_color = (
                (200, 100, 100) if confidence < 0.5 else
                (200, 200, 100) if confidence < 0.7 else
                (100, 200, 100)
            )
            pygame.draw.rect(self.screen, conf_color, (bar_x, bar_y, fill_width, bar_height))
            # Border
            pygame.draw.rect(self.screen, (100, 100, 120), (bar_x, bar_y, bar_width, bar_height), 1)
            y_offset += bar_height + 10
            
            # Stability indicator
            stab_text = f"Stability: {stability:.0%}"
            stab_color = (
                (255, 150, 150) if stability < 0.3 else
                (255, 255, 150) if stability < 0.7 else
                (150, 255, 150)
            )
            stab_surf = self.font_small.render(stab_text, True, stab_color)
            self.screen.blit(stab_surf, (overlay_x + 20, y_offset))
            y_offset += line_height + 10
            
        else:
            # No pitch detected
            no_pitch_text = "No pitch detected"
            no_pitch_surf = self.font_small.render(no_pitch_text, True, (150, 150, 150))
            self.screen.blit(no_pitch_surf, (overlay_x + 20, y_offset))
            y_offset += line_height + 10
        
        # Method breakdown
        methods = self.pitch_info.get('methods', {})
        if methods:
            # Separator line
            pygame.draw.line(
                self.screen, (80, 80, 100), 
                (overlay_x + 20, y_offset), 
                (overlay_x + overlay_width - 20, y_offset), 1
            )
            y_offset += 10
            
            # Methods header
            methods_text = self.font_small.render("Detection Methods:", True, (180, 180, 200))
            self.screen.blit(methods_text, (overlay_x + 20, y_offset))
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
                    
                    method_surf = self.font_tiny.render(method_text, True, method_color)
                    self.screen.blit(method_surf, (overlay_x + 20, y_offset))
                    y_offset += 18
        
        # Pitch history visualization (mini graph) - anchored to bottom
        if hasattr(self.cepstral_analyzer, 'pitch_history') and len(self.cepstral_analyzer.pitch_history) > 10:
            graph_height = 40
            graph_width = overlay_width - 40
            graph_x = overlay_x + 20
            # Anchor to bottom with consistent spacing
            graph_y = overlay_y + overlay_height - graph_height - 20  # 20px from bottom
            
            # Background
            pygame.draw.rect(self.screen, (30, 30, 40), (graph_x, graph_y, graph_width, graph_height))
            
            # Draw pitch history
            history = list(self.cepstral_analyzer.pitch_history)
            if history:
                # Find min/max for scaling
                valid_pitches = [p for p in history if p > 0]
                if valid_pitches:
                    min_pitch = min(valid_pitches) * 0.9
                    max_pitch = max(valid_pitches) * 1.1
                    
                    # Draw graph
                    points = []
                    for i, pitch in enumerate(history[-50:]):  # Last 50 values
                        if pitch > 0:
                            x = graph_x + int(i * graph_width / 50)
                            y = graph_y + graph_height - int((pitch - min_pitch) / (max_pitch - min_pitch) * graph_height)
                            points.append((x, y))
                    
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, (150, 150, 255), False, points, 2)
            
            # Border
            pygame.draw.rect(self.screen, (80, 80, 100), (graph_x, graph_y, graph_width, graph_height), 1)
            
            # Label
            hist_label = self.font_tiny.render("Pitch History", True, (150, 150, 170))
            self.screen.blit(hist_label, (graph_x, graph_y - 15))
                
    def freq_to_note(self, freq: float) -> str:
        """Convert frequency to musical note"""
        if freq <= 0:
            return None
            
        # A4 = 440 Hz
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        
        if freq < C0:
            return None
            
        # Calculate semitones from C0
        h = 12 * np.log2(freq / C0)
        octave = int(h / 12)
        n = int(h % 12)
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        return f"{notes[n]}{octave}"
    
    def process_vu_meters(self, audio_data: np.ndarray, dt: float):
        """Process audio for VU meter display with proper ballistics"""
        if audio_data is None or len(audio_data) == 0:
            return
            
        # For mono input, duplicate to both channels
        left_channel = audio_data
        right_channel = audio_data
        
        # Add samples to VU buffers
        for sample in left_channel:
            self.vu_left_buffer.append(sample)
            self.vu_right_buffer.append(sample)
            
        # Calculate RMS over integration period
        left_rms = np.sqrt(np.mean(np.array(self.vu_left_buffer) ** 2))
        right_rms = np.sqrt(np.mean(np.array(self.vu_right_buffer) ** 2))
        
        # Convert to dB relative to reference
        self.vu_left_db = self._linear_to_db_vu(left_rms / self.vu_reference_level)
        self.vu_right_db = self._linear_to_db_vu(right_rms / self.vu_reference_level)
        
        # Apply damping for smooth needle movement
        self.vu_left_display += (self.vu_left_db - self.vu_left_display) * (1.0 - self.vu_damping)
        self.vu_right_display += (self.vu_right_db - self.vu_right_display) * (1.0 - self.vu_damping)
        
        # Update peaks
        if self.vu_left_db > self.vu_left_peak_db:
            self.vu_left_peak_db = self.vu_left_db
            self.vu_left_peak_time = 0.0
        else:
            self.vu_left_peak_time += dt
            if self.vu_left_peak_time > self.vu_peak_hold_time:
                self.vu_left_peak_db -= 20.0 * dt  # Decay at 20 dB/s
                
        if self.vu_right_db > self.vu_right_peak_db:
            self.vu_right_peak_db = self.vu_right_db
            self.vu_right_peak_time = 0.0
        else:
            self.vu_right_peak_time += dt
            if self.vu_right_peak_time > self.vu_peak_hold_time:
                self.vu_right_peak_db -= 20.0 * dt  # Decay at 20 dB/s
    
    def _linear_to_db_vu(self, linear: float) -> float:
        """Convert linear scale to dB for VU meter"""
        if linear <= 0:
            return -60.0
        return 20.0 * np.log10(linear)

    def calculate_band_energy(self, freq_min: float, freq_max: float) -> float:
        """Calculate relative energy in frequency band"""
        band_energy = 0.0
        total_energy = np.sum(self.bar_heights)

        if total_energy > 0:
            for i in range(self.bars):
                freq_start, freq_end = self.get_frequency_range_for_bar(i)
                if freq_start >= freq_min and freq_end <= freq_max:
                    band_energy += self.bar_heights[i]
            return band_energy / total_energy
        return 0.0

    def calculate_spectral_tilt(self) -> float:
        """Calculate spectral tilt in dB per octave"""
        # Compare energy in low vs high frequencies
        low_energy = self.calculate_band_energy(100, 400)
        high_energy = self.calculate_band_energy(2000, 8000)

        if low_energy > 0 and high_energy > 0:
            return 20 * np.log10(high_energy / low_energy) / 4.3  # ~4.3 octaves
        return 0.0

    def calculate_crest_factor(self) -> float:
        """Calculate crest factor (peak to RMS ratio)"""
        if len(self.bar_heights) > 0:
            peak = np.max(self.bar_heights)
            rms = np.sqrt(np.mean(self.bar_heights**2))
            if rms > 0:
                return 20 * np.log10(peak / rms)
        return 0.0
    
    def draw_chromagram_info(self):
        """OMEGA-1: Draw chromagram and musical key detection information"""
        if not hasattr(self, "chromagram_info") or not self.chromagram_info:
            return
            
        # Calculate position - below pitch detection or harmonic analysis
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 290
        overlay_height = 280
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
        
        # Position below pitch detection if shown, otherwise below harmonic analysis
        if self.show_pitch_detection:
            # Header (280) + gap (10) + harmonic (200) + gap (10) + pitch (320) + gap (10)
            overlay_y = 280 + 10 + 200 + 10 + 320 + 10
        else:
            # Header (280) + gap (10) + harmonic panel (200) + gap (10)
            overlay_y = 280 + 10 + 200 + 10
        
        # Semi-transparent background
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((35, 25, 45))  # Purple tint for musical theme
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        # Border
        pygame.draw.rect(
            self.screen, (140, 100, 180), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )
        
        y_offset = overlay_y + 20
        
        # Title
        title_text = self.font_medium.render("OMEGA-1 Key Detection", True, (255, 200, 255))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 35
        
        # Get chromagram data
        chromagram = self.chromagram_info.get('chromagram', np.zeros(12))
        key = self.chromagram_info.get('key', 'C')
        is_major = self.chromagram_info.get('is_major', True)
        confidence = self.chromagram_info.get('confidence', 0.0)
        stability = self.chromagram_info.get('stability', 0.0)
        alt_keys = self.chromagram_info.get('alternative_keys', [])
        
        # Display detected key
        key_color = (
            (255, 150, 150) if confidence < 0.5 else
            (255, 255, 150) if confidence < 0.7 else
            (150, 255, 150)
        )
        key_text = f"Key: {key}"
        key_surf = self.font_large.render(key_text, True, key_color)
        self.screen.blit(key_surf, (overlay_x + 20, y_offset))
        y_offset += 35
        
        # Confidence and stability
        conf_text = f"Confidence: {confidence:.0%}"
        conf_surf = self.font_small.render(conf_text, True, (200, 200, 220))
        self.screen.blit(conf_surf, (overlay_x + 20, y_offset))
        y_offset += 20
        
        stab_text = f"Stability: {stability:.0%}"
        stab_color = (
            (255, 150, 150) if stability < 0.3 else
            (255, 255, 150) if stability < 0.7 else
            (150, 255, 150)
        )
        stab_surf = self.font_small.render(stab_text, True, stab_color)
        self.screen.blit(stab_surf, (overlay_x + 20, y_offset))
        y_offset += 25
        
        # Draw chromagram visualization
        chroma_x = overlay_x + 20
        chroma_y = y_offset
        chroma_width = overlay_width - 40
        chroma_height = 80
        
        # Background for chromagram
        pygame.draw.rect(self.screen, (20, 20, 30), 
                        (chroma_x, chroma_y, chroma_width, chroma_height))
        
        # Draw each chroma bin
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        bar_width = chroma_width // 12
        
        for i, (note, value) in enumerate(zip(note_names, chromagram)):
            bar_x = chroma_x + i * bar_width
            bar_height = int(chroma_height * min(value, 1.0))
            
            # Color based on note relationship to detected key
            if note == key.replace(' major', '').replace(' minor', ''):
                # Root note - bright color
                color = (255, 200, 100)
            elif note in key:
                # Part of key - medium brightness
                color = (200, 150, 100)
            else:
                # Not in key - dimmer
                color = (100, 100, 150)
            
            # Draw bar
            if bar_height > 0:
                pygame.draw.rect(self.screen, color,
                               (bar_x + 2, chroma_y + chroma_height - bar_height, 
                                bar_width - 4, bar_height))
            
            # Draw note labels
            label_color = (255, 255, 255) if note == key.replace(' major', '').replace(' minor', '') else (150, 150, 150)
            note_surf = self.font_tiny.render(note, True, label_color)
            note_rect = note_surf.get_rect(centerx=bar_x + bar_width // 2, 
                                          bottom=chroma_y + chroma_height + 15)
            self.screen.blit(note_surf, note_rect)
        
        # Border around chromagram
        pygame.draw.rect(self.screen, (100, 100, 120), 
                        (chroma_x, chroma_y, chroma_width, chroma_height), 1)
        
        y_offset = chroma_y + chroma_height + 25
        
        # Alternative keys if confidence is low
        if confidence < 0.7 and alt_keys:
            alt_text = "Alternative keys:"
            alt_surf = self.font_tiny.render(alt_text, True, (150, 150, 170))
            self.screen.blit(alt_surf, (overlay_x + 20, y_offset))
            y_offset += 15
            
            for alt_key, alt_conf in alt_keys[:2]:  # Show top 2 alternatives
                alt_key_text = f"  {alt_key}: {alt_conf:.0%}"
                alt_key_surf = self.font_tiny.render(alt_key_text, True, (130, 130, 150))
                self.screen.blit(alt_key_surf, (overlay_x + 20, y_offset))
                y_offset += 15

    def draw_ab_comparison(self):
        """Draw A/B comparison between current and reference spectrum"""
        if self.reference_spectrum is None:
            return

        # Draw reference spectrum as thin white lines
        spectrum_width = self.width - self.meter_panel_width - 40
        spectrum_height = self.height - 200
        spectrum_top = 80
        bar_width = spectrum_width / self.bars
        max_bar_height = (spectrum_height // 2) - 20
        center_y = spectrum_top + spectrum_height // 2

        for i, ref_value in enumerate(self.reference_spectrum):
            if ref_value > 0.01:
                x = 20 + i * bar_width
                ref_height = int(ref_value * max_bar_height)

                # Draw reference as thin white line (upper)
                pygame.draw.line(
                    self.screen,
                    (200, 200, 200, 150),
                    (int(x), center_y - ref_height),
                    (int(x + bar_width), center_y - ref_height),
                    2,
                )

                # Draw reference as thin white line (lower)
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150, 120),
                    (int(x), center_y),
                    (int(x + bar_width), center_y + ref_height),
                    2,
                )

        # Show difference information
        if len(self.bar_heights) == len(self.reference_spectrum):
            # Calculate overall difference in dB
            current_rms = np.sqrt(np.mean(self.bar_heights**2))
            ref_rms = np.sqrt(np.mean(self.reference_spectrum**2))

            if ref_rms > 0 and current_rms > 0:
                diff_db = 20 * np.log10(current_rms / ref_rms)
                diff_text = f"Δ: {diff_db:+.1f}dB"
                diff_color = (
                    (100, 255, 100)
                    if abs(diff_db) < 3
                    else (255, 200, 100) if abs(diff_db) < 6 else (255, 100, 100)
                )
                diff_surface = self.font_medium.render(diff_text, True, diff_color)
                self.screen.blit(diff_surface, (self.width - 200, 100))

                # Show A/B comparison label
                label_surface = self.font_small.render("A/B COMPARISON", True, (200, 200, 220))
                self.screen.blit(label_surface, (self.width - 200, 120))

    def print_debug_output(self):
        """Print detailed debug information to terminal"""
        print("\n" + "=" * 80)
        print(f"PROFESSIONAL AUDIO ANALYZER V4 - SPECTRUM SNAPSHOT - {self.bars} bars")
        print("Musical Perceptual Frequency Mapping")
        print("=" * 80)

        # Performance metrics
        current_fps = self.fps_counter[-1] if self.fps_counter else 0
        avg_fps = np.mean(self.fps_counter) if len(self.fps_counter) > 0 else 0

        # Calculate various latencies
        process_latency_ms = self.last_process_time * 1000
        audio_buffer_ms = (self.capture.chunk_size / SAMPLE_RATE) * 1000
        fft_window_ms = (FFT_SIZE_BASE / SAMPLE_RATE) * 1000
        total_latency_ms = audio_buffer_ms + process_latency_ms

        print(f"[PERFORMANCE METRICS]")
        print(f"FPS: Current={current_fps}, Average={avg_fps:.1f}, Target=60")
        print(f"Processing latency: {process_latency_ms:.1f}ms")
        print(f"Audio buffer: {self.capture.chunk_size} samples ({audio_buffer_ms:.1f}ms)")
        print(f"FFT window: {FFT_SIZE_BASE} samples ({fft_window_ms:.1f}ms)")
        print(f"Total latency: {total_latency_ms:.1f}ms")
        
        # Content type and adaptive allocation
        low_end_bars = int(self.bars * self.current_allocation)
        print(f"\n[PERCEPTUAL FREQUENCY MAPPING]")
        print(f"Mode: Musical Perceptual (Bark-scale inspired)")
        print(f"Distribution:")
        print(f"  Bass (20-250Hz): 25% of bars")
        print(f"  Low-mid (250-500Hz): 15% of bars") 
        print(f"  Mid (500-2kHz): 25% of bars")
        print(f"  High-mid (2k-6kHz): 20% of bars")
        print(f"  High (6k-20kHz): 15% of bars")

        # Performance status
        if avg_fps > 55:
            perf_status = "✅ Excellent"
        elif avg_fps > 45:
            perf_status = "⚠️  Good"
        elif avg_fps > 30:
            perf_status = "⚠️  Fair"
        else:
            perf_status = "❌ Poor"

        print(f"Performance: {perf_status}")
        print()

        # Create ASCII visualization of spectrum with proper frequency representation
        height = 16
        width = 80  # Terminal width

        # Create empty grid
        ascii_bars = []
        for _ in range(height):
            ascii_bars.append([" " for _ in range(width)])

        # Map terminal positions to frequency ranges more accurately
        for pos in range(width):
            # Calculate frequency position (0.0 to 1.0 across the spectrum)
            freq_position = pos / (width - 1)
            
            # Find the actual bar index that corresponds to this frequency position
            # Use logarithmic mapping similar to how the visual spectrum works
            target_freq = 20 * (20000 / 20) ** freq_position  # Log scale from 20Hz to 20kHz
            
            # Find the bar that covers this frequency
            best_bar_idx = 0
            best_distance = float('inf')
            
            for bar_idx in range(self.bars):
                freq_range = self.get_frequency_range_for_bar(bar_idx)
                if freq_range[0] > 0:  # Valid frequency range
                    bar_center = (freq_range[0] + freq_range[1]) / 2
                    distance = abs(np.log10(target_freq) - np.log10(bar_center))
                    if distance < best_distance:
                        best_distance = distance
                        best_bar_idx = bar_idx
            
            # Get the bar height for this frequency position
            bar_height = self.bar_heights[best_bar_idx] if best_bar_idx < len(self.bar_heights) else 0
            bar_pixels = int(bar_height * height)

            # Use different characters for different heights
            for j in range(bar_pixels):
                if j < height:
                    if j == 0:
                        ascii_bars[height - 1 - j][pos] = "▄"
                    elif j == bar_pixels - 1 and bar_pixels < height:
                        ascii_bars[height - 1 - j][pos] = "▄"
                    else:
                        ascii_bars[height - 1 - j][pos] = "█"

        # Print ASCII visualization
        for row in ascii_bars:
            print("".join(row))

        print()
        print("-" * 80)
        
        # Calculate actual frequency positions based on our perceptual mapping
        # With our distribution: 25% bass (20-250Hz), 15% low-mid (250-500Hz), 
        # 25% mid (500-2kHz), 20% high-mid (2k-6kHz), 15% high (6k-20kHz)
        
        # Key frequency markers at character positions
        # Position 0: 20Hz
        # Position 20 (25%): ~250Hz  
        # Position 32 (40%): ~500Hz
        # Position 52 (65%): ~2kHz
        # Position 68 (85%): ~6kHz
        # Position 80 (100%): 20kHz
        
        freq_line = [' '] * 80
        freq_line[0:3] = list("20 ")
        freq_line[19:23] = list("250 ")
        freq_line[31:35] = list("500 ")
        freq_line[50:53] = list("2k ")
        freq_line[66:69] = list("6k ")
        freq_line[76:80] = list("20k ")
        
        print("".join(freq_line) + "Hz")
        print()

        # Calculate statistics
        non_zero_bars = self.bar_heights[self.bar_heights > 0.01]
        if len(non_zero_bars) > 0:
            max_val = np.max(self.bar_heights)
            avg_val = np.mean(non_zero_bars)
            min_val = np.min(non_zero_bars)
            print(f"Stats: Max={max_val:.2f}, Avg={avg_val:.2f}, Min={min_val:.2f}")

        # Find top peaks with unique frequencies
        # Create a dictionary to store the highest amplitude for each frequency range
        freq_peaks = {}
        for idx in range(len(self.bar_heights)):
            if self.bar_heights[idx] > 0.1:
                freq_start, freq_end = self.get_frequency_range_for_bar(idx)
                center_freq = (freq_start + freq_end) / 2

                # Round frequency to nearest 10Hz to group similar frequencies
                freq_key = round(center_freq / 10) * 10

                if freq_key not in freq_peaks or self.bar_heights[idx] > freq_peaks[freq_key][1]:
                    freq_peaks[freq_key] = (center_freq, self.bar_heights[idx])

        # Sort by amplitude and get top 5 unique frequencies
        sorted_peaks = sorted(freq_peaks.values(), key=lambda x: x[1], reverse=True)[:5]

        if sorted_peaks:
            print(f"Top peaks: ", end="")
            print(", ".join([f"{freq:.0f}Hz({val:.2f})" for freq, val in sorted_peaks]))

        print()
        print("=" * 80)
        print(f"PROFESSIONAL AUDIO ANALYZER V4 - DEBUG - {time.strftime('%H:%M:%S')}")
        print("Musical Perceptual Frequency Mapping")
        print("=" * 80)
        print()

        # Frequency distribution analysis
        print("[FREQUENCY DISTRIBUTION]")
        bands = [
            ("Bass", 60, 250),
            ("Low-mid", 250, 500),
            ("Mid", 500, 2000),
            ("High-mid", 2000, 4000),
            ("Presence", 4000, 6000),
            ("Brilliance", 6000, 10000),
            ("Air", 10000, 20000),
        ]

        for band_name, low_freq, high_freq in bands:
            band_energy = self.calculate_band_energy(low_freq, high_freq)
            band_bars = []
            for i in range(self.bars):
                freq_start, freq_end = self.get_frequency_range_for_bar(i)
                if freq_start >= low_freq and freq_end <= high_freq:
                    band_bars.append(self.bar_heights[i])

            if band_bars:
                band_max = max(band_bars)
                band_avg = np.mean(band_bars)
                bar_char = (
                    "█"
                    if band_avg > 0.8
                    else (
                        "▅"
                        if band_avg > 0.5
                        else "▄" if band_avg > 0.3 else "▂" if band_avg > 0.1 else "▁"
                    )
                )
                print(
                    f"{band_name:12} [{low_freq:5d}-{high_freq:5d}Hz]: {bar_char} avg={band_avg:.2f} max={band_max:.2f}"
                )

        print()
        
        # OMEGA: Pitch detection debug output
        print("[OMEGA PITCH DETECTION]")
        if hasattr(self, 'pitch_info'):
            pitch = self.pitch_info.get('pitch', 0.0)
            confidence = self.pitch_info.get('confidence', 0.0)
            note = self.pitch_info.get('note', '')
            octave = self.pitch_info.get('octave', 0)
            cents = self.pitch_info.get('cents_offset', 0)
            stability = self.pitch_info.get('stability', 0.0)
            methods = self.pitch_info.get('methods', {})
            
            print(f"Detected Pitch: {pitch:.2f} Hz" if pitch > 0 else "No pitch detected")
            if pitch > 0 and note:
                print(f"Musical Note: {note}{octave} {cents:+d}¢")
            print(f"Confidence: {confidence:.1%}")
            print(f"Stability: {stability:.1%}")
            
            print("\nMethod Results:")
            for method, (m_pitch, m_conf) in methods.items():
                print(f"  {method.capitalize()}: {m_pitch:.1f} Hz ({m_conf:.1%})" if m_pitch > 0 else f"  {method.capitalize()}: No detection")
            
            # Debug the audio buffer
            if hasattr(self, 'ring_buffer'):
                audio_rms = np.sqrt(np.mean(self.ring_buffer**2))
                print(f"\nAudio Buffer RMS: {audio_rms:.6f}")
                print(f"Audio Buffer Max: {np.max(np.abs(self.ring_buffer)):.6f}")
                
            # Check if cepstral analyzer has valid data
            if hasattr(self, 'cepstral_analyzer'):
                print(f"YIN window size: {self.cepstral_analyzer.yin_window_size}")
                print(f"Pitch history length: {len(self.cepstral_analyzer.pitch_history)}")
        else:
            print("Pitch info not initialized")
            
        print()
        print("[GAP ANALYSIS]")
        # Check for gaps in low frequency bars
        gap_found = False
        for i in range(min(64, self.bars - 1)):  # Check first 64 bars
            freq_start1, freq_end1 = self.get_frequency_range_for_bar(i)
            freq_start2, freq_end2 = self.get_frequency_range_for_bar(i + 1)
            if freq_end1 < freq_start2 - 1:  # Gap found
                gap_found = True
                print(
                    f"Gap detected: Bar {i} ends at {freq_end1:.1f}Hz, Bar {i+1} starts at {freq_start2:.1f}Hz"
                )

        if not gap_found:
            print("No gaps detected in low frequency bars")
        
        # Professional Metering Information
        print()
        print("[PROFESSIONAL METERING - ITU-R BS.1770-4]")
        # Calculate LUFS on demand for debug if not available
        if not hasattr(self, 'lufs_info') or not self.lufs_info:
            # One-time calculation for debug output
            if hasattr(self, 'ring_buffer'):
                # Use recent audio from ring buffer
                audio_for_lufs = self.ring_buffer[-SAMPLE_RATE:] if len(self.ring_buffer) >= SAMPLE_RATE else self.ring_buffer
                self.lufs_info = self.professional_metering.calculate_lufs(audio_for_lufs)
                self.true_peak = self.professional_metering.calculate_true_peak(audio_for_lufs)
        
        if hasattr(self, 'lufs_info') and self.lufs_info:
            lufs = self.lufs_info
            print(f"Momentary LUFS (400ms): {lufs['momentary']:+6.1f} LUFS")
            print(f"Short-term LUFS (3s):   {lufs['short_term']:+6.1f} LUFS")
            print(f"Integrated LUFS:        {lufs['integrated']:+6.1f} LU")
            print(f"Loudness Range (LRA):   {lufs['range']:6.1f} LU")
            
            # Add status descriptions
            if lufs['momentary'] > -100:
                if lufs['momentary'] > -9:
                    status = "TOO LOUD (risk of distortion)"
                elif lufs['momentary'] > -14:
                    status = "Loud (streaming/YouTube target: -14 LUFS)"
                elif lufs['momentary'] > -23:
                    status = "Normal (broadcast target: -23 LUFS)"
                elif lufs['momentary'] > -35:
                    status = "Quiet"
                else:
                    status = "Very quiet"
                print(f"Status: {status}")
        else:
            print("LUFS data not available")
        
        if hasattr(self, 'true_peak') and self.true_peak is not None:
            print(f"\nTrue Peak (4x oversampled): {self.true_peak:+6.1f} dBTP")
            if self.true_peak > -0.1:
                print("  ⚠️  WARNING: Potential clipping!")
            elif self.true_peak > -3:
                print("  ⚠️  Hot signal - watch levels")
            elif self.true_peak > -6:
                print("  ✅ Good headroom")
            else:
                print("  ℹ️  Safe levels")
        
        # Transient Analysis
        if hasattr(self, 'transient_info') and self.transient_info:
            trans = self.transient_info
            print(f"\n[TRANSIENT ANALYSIS]")
            print(f"Attack Time:  {trans.get('attack_time', 0):6.1f} ms")
            print(f"Punch Factor: {trans.get('punch_factor', 0):6.2f}")
            if trans.get('attack_time', 100) < 10:
                print("  → Very fast attack (percussive)")
            elif trans.get('attack_time', 100) < 30:
                print("  → Fast attack")
            elif trans.get('attack_time', 100) < 100:
                print("  → Medium attack")
            else:
                print("  → Slow attack (pad/string-like)")
        
        # Voice Detection
        if hasattr(self, 'voice_info') and self.voice_info:
            voice = self.voice_info
            print(f"\n[VOICE DETECTION]")
            print(f"Has Voice: {'YES' if voice['has_voice'] else 'NO'}")
            if voice['has_voice'] or voice.get('voice_confidence', 0) > 0:
                print(f"Confidence: {voice.get('voice_confidence', 0)*100:.1f}%")
                print(f"Type: {voice.get('voice_type', 'unknown').upper()}")
                if voice.get('is_singing'):
                    print("Classification: SINGING")
                else:
                    print("Classification: SPEAKING")

        print()
        print("[DYNAMIC RANGE]")
        if len(non_zero_bars) > 0:
            dynamic_range_db = 20 * np.log10(
                np.max(self.bar_heights) / (np.min(non_zero_bars) + 1e-10)
            )
            print(f"Dynamic Range: {dynamic_range_db:.1f} dB")
            active_bars = len(non_zero_bars)
            print(f"Active bars: {active_bars}/{self.bars} ({active_bars/self.bars*100:.1f}%)")

        # Auto-tune recommendation
        if hasattr(self, "capture") and hasattr(self.capture, "chunk_size"):
            current_chunk = self.capture.chunk_size
            if dynamic_range_db < 20:
                print(f"Auto-tune: Consider increasing chunk size from {current_chunk}")
            elif dynamic_range_db > 40:
                print(f"Auto-tune: Consider decreasing chunk size from {current_chunk}")
            else:
                print(f"Auto-tune: Chunk size {current_chunk} is optimal")

        print()
        
        # Note: Analysis starts at 20Hz (no ultra sub-bass below 20Hz analyzed)

        # Bass detail visualization (20-200Hz)
        print("=" * 80)
        print("BASS DETAIL VISUALIZATION (20-200Hz)")
        print("=" * 80)

        # Create a detailed view of bass frequencies
        bass_width = 80
        bass_freq_min = 20
        bass_freq_max = 200

        # Create frequency markers
        freq_markers = [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200]
        marker_line = [" "] * bass_width
        label_line = [" "] * bass_width

        # Place frequency markers
        for freq in freq_markers:
            pos = int(
                (np.log10(freq) - np.log10(bass_freq_min))
                / (np.log10(bass_freq_max) - np.log10(bass_freq_min))
                * (bass_width - 1)
            )
            if 0 <= pos < bass_width:
                marker_line[pos] = "|"
                # Add frequency labels for key frequencies
                if freq in [20, 40, 60, 100, 150, 200]:
                    label_str = str(freq)
                    # Ensure label fits within display width
                    if pos + len(label_str) > bass_width:
                        # Shift label left to fit
                        pos = max(0, bass_width - len(label_str))
                    for i, char in enumerate(label_str):
                        if pos + i < bass_width:
                            label_line[pos + i] = char

        # Create bass visualization bars
        bass_bars = []
        for row in range(10):  # 10 rows for visualization
            bass_bars.append([" "] * bass_width)

        # Fill in bass frequency content
        for i in range(self.bars):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2

            # Only process bass frequencies
            if bass_freq_min <= center_freq <= bass_freq_max:
                # Calculate position on logarithmic scale
                pos = int(
                    (np.log10(center_freq) - np.log10(bass_freq_min))
                    / (np.log10(bass_freq_max) - np.log10(bass_freq_min))
                    * (bass_width - 1)
                )

                if 0 <= pos < bass_width:
                    # Get bar height (0-10 scale)
                    height = int(self.bar_heights[i] * 10)

                    # Draw vertical bar
                    for h in range(height):
                        if h < 10:
                            char = (
                                "█"
                                if self.bar_heights[i] > 0.8
                                else (
                                    "▓"
                                    if self.bar_heights[i] > 0.5
                                    else "▒" if self.bar_heights[i] > 0.2 else "░"
                                )
                            )
                            bass_bars[9 - h][pos] = char

        # Print bass visualization
        for row in bass_bars:
            print("".join(row))

        print("".join(marker_line))
        print("".join(label_line) + " Hz")

        # Bass statistics
        bass_bars_indices = []
        bass_values = []
        for i in range(self.bars):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2
            if bass_freq_min <= center_freq <= bass_freq_max:
                bass_bars_indices.append(i)
                bass_values.append(self.bar_heights[i])

        if bass_values:
            print()
            print(
                f"Bass stats: {len(bass_bars_indices)} bars, "
                + f"Max={max(bass_values):.2f}, "
                + f"Avg={np.mean(bass_values):.2f}, "
                + f"Energy={sum(bass_values)/len(self.bar_heights):.1%}"
            )

            # Find peak bass frequency
            peak_idx = bass_bars_indices[np.argmax(bass_values)]
            freq_start, freq_end = self.get_frequency_range_for_bar(peak_idx)
            peak_freq = (freq_start + freq_end) / 2
            print(f"Peak bass frequency: {peak_freq:.1f}Hz")

        print()
        
        # VU Meter visualization
        if self.show_vu_meters:
            print("=" * 80)
            print("VU METERS (0 VU = -20 dBFS)")
            print("=" * 80)
            
            # Get current VU levels
            left_db = self.vu_left_display if hasattr(self, 'vu_left_display') else -20.0
            right_db = self.vu_right_display if hasattr(self, 'vu_right_display') else -20.0
            left_peak = self.vu_left_peak_db if hasattr(self, 'vu_left_peak_db') else -20.0
            right_peak = self.vu_right_peak_db if hasattr(self, 'vu_right_peak_db') else -20.0
            
            # VU meter scale from -20 to +3 dB
            meter_width = 60
            scale_min = -20.0
            scale_max = 3.0
            
            # Create scale markers
            scale_line = [' '] * meter_width
            label_line = [' '] * meter_width
            
            # Place scale markers
            scale_positions = [(-20, "-20"), (-10, "-10"), (-7, "-7"), (-5, "-5"), 
                             (-3, "-3"), (0, "0"), (+3, "+3")]
            
            for db_val, label in scale_positions:
                pos = int((db_val - scale_min) / (scale_max - scale_min) * (meter_width - 1))
                if 0 <= pos < meter_width:
                    scale_line[pos] = '|'
                    # Add labels for key positions
                    if len(label) <= 3:
                        for i, char in enumerate(label):
                            if pos + i < meter_width:
                                label_line[pos + i] = char
            
            # Draw meters
            def draw_vu_meter(name, value, peak):
                # Normalize values to 0-1 range
                value_norm = (value - scale_min) / (scale_max - scale_min)
                peak_norm = (peak - scale_min) / (scale_max - scale_min)
                
                # Clamp to valid range
                value_norm = max(0.0, min(1.0, value_norm))
                peak_norm = max(0.0, min(1.0, peak_norm))
                
                # Calculate positions
                value_pos = int(value_norm * (meter_width - 1))
                peak_pos = int(peak_norm * (meter_width - 1))
                
                # Create meter visualization
                meter = [' '] * meter_width
                
                # Fill meter up to current value
                for i in range(value_pos + 1):
                    if value > 0 and i >= int((0 - scale_min) / (scale_max - scale_min) * (meter_width - 1)):
                        meter[i] = '█'  # Red zone
                    elif value > -3 and i >= int((-3 - scale_min) / (scale_max - scale_min) * (meter_width - 1)):
                        meter[i] = '▓'  # Yellow zone
                    else:
                        meter[i] = '▒'  # Green zone
                
                # Add peak indicator
                if 0 <= peak_pos < meter_width:
                    meter[peak_pos] = '┃'
                
                # Print meter
                print(f"{name:5} [{(''.join(meter)).ljust(meter_width)}] {value:+5.1f} dB (peak: {peak:+5.1f} dB)")
            
            # Draw scale
            print("      " + "".join(scale_line))
            print("      " + "".join(label_line) + " dB")
            print()
            
            # Draw left and right meters
            draw_vu_meter("LEFT", left_db, left_peak)
            draw_vu_meter("RIGHT", right_db, right_peak)
            
            # VU meter status
            if left_db > 0 or right_db > 0:
                print("\n⚠️  WARNING: Signal exceeding 0 VU!")
            elif left_db > -3 or right_db > -3:
                print("\n⚠️  Hot signal - watch levels")
            else:
                print("\n✅ Signal levels OK")
        
        print()

    def draw_professional_meters(self, x: int, y: int, width: int, height: int):
        """Enhanced professional meters panel with better readability"""
        if not hasattr(self, "lufs_info"):
            return

        # Semi-transparent background for better integration
        panel_bg = pygame.Surface((width, height))
        panel_bg.set_alpha(240)
        panel_bg.fill((25, 30, 40))
        self.screen.blit(panel_bg, (x, y))

        # Border with professional appearance
        pygame.draw.rect(self.screen, (80, 90, 110), (x, y, width, height), 2)

        # Title with larger, more readable font
        title_y = y + int(15 * self.ui_scale)
        title = self.font_medium.render("PROFESSIONAL METERS", True, (180, 190, 200))
        self.screen.blit(title, (x + int(10 * self.ui_scale), title_y))

        # Meters with consistent spacing and better layout
        current_y = title_y + int(40 * self.ui_scale)
        spacing = int(30 * self.ui_scale)

        # LUFS Meters with consistent formatting
        lufs = self.lufs_info

        meters = [
            ("M:", f"{lufs['momentary']:+5.1f} LUFS", self.get_lufs_color(lufs["momentary"])),
            ("S:", f"{lufs['short_term']:+5.1f} LUFS", self.get_lufs_color(lufs["short_term"])),
            ("I:", f"{lufs['integrated']:+5.1f} LU", self.get_lufs_color(lufs["integrated"])),
            ("LRA:", f"{lufs['range']:4.1f} LU", (150, 200, 150)),
            (
                "TP:",
                f"{lufs.get('true_peak', 0):+5.1f} dBTP",
                self.get_peak_color(lufs.get("true_peak", 0)),
            ),
        ]

        for label, value, color in meters:
            # Label with consistent positioning
            label_surf = self.font_small.render(label, True, (140, 150, 160))
            self.screen.blit(label_surf, (x + int(15 * self.ui_scale), current_y))

            # Value with color coding and proper spacing
            value_surf = self.font_meters.render(value, True, color)
            self.screen.blit(value_surf, (x + int(50 * self.ui_scale), current_y))

            current_y += spacing

        # Add spacing before additional analysis
        current_y += int(20 * self.ui_scale)

        # Transient Analysis with improved formatting
        if hasattr(self, "transient_info"):
            trans = self.transient_info
            attack_text = f"Attack: {trans.get('attack_time', 0):.1f}ms"
            punch_text = f"Punch: {trans.get('punch_factor', 0):.2f}"

            attack_surf = self.font_small.render(attack_text, True, (200, 220, 180))
            punch_surf = self.font_small.render(punch_text, True, (220, 200, 180))

            self.screen.blit(attack_surf, (x + 10, current_y))
            current_y += 20
            self.screen.blit(punch_surf, (x + 10, current_y))

    def get_lufs_color(self, lufs_value: float) -> Tuple[int, int, int]:
        """Get color for LUFS value based on broadcast standards"""
        if lufs_value > -9:  # Too loud
            return (255, 100, 100)
        elif lufs_value > -14:  # Loud
            return (255, 200, 100)
        elif lufs_value > -23:  # Normal range
            return (100, 255, 100)
        elif lufs_value > -35:  # Quiet
            return (180, 180, 255)
        else:  # Very quiet
            return (120, 120, 120)

    def get_peak_color(self, peak_db: float) -> Tuple[int, int, int]:
        """Get color for True Peak value"""
        if peak_db > -0.1:  # Clipping risk
            return (255, 50, 50)
        elif peak_db > -3:  # Hot
            return (255, 150, 50)
        elif peak_db > -6:  # Good level
            return (100, 255, 100)
        elif peak_db > -20:  # Normal
            return (180, 200, 220)
        else:  # Low
            return (120, 120, 120)

    def draw_harmonic_analysis(self, x: int, y: int, width: int, height: int):
        """Draw harmonic analysis panel"""
        if not hasattr(self, "harmonic_info"):
            return

        # Background
        analysis_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (25, 20, 35), analysis_rect)
        pygame.draw.rect(self.screen, (70, 60, 90), analysis_rect, 2)

        current_y = y + 10

        # Title
        title = self.font_medium.render("HARMONIC ANALYSIS", True, (255, 255, 255))
        self.screen.blit(title, (x + 10, current_y))
        current_y += 35

        harmonic_info = self.harmonic_info

        # Dominant fundamental
        if harmonic_info["dominant_fundamental"] > 0:
            fund_text = f"Fund: {harmonic_info['dominant_fundamental']:.1f} Hz"
            fund_surf = self.font_small.render(fund_text, True, (255, 220, 120))
            self.screen.blit(fund_surf, (x + 10, current_y))
            current_y += 25

        # Instrument matches
        instruments = harmonic_info.get("instrument_matches", [])
        for i, match in enumerate(instruments[:3]):  # Top 3 matches
            instrument = match["instrument"].replace("_", " ").title()
            confidence = match["confidence"]

            color_intensity = int(100 + confidence * 155)
            color = (color_intensity, color_intensity, 255)

            match_text = f"{instrument}: {confidence:.1%}"
            match_surf = self.font_small.render(match_text, True, color)
            self.screen.blit(match_surf, (x + 10, current_y))
            current_y += 20

    def draw_vu_meters(self, x: int, y: int, width: int, height: int):
        """Draw professional VU meters on the right side"""
        if not self.show_vu_meters:
            return
            
        # Background panel
        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (20, 25, 35), panel_rect)
        pygame.draw.rect(self.screen, (60, 70, 90), panel_rect, 2)
        
        # Title
        title_text = self.font_medium.render("VU METERS", True, (200, 220, 240))
        title_rect = title_text.get_rect(centerx=x + width // 2, top=y + 10)
        self.screen.blit(title_text, title_rect)
        
        # VU meter parameters - align with -50dB level on main spectrum
        # Calculate where -50dB is on the main spectrum (50/60 = 0.833 from center)
        # This gives us more room for title and labels
        meter_y_start = y + 60  # More space for title
        # Calculate meter height to align with -50dB position
        # The spectrum uses normalized_pos = 50/60 = 0.833 of half height from center
        # So we want our meters to end at approximately that position
        spectrum_height_ratio = 0.833  # -50dB position ratio
        meter_height = int((height - 160) * spectrum_height_ratio)  # Adjusted for more label space
        meter_width = 40
        meter_spacing = 20
        
        # Calculate meter positions
        total_meter_width = 2 * meter_width + meter_spacing
        meters_x_start = x + (width - total_meter_width) // 2
        
        # Scale parameters
        scale_min = -20.0  # dB
        scale_max = 3.0    # dB
        
        # Draw left meter
        left_x = meters_x_start
        self._draw_single_vu_meter(left_x, meter_y_start, meter_width, meter_height,
                                   self.vu_left_display, self.vu_left_peak_db,
                                   scale_min, scale_max, "L")
        
        # Draw right meter
        right_x = meters_x_start + meter_width + meter_spacing
        self._draw_single_vu_meter(right_x, meter_y_start, meter_width, meter_height,
                                   self.vu_right_display, self.vu_right_peak_db,
                                   scale_min, scale_max, "R")
        
        # Draw reference info
        ref_text = self.font_tiny.render("0 VU = -20 dBFS", True, (150, 170, 190))
        ref_rect = ref_text.get_rect(centerx=x + width // 2, bottom=y + height - 10)
        self.screen.blit(ref_text, ref_rect)
        
        # Draw current values
        values_text = self.font_small.render(
            f"L: {self.vu_left_display:+5.1f} dB  R: {self.vu_right_display:+5.1f} dB",
            True, (180, 200, 220)
        )
        values_rect = values_text.get_rect(centerx=x + width // 2, bottom=y + height - 30)
        self.screen.blit(values_text, values_rect)
    
    def _draw_single_vu_meter(self, x: int, y: int, width: int, height: int,
                              value: float, peak: float, scale_min: float, scale_max: float,
                              label: str):
        """Draw a single VU meter"""
        # Meter background
        meter_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (15, 20, 30), meter_rect)
        pygame.draw.rect(self.screen, (50, 60, 80), meter_rect, 1)
        
        # Draw scale marks
        scale_positions = [
            (-20, "−20", True),
            (-10, "−10", True),
            (-7, "−7", True),
            (-5, "−5", True),
            (-3, "−3", True),
            (0, "0", True),
            (3, "+3", True)
        ]
        
        for db, db_label, is_major in scale_positions:
            # Calculate Y position
            normalized = (db - scale_min) / (scale_max - scale_min)
            mark_y = int(y + height * (1.0 - normalized))
            
            # Draw tick mark
            if is_major:
                tick_length = 8
                tick_color = (180, 180, 200)
                # Draw label
                if db in [-20, -10, 0, 3]:
                    label_surf = self.font_tiny.render(db_label, True, tick_color)
                    label_rect = label_surf.get_rect(right=x - 5, centery=mark_y)
                    if label_rect.top >= y and label_rect.bottom <= y + height:
                        self.screen.blit(label_surf, label_rect)
            else:
                tick_length = 4
                tick_color = (100, 100, 120)
                
            pygame.draw.line(self.screen, tick_color, (x, mark_y), (x + tick_length, mark_y), 1)
        
        # Calculate needle position
        value_normalized = (value - scale_min) / (scale_max - scale_min)
        value_normalized = max(0.0, min(1.0, value_normalized))
        value_y = int(y + height * (1.0 - value_normalized))
        
        # Calculate peak position
        peak_normalized = (peak - scale_min) / (scale_max - scale_min)
        peak_normalized = max(0.0, min(1.0, peak_normalized))
        peak_y = int(y + height * (1.0 - peak_normalized))
        
        # Draw the meter bar with gradient effect
        bar_x = x + 10
        bar_width = width - 20
        
        # Fill up to current value with color coding
        if value_y < y + height:
            fill_height = y + height - value_y
            fill_rect = pygame.Rect(bar_x, value_y, bar_width, fill_height)
            
            # Color based on level
            if value > 0:
                # Red zone
                color = (200, 50, 40)
            elif value > -3:
                # Yellow zone
                color = (200, 180, 40)
            else:
                # Green zone
                color = (40, 180, 60)
                
            pygame.draw.rect(self.screen, color, fill_rect)
            
            # Add gradient effect
            for i in range(3):
                grad_color = tuple(int(c * (1 - i * 0.2)) for c in color)
                grad_rect = pygame.Rect(bar_x + i, value_y + i, bar_width - 2*i, fill_height - 2*i)
                pygame.draw.rect(self.screen, grad_color, grad_rect, 1)
        
        # Draw needle pointer
        needle_color = (255, 255, 255)
        pygame.draw.line(self.screen, needle_color, (x, value_y), (x + width, value_y), 2)
        
        # Draw peak indicator
        if peak > scale_min:
            peak_color = (255, 100, 100)
            pygame.draw.line(self.screen, peak_color, (bar_x, peak_y), (bar_x + bar_width, peak_y), 1)
        
        # Draw channel label below the meter
        label_surf = self.font_medium.render(label, True, (200, 220, 240))
        label_rect = label_surf.get_rect(centerx=x + width // 2, top=y + height + 5)
        self.screen.blit(label_surf, label_rect)
    
    def draw_bass_zoom_window(self, x: int, y: int, width: int, height: int):
        """Draw detailed bass frequency zoom window (20-200 Hz) with peak hold"""
        # Background
        zoom_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (15, 25, 40), zoom_rect)
        pygame.draw.rect(self.screen, (60, 80, 120), zoom_rect, 2)

        # Title
        title = self.font_small.render("BASS DETAIL (20-200 Hz)", True, (200, 220, 255))
        self.screen.blit(title, (x + 5, y + 5))

        # Reserve space for scale at bottom (35px for scale + labels)
        scale_height = 35
        visualization_height = height - 40 - scale_height  # Title space + scale space

        # Use the enhanced 64-bar bass detail data
        if hasattr(self, 'bass_bar_values'):
            # Simple approach: make bars fill entire width with no gaps
            total_width = width - 20
            bar_width = total_width / self.bass_detail_bars  # Exact division
            max_height = visualization_height
            current_time = time.time()
            bar_bottom = y + height - scale_height - 5  # Leave space for scale

            for j in range(self.bass_detail_bars):
                # Clamp amplitude to prevent overflow
                amplitude = min(self.bass_bar_values[j], 1.0)
                freq_range = self.bass_freq_ranges[j]
                
                # Calculate exact bar position to eliminate gaps
                bar_height = min(int(amplitude * max_height), max_height)
                bar_x = x + 10 + j * bar_width
                bar_y = max(y + 25, bar_bottom - bar_height)
                
                # Smooth gradient color based on frequency position in bass range
                freq_center = (freq_range[0] + freq_range[1]) / 2
                
                # Calculate position in 20-200Hz range (0.0 to 1.0)
                bass_min = 20.0
                bass_max = 200.0
                freq_position = (freq_center - bass_min) / (bass_max - bass_min)
                freq_position = max(0.0, min(1.0, freq_position))  # Clamp to 0-1
                
                # Create smooth gradient: Purple -> Red -> Orange -> Yellow -> Green
                if freq_position < 0.2:  # 20-56Hz: Purple to Magenta
                    t = freq_position / 0.2
                    color = (
                        int(150 + 105 * t),  # 150->255 (purple to magenta)
                        int(50 + 50 * t),    # 50->100
                        int(255 - 55 * t)    # 255->200
                    )
                elif freq_position < 0.4:  # 56-92Hz: Magenta to Red  
                    t = (freq_position - 0.2) / 0.2
                    color = (
                        255,                  # Stay at 255 (red)
                        int(100 - 100 * t),  # 100->0 (lose green)
                        int(200 - 200 * t)   # 200->0 (lose blue)
                    )
                elif freq_position < 0.6:  # 92-128Hz: Red to Orange
                    t = (freq_position - 0.4) / 0.2
                    color = (
                        255,                 # Stay at 255 (red)
                        int(150 * t),       # 0->150 (add orange)
                        0                   # Stay at 0
                    )
                elif freq_position < 0.8:  # 128-164Hz: Orange to Yellow
                    t = (freq_position - 0.6) / 0.2
                    color = (
                        255,                 # Stay at 255
                        int(150 + 105 * t), # 150->255 (orange to yellow)
                        0                   # Stay at 0
                    )
                else:  # 164-200Hz: Yellow to Green
                    t = (freq_position - 0.8) / 0.2
                    color = (
                        int(255 - 55 * t),  # 255->200 (lose some red)
                        255,                # Stay at 255 (green)
                        int(100 * t)       # 0->100 (add some blue for lime)
                    )

                # Enhance brightness for kick detection
                if (
                    hasattr(self, "drum_info")
                    and self.drum_info["kick"].get("display_strength", 0) > 0.1
                    and freq_center <= 120
                ):
                    # Increase brightness by 30% for kick detection
                    color = tuple(min(255, int(c * 1.3)) for c in color)

                # Draw bar that exactly fills its space (round up width to ensure no gaps)
                actual_bar_width = int(bar_width) + 1 if j < self.bass_detail_bars - 1 else int(bar_width)
                pygame.draw.rect(
                    self.screen, color, (int(bar_x), bar_y, actual_bar_width, bar_height)
                )

                # Draw peak hold line
                if j < len(self.bass_peak_values) and self.bass_peak_values[j] > 0.05:
                    # Clamp peak value to prevent overflow
                    clamped_peak = min(self.bass_peak_values[j], 1.0)
                    peak_height = min(int(clamped_peak * max_height), max_height)
                    peak_y = max(y + 25, bar_bottom - peak_height)
                    pygame.draw.line(
                        self.screen,
                        (255, 255, 255, 180),
                        (int(bar_x), peak_y),
                        (int(bar_x + actual_bar_width - 1), peak_y),
                        1
                    )

            # Draw frequency scale line (now properly positioned)
            scale_y = y + height - scale_height
            pygame.draw.line(self.screen, (100, 100, 120), 
                           (x + 10, scale_y), 
                           (x + width - 10, scale_y), 2)
            
            # Draw background for scale area
            scale_bg_rect = pygame.Rect(x + 5, scale_y - 2, width - 10, scale_height - 3)
            pygame.draw.rect(self.screen, (10, 15, 25), scale_bg_rect)
            
            # Frequency labels with tick marks
            key_freqs = [20, 30, 40, 60, 80, 100, 150, 200]
            for target_freq in key_freqs:
                # Find position for this frequency on logarithmic scale
                log_pos = (np.log10(target_freq) - np.log10(20)) / (np.log10(200) - np.log10(20))
                if 0 <= log_pos <= 1:
                    label_x = x + 10 + log_pos * (width - 20)
                    
                    # Draw tick mark
                    pygame.draw.line(self.screen, (150, 150, 160),
                                       (int(label_x), scale_y - 2),
                                       (int(label_x), scale_y + 6), 2)
                    
                    # Draw label
                    freq_text = f"{target_freq}"
                    freq_surf = self.font_tiny.render(freq_text, True, (200, 220, 240))
                    text_rect = freq_surf.get_rect(centerx=int(label_x), top=scale_y + 8)
                    self.screen.blit(freq_surf, text_rect)
            
            # Add "Hz" label at the end
            hz_surf = self.font_tiny.render("Hz", True, (200, 220, 240))
            self.screen.blit(hz_surf, (x + width - 30, scale_y + 8))

    def draw_frame(self):
        """Draw the enhanced professional visualization"""
        # Clear screen with professional dark background
        self.screen.fill((8, 10, 15))

        # Calculate layout with professional panels - doubled header size
        header_height = 280  # Doubled from 140 to give more room
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0

        # Professional header
        header_rect = pygame.Rect(0, 0, self.width, header_height)
        pygame.draw.rect(self.screen, (18, 22, 32), header_rect)

        # Left border area for SUB indicator and spacing
        left_border_rect = pygame.Rect(
            0, header_height, self.left_border_width, self.height - header_height
        )
        pygame.draw.rect(self.screen, (12, 15, 20), left_border_rect)
        pygame.draw.line(
            self.screen,
            (40, 50, 60),
            (self.left_border_width, header_height),
            (self.left_border_width, self.height),
            1,
        )
        
        # Right border area for VU meters - extend from spectrum top to bottom
        if self.show_vu_meters:
            # Calculate where spectrum starts (after header + gap)
            spectrum_top = header_height + 10
            right_border_rect = pygame.Rect(
                self.width - vu_width, spectrum_top, vu_width, self.height - spectrum_top
            )
            pygame.draw.rect(self.screen, (12, 15, 20), right_border_rect)
            pygame.draw.line(
                self.screen,
                (40, 50, 60),
                (self.width - vu_width, spectrum_top),
                (self.width - vu_width, self.height),
                1,
            )

        # Draw organized header with responsive layout
        self.draw_organized_header(header_height)

        # Current time for any additional processing
        current_time = time.time()

        # Professional meters panel
        if self.show_professional_meters:
            meter_x = self.width - self.meter_panel_width - vu_width
            self.draw_professional_meters(
                meter_x,
                header_height,
                self.meter_panel_width,
                self.height - header_height - bass_zoom_height,
            )
        
        # Harmonic analysis panel
        if self.show_harmonic_analysis and hasattr(self, "harmonic_info"):
            harmonic_width = 250
            harmonic_x = self.width - meter_width - vu_width - harmonic_width - 10
            self.draw_harmonic_analysis(harmonic_x, header_height + 10, harmonic_width, 200)

        # Main spectrum visualization - adjusted for left border and VU meters
        vis_width = self.width - meter_width - vu_width - self.left_border_width - 10
        # Account for frequency scale height (40px) + gap (10px) = 50px total
        vis_height = self.height - header_height - bass_zoom_height - 50
        vis_start_x = self.left_border_width
        vis_start_y = header_height + 10

        # VU meters panel - extend to bottom of bass detail window (moved after vis calculations)
        if self.show_vu_meters:
            vu_x = self.width - self.vu_meter_width
            # Start at spectrum top (after header)
            vu_y = vis_start_y
            # Extend to bottom of bass detail window
            if self.show_bass_zoom:
                vu_height = (self.height - bass_zoom_height) - vu_y + bass_zoom_height
            else:
                vu_height = self.height - vu_y - 10
            
            self.draw_vu_meters(
                vu_x,
                vu_y,
                self.vu_meter_width,
                vu_height,
            )

        # Bass zoom window - extended to full spectrum width (moved after vis calculations)
        if self.show_bass_zoom:
            bass_y = self.height - bass_zoom_height
            # Calculate the x position where the spectrum ends (at the 20kHz mark)
            # This should align with the right edge of the main spectrum
            spectrum_end_x = vis_start_x + vis_width
            bass_width = spectrum_end_x - self.left_border_width
            
            self.draw_bass_zoom_window(
                self.left_border_width,
                bass_y,
                bass_width,
                bass_zoom_height - 10,
            )

        center_y = vis_start_y + vis_height // 2
        # Ensure bars don't extend beyond visualization area
        max_bar_height = (vis_height // 2) - 10

        # Professional studio enhancements - draw order is important
        self.update_peak_holds()

        # Draw professional grid and separators first (background elements)
        self.draw_smart_grid_labels()
        self.draw_frequency_band_separators()
        
        # Draw analysis grid if enabled (adds vertical frequency lines)
        if self.show_frequency_grid:
            self.draw_enhanced_analysis_grid()
        
        # Draw frequency scale after spectrum but before bass zoom
        # This will be drawn later after the spectrum bars

        # Enhanced spectrum bars
        bar_width = vis_width / self.bars

        for i in range(self.bars):
            if self.bar_heights[i] > 0.01:
                # Clamp bar height to prevent overflow when normalization is off
                clamped_height = min(self.bar_heights[i], 1.0)
                height = int(clamped_height * max_bar_height)
                x = vis_start_x + int(i * bar_width)
                width = max(1, int(bar_width))

                color = self.colors[i]

                # Enhanced reactivity
                freq_range = self.get_frequency_range_for_bar(i)

                # Voice reactivity
                if hasattr(self, "voice_info") and self.voice_info["has_voice"]:
                    pitch = self.voice_info.get("pitch", 0)
                    if pitch > 0 and freq_range[0] <= pitch <= freq_range[1]:
                        boost = 1.4
                        color = tuple(min(255, int(c * boost)) for c in color)

                # Harmonic highlighting
                if (
                    hasattr(self, "harmonic_info")
                    and self.harmonic_info.get("dominant_fundamental", 0) > 0
                ):
                    fund_freq = self.harmonic_info["dominant_fundamental"]
                    # Highlight fundamental and harmonics
                    for n in range(1, 8):
                        harmonic_freq = fund_freq * n
                        if abs(freq_range[0] - harmonic_freq) < 20:
                            boost = 1.3
                            color = tuple(min(255, int(c * boost)) for c in color)

                # Beat reactivity with enhanced colors
                if hasattr(self, "drum_info"):
                    kick_strength = self.drum_info["kick"].get("display_strength", 0)
                    snare_strength = self.drum_info["snare"].get("display_strength", 0)

                    if freq_range[0] <= 120 and kick_strength > 0.1:
                        boost = 1.0 + kick_strength * 1.2
                        color = tuple(min(255, int(c * boost)) for c in color)
                    elif 150 <= freq_range[0] <= 800 and snare_strength > 0.1:
                        boost = 1.0 + snare_strength * 0.8
                        color = tuple(min(255, int(c * boost)) for c in color)

                # Upper bar
                pygame.draw.rect(self.screen, color, (x, center_y - height, width, height))

                # Lower bar (mirrored)
                lower_color = tuple(int(c * 0.75) for c in color)
                pygame.draw.rect(self.screen, lower_color, (x, center_y, width, height))

        # Center line is now drawn in draw_smart_grid_labels() with 0dB label

        # Draw frequency scale now (after spectrum bars but before other overlays)
        self.draw_enhanced_frequency_scale()
        
        # Professional studio overlays (foreground elements)
        self.draw_peak_hold_indicators()
        self.draw_sub_bass_indicator()
        self.draw_adaptive_allocation_indicator()

        # Room mode analysis
        if self.show_room_analysis:
            self.detect_and_display_room_modes()
            
        # OMEGA: Pitch detection display (after harmonic analysis)
        if self.show_pitch_detection:
            self.draw_pitch_detection_info()
            
        # OMEGA-1: Chromagram and key detection display
        if self.show_chromagram:
            self.draw_chromagram_info()

        # Technical analysis overlay
        self.draw_technical_overlay()
        
        # Voice info display
        if self.show_voice_info:
            self.draw_voice_info()
            
        # Formant overlays
        if self.show_formants:
            self.draw_formant_overlays(vis_start_x, vis_start_y, vis_width, vis_height, center_y)

        # A/B comparison overlay
        if self.comparison_mode and self.reference_spectrum is not None:
            self.draw_ab_comparison()

        # Performance info now integrated in header

        # Room mode warnings
        if self.show_room_analysis and hasattr(self, "room_modes") and len(self.room_modes) > 0:
            warning_y = 110
            for mode in self.room_modes[:2]:  # Show top 2 room modes
                severity_color = (255, int(200 * (1 - mode["severity"])), 100)
                mode_text = f"Room Mode: {mode['frequency']:.0f}Hz (Q={mode['q_factor']:.1f})"
                mode_surf = self.font_tiny.render(mode_text, True, severity_color)
                self.screen.blit(mode_surf, (10, warning_y))
                warning_y += 15

    def resize_window(self, new_width: int, new_height: int):
        """Resize window with professional aspect ratios and update fonts"""
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)

        # Update fonts for new window size
        self.update_fonts(new_width)

        print(f"📏 Professional window resized to {new_width}x{new_height}")
        print(f"🔤 Font scale factor: {self.ui_scale:.2f}")

    def run(self):
        """Main professional analysis loop"""
        if not self.capture.start_capture():
            print("❌ Failed to start audio capture. Exiting.")
            return

        running = True
        frame_count = 0
        fps_timer = time.time()

        try:
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_s:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"professional_audio_analyzer_v5_{timestamp}.png"
                            pygame.image.save(self.screen, filename)
                            print(f"📸 Professional screenshot saved: {filename}")
                        elif event.key == pygame.K_m:
                            self.show_professional_meters = not self.show_professional_meters
                            print(
                                f"📊 Professional meters: {'ON' if self.show_professional_meters else 'OFF'}"
                            )
                        elif event.key == pygame.K_h:
                            self.show_harmonic_analysis = not self.show_harmonic_analysis
                            print(
                                f"🎼 Harmonic analysis: {'ON' if self.show_harmonic_analysis else 'OFF'}"
                            )
                        elif event.key == pygame.K_r:
                            self.show_room_analysis = not self.show_room_analysis
                            print(f"🏠 Room analysis: {'ON' if self.show_room_analysis else 'OFF'}")
                        elif event.key == pygame.K_z:
                            self.show_bass_zoom = not self.show_bass_zoom
                            print(f"🔍 Bass zoom: {'ON' if self.show_bass_zoom else 'OFF'}")
                        elif event.key == pygame.K_j:
                            self.show_vu_meters = not self.show_vu_meters
                            print(f"📊 VU meters: {'ON' if self.show_vu_meters else 'OFF'}")
                        elif event.key == pygame.K_v:
                            self.show_voice_info = not self.show_voice_info
                            print(f"🎤 Voice info: {'ON' if self.show_voice_info else 'OFF'}")
                        elif event.key == pygame.K_f:
                            self.show_formants = not self.show_formants
                            print(f"🔊 Formants: {'ON' if self.show_formants else 'OFF'}")
                        elif event.key == pygame.K_a:
                            self.show_advanced_info = not self.show_advanced_info
                            print(f"🔍 Advanced info: {'ON' if self.show_advanced_info else 'OFF'}")
                        elif event.key == pygame.K_g:
                            self.show_frequency_grid = not self.show_frequency_grid
                            print(
                                f"📊 Analysis grid: {'ON' if self.show_frequency_grid else 'OFF'}"
                            )
                        elif event.key == pygame.K_b:
                            self.show_band_separators = not self.show_band_separators
                            print(
                                f"📊 Band separators: {'ON' if self.show_band_separators else 'OFF'}"
                            )
                        elif event.key == pygame.K_t:
                            self.show_technical_overlay = not self.show_technical_overlay
                            print(
                                f"🔬 Technical overlay: {'ON' if self.show_technical_overlay else 'OFF'}"
                            )
                        # Compensation toggles for debugging
                        elif event.key == pygame.K_q:
                            self.freq_compensation_enabled = not self.freq_compensation_enabled
                            print(f"🎚️ Frequency compensation: {'ON' if self.freq_compensation_enabled else 'OFF'}")
                        elif event.key == pygame.K_w:
                            self.psychoacoustic_enabled = not self.psychoacoustic_enabled
                            print(f"👂 Psychoacoustic weighting: {'ON' if self.psychoacoustic_enabled else 'OFF'}")
                        elif event.key == pygame.K_e:
                            self.normalization_enabled = not self.normalization_enabled
                            print(f"📊 Normalization: {'ON' if self.normalization_enabled else 'OFF'}")
                        elif event.key == pygame.K_u:
                            self.smoothing_enabled = not self.smoothing_enabled
                            print(f"〰️ Smoothing: {'ON' if self.smoothing_enabled else 'OFF'}")
                        elif event.key == pygame.K_p:
                            # OMEGA: Toggle pitch detection
                            self.show_pitch_detection = not self.show_pitch_detection
                            print(f"🎵 Pitch Detection (OMEGA): {'ON' if self.show_pitch_detection else 'OFF'}")
                        elif event.key == pygame.K_k:
                            # OMEGA-1: Toggle chromagram and key detection
                            self.show_chromagram = not self.show_chromagram
                            print(f"🎹 Chromagram & Key Detection (OMEGA-1): {'ON' if self.show_chromagram else 'OFF'}")
                        elif event.key == pygame.K_y:
                            # Toggle dynamic quality mode (moved from P)
                            self.dynamic_quality_enabled = not self.dynamic_quality_enabled
                            print(f"🚀 Dynamic Quality: {'ON' if self.dynamic_quality_enabled else 'OFF'}")
                            if not self.dynamic_quality_enabled and self.quality_mode == "performance":
                                # Switch back to quality mode if dynamic is disabled
                                self.quality_mode = "quality"
                                self._switch_to_quality_mode()
                        elif event.key == pygame.K_c:
                            # Store reference spectrum for A/B comparison
                            if self.reference_spectrum is None:
                                self.reference_spectrum = self.bar_heights.copy()
                                self.reference_stored = True
                                print("📊 Reference spectrum stored for A/B comparison")
                            else:
                                self.comparison_mode = not self.comparison_mode
                                print(
                                    f"⚖️ A/B comparison: {'ON' if self.comparison_mode else 'OFF'}"
                                )
                        elif event.key == pygame.K_x:
                            # Clear reference spectrum
                            self.reference_spectrum = None
                            self.reference_stored = False
                            self.comparison_mode = False
                            print("🗑️ Reference spectrum cleared")
                        elif event.key == pygame.K_SLASH or event.key == pygame.K_QUESTION:
                            # Show help
                            self.show_help_controls()
                        # Window sizes
                        elif event.key == pygame.K_1:
                            self.resize_window(1400, 900)  # Professional Compact
                        elif event.key == pygame.K_2:
                            self.resize_window(1800, 1000)  # Professional Standard
                        elif event.key == pygame.K_3:
                            self.resize_window(2200, 1200)  # Professional Wide
                        elif event.key == pygame.K_4:
                            self.resize_window(2600, 1400)  # Professional Ultra-Wide
                        elif event.key == pygame.K_5:
                            self.resize_window(3000, 1600)  # Professional Cinema
                        elif event.key == pygame.K_6:
                            self.resize_window(3400, 1800)  # Professional Studio
                        elif event.key == pygame.K_7:
                            self.resize_window(1920, 1080)  # Full HD Standard
                        elif event.key == pygame.K_8:
                            self.resize_window(2560, 1440)  # 2K Monitor
                        elif event.key == pygame.K_9:
                            self.resize_window(3840, 2160)  # 4K Monitor
                        elif event.key == pygame.K_0:
                            # Fullscreen toggle
                            pygame.display.toggle_fullscreen()
                        elif event.key == pygame.K_d:
                            # Debug output
                            self.print_debug_output()
                            print("🐛 Debug output printed to terminal")
                        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            # Increase gain
                            self.input_gain *= 1.5
                            self.input_gain = min(self.input_gain, 16.0)  # Max +24dB
                            gain_db = 20 * np.log10(self.input_gain)
                            print(f"🔊 Input gain: +{gain_db:.1f}dB")
                        elif event.key == pygame.K_MINUS:
                            # Decrease gain
                            self.input_gain /= 1.5
                            self.input_gain = max(self.input_gain, 0.25)  # Min -12dB
                            gain_db = 20 * np.log10(self.input_gain)
                            print(f"🔊 Input gain: {gain_db:+.1f}dB")
                        elif event.key == pygame.K_0:
                            # Reset gain to default
                            self.input_gain = 4.0
                            print(f"🔊 Input gain reset to +12.0dB")
                        elif event.key == pygame.K_k:
                            # Toggle adaptive allocation
                            self.adaptive_allocation_enabled = not self.adaptive_allocation_enabled
                            if not self.adaptive_allocation_enabled:
                                # Reset to default 75% allocation
                                self.current_allocation = 0.75
                                self.current_content_type = 'instrumental'
                            print(f"🎛️ Adaptive allocation: {'ON' if self.adaptive_allocation_enabled else 'OFF'}")
                        elif event.key == pygame.K_i:
                            # Toggle midrange boost
                            self.midrange_boost_enabled = not self.midrange_boost_enabled
                            # Recalculate frequency compensation when toggled
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost (1k-6k): {'ON' if self.midrange_boost_enabled else 'OFF'}")
                        elif event.key == pygame.K_o:
                            # Increase midrange boost factor
                            self.midrange_boost_factor = min(self.midrange_boost_factor + 0.5, 10.0)
                            # Recalculate frequency compensation
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost factor: {self.midrange_boost_factor:.1f}x")
                        elif event.key == pygame.K_l:
                            # Decrease midrange boost factor
                            self.midrange_boost_factor = max(self.midrange_boost_factor - 0.5, 1.0)
                            # Recalculate frequency compensation
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost factor: {self.midrange_boost_factor:.1f}x")

                # Process audio and draw
                self.process_frame()
                self.draw_frame()

                # Update display
                pygame.display.flip()
                self.clock.tick(60)

                # FPS tracking
                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps_counter.append(frame_count)
                    frame_count = 0
                    fps_timer = time.time()

        finally:
            # Clean shutdown
            self.bass_thread_running = False
            self.bass_processing_queue.put(None)  # Signal thread to exit
            self.bass_processing_thread.join(timeout=1.0)  # Wait for thread to finish
            self.capture.stop_capture()
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Professional Audio Analyzer v4 - Perceptual Frequency Mapping")
    parser.add_argument("--width", type=int, default=2200, help="Window width")
    parser.add_argument("--height", type=int, default=1200, help="Window height")
    parser.add_argument("--bars", type=int, default=768, help="Number of spectrum bars")
    parser.add_argument("--source", type=str, default=None, help="PipeWire source name")

    args = parser.parse_args()

    print("\n" + "=" * 90)
    print("PROFESSIONAL AUDIO ANALYZER V4 - PERCEPTUAL FREQUENCY MAPPING")
    print("=" * 90)
    print("Enhanced Features:")
    print("  🎚️ Multi-Resolution FFT: 8192 samples for bass, optimized for each range")
    print("  📊 Professional Metering: LUFS, K-weighting, True Peak per ITU-R BS.1770-4")
    print("  🌈 Enhanced Bass Detail: Gradient visualization with faster response")
    print("  ⚖️ A/B Comparison: Store and compare frequency responses")
    print("  🎛️ Complete Control Suite: All features fully toggleable")
    
    print("\n🎹 QUICK START - ESSENTIAL HOTKEYS:")
    print("  ?/: Show complete help                    ESC: Exit")
    print("  M: Professional meters                    Z: Bass detail panel")
    print("  C: Store reference for A/B comparison     0: Fullscreen toggle")
    print("  S: Save screenshot                        1-9: Window sizes")
    
    print("\n🎚️ AUDIO PROCESSING TOGGLES:")
    print("  Q: Frequency compensation                 W: Psychoacoustic weighting")  
    print("  E: Normalization                          U: Smoothing")
    print("  +/-: Input gain                           D: Debug output")
    print("  I: Midrange boost (1k-6k)                 O/L: Adjust boost factor")
    
    print("\n📊 DISPLAY TOGGLES:")
    print("  H: Harmonic analysis    R: Room analysis     G: Analysis grid")
    print("  B: Band separators      T: Technical overlay V: Voice info")
    print("  F: Formants            A: Advanced info")
    
    print("💡 Press '?' anytime during operation for complete help with current status")
    print("=" * 90)
    print("  🎼 Harmonic Analysis: Instrument identification, harmonic series tracking")
    print("  🔊 Phase Coherence: Stereo imaging analysis (ready for stereo input)")
    print("  🏠 Room Mode Detection: Studio acoustics analysis for mixing rooms")
    print("  ⚡ Transient Analysis: Attack time, punch factor, dynamics measurement")
    print("  🎯 Adaptive Frequency Allocation: Dynamic bass detail (60-80% based on content)")
    print("  🎨 Professional UI: Studio-grade visualization with multiple panels")
    print("Professional Controls:")
    print("  M: Professional meters | H: Harmonic analysis | R: Room analysis")
    print("  Z: Bass zoom window | S: Screenshot | V/F/A: Voice/Formant/Advanced")
    print("  K: Toggle adaptive frequency allocation | D: Debug output")
    print("  1-6: Professional window presets (1400x900 to 3400x1800)")
    print("=" * 90)

    analyzer = ProfessionalLiveAudioAnalyzer(
        width=args.width, height=args.height, bars=args.bars, source_name=args.source
    )

    analyzer.run()


if __name__ == "__main__":
    main()
