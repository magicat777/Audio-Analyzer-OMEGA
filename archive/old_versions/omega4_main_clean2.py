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


from omega4.config import (
    SAMPLE_RATE, CHUNK_SIZE, BARS_DEFAULT, BARS_MAX, 
    MAX_FREQ, FFT_SIZE_BASE, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    TARGET_FPS, BACKGROUND_COLOR, GRID_COLOR, TEXT_COLOR,
    get_config
)

from omega4.visualization.display_interface import SpectrumDisplay

from omega4.panels.professional_meters import ProfessionalMetersPanel
from omega4.panels.vu_meters import VUMetersPanel
from omega4.panels.bass_zoom import BassZoomPanel
from omega4.panels.harmonic_analysis import HarmonicAnalysisPanel
from omega4.panels.pitch_detection import PitchDetectionPanel
from omega4.panels.chromagram import ChromagramPanel
from omega4.panels.genre_classification import GenreClassificationPanel

from omega4.analyzers import (
    PhaseCoherenceAnalyzer,
    TransientAnalyzer,
    RoomModeAnalyzer,
    EnhancedDrumDetector
)

from omega4.audio import (
    PipeWireMonitorCapture,
    AudioCaptureManager,
    MultiResolutionFFT,
    AudioProcessingPipeline,
    ContentTypeDetector,
    VoiceDetectionWrapper
)






from omega4.panels.harmonic_analysis import HarmonicAnalyzer

from omega4.panels.pitch_detection import CepstralAnalyzer

        self.nyquist = sample_rate / 2

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

        self.harmonic_history = deque(maxlen=30)
        self.instrument_confidence = {}

    def detect_harmonic_series(self, fft_data: np.ndarray, freqs: np.ndarray) -> Dict[str, Any]:
        """Detect harmonic series and identify potential instruments"""
        harmonics_found = []

        peaks, properties = scipy_signal.find_peaks(
            fft_data,
            height=np.max(fft_data) * 0.1,  # At least 10% of maximum
            distance=10,  # Minimum separation
        )

        for peak_idx in peaks:
            fundamental_freq = freqs[peak_idx]
            fundamental_magnitude = fft_data[peak_idx]

            if 20 <= fundamental_freq <= 2000:  # Reasonable fundamental range
                harmonic_info = self.analyze_harmonics_from_fundamental(
                    fft_data, freqs, fundamental_freq, fundamental_magnitude
                )

                if harmonic_info["strength"] > 0.3:  # Significant harmonic content
                    harmonics_found.append(harmonic_info)

        harmonics_found.sort(key=lambda x: x["strength"], reverse=True)

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

            freq_idx = np.argmin(np.abs(freqs - harmonic_freq))
            actual_freq = freqs[freq_idx]

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

        harmonic_strength = 0.0
        if len(harmonics) >= 2:
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

            for instrument, profile in self.instrument_profiles.items():
                score = self.calculate_instrument_match_score(fundamental, harmonics, profile)

                if instrument not in instrument_scores:
                    instrument_scores[instrument] = []
                instrument_scores[instrument].append(score)

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

        fund_min, fund_max = profile["fundamental_range"]
        if not (fund_min <= fundamental <= fund_max):
            return 0.0

        expected_pattern = profile["harmonic_pattern"]

        if len(harmonics) >= 2:
            pattern_score = 0.0
            pattern_matches = 0

            for i, expected_strength in enumerate(expected_pattern):
                harmonic_num = i + 1

                actual_harmonic = None
                for h in harmonics:
                    if h["number"] == harmonic_num:
                        actual_harmonic = h
                        break

                if actual_harmonic:
                    actual_strength = actual_harmonic["relative_strength"]
                    difference = abs(actual_strength - expected_strength)
                    match_score = max(0.0, 1.0 - difference)
                    pattern_score += match_score
                    pattern_matches += 1

            if pattern_matches > 0:
                score = pattern_score / pattern_matches

        return min(1.0, score)






            return "oblique"  # Oblique modes


        self.pitch_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        
        self.min_pitch = 50  # Hz (roughly G1)
        self.max_pitch = 2000  # Hz (roughly B6)
        self.min_period = int(sample_rate / self.max_pitch)
        self.max_period = int(sample_rate / self.min_pitch)
        
        self.yin_threshold = 0.15
        self.yin_window_size = 1024  # Reduced for lower latency
        
        self.high_confidence = 0.8
        self.medium_confidence = 0.5
        
    def compute_cepstrum(self, signal: np.ndarray) -> np.ndarray:
        """Compute real cepstrum for pitch detection"""
        windowed = signal * np.hanning(len(signal))
        
        spectrum = np.fft.rfft(windowed)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.irfft(log_spectrum)
        
        return cepstrum[:len(cepstrum)//2]
    
    def detect_pitch_cepstral(self, cepstrum: np.ndarray) -> Tuple[float, float]:
        """Detect pitch using cepstral peak detection"""
        min_sample = self.min_period
        max_sample = min(self.max_period, len(cepstrum) - 1)
        
        if max_sample <= min_sample:
            return 0.0, 0.0
            
        valid_range = cepstrum[min_sample:max_sample]
        peaks, properties = scipy_signal.find_peaks(
            valid_range, 
            height=np.max(valid_range) * 0.3,
            distance=20
        )
        
        if len(peaks) == 0:
            return 0.0, 0.0
            
        peak_idx = peaks[np.argmax(properties['peak_heights'])]
        period = peak_idx + min_sample
        pitch = self.sample_rate / period
        
        confidence = min(1.0, properties['peak_heights'][np.argmax(properties['peak_heights'])] / np.std(cepstrum))
        
        return pitch, confidence
    
    def compute_autocorrelation(self, signal: np.ndarray) -> np.ndarray:
        """Compute normalized autocorrelation function"""
        signal = signal - np.mean(signal)
        
        n = len(signal)
        padded_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        
        fft = np.fft.rfft(signal, padded_size)
        power_spectrum = fft * np.conj(fft)
        autocorr = np.fft.irfft(power_spectrum)
        
        autocorr = autocorr[:n]
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        return autocorr
    
    def detect_pitch_autocorrelation(self, signal: np.ndarray) -> Tuple[float, float]:
        """Detect pitch using autocorrelation method"""
        autocorr = self.compute_autocorrelation(signal)
        
        min_lag = self.min_period
        max_lag = min(self.max_period, len(autocorr) - 1)
        
        if max_lag <= min_lag:
            return 0.0, 0.0
            
        valid_range = autocorr[min_lag:max_lag]
        peaks, properties = scipy_signal.find_peaks(
            valid_range,
            height=0.3,  # At least 30% correlation
            distance=20
        )
        
        if len(peaks) == 0:
            return 0.0, 0.0
            
        peak_idx = peaks[0]  # First peak is usually fundamental
        period = peak_idx + min_lag
        pitch = self.sample_rate / period
        
        confidence = autocorr[period]
        
        return pitch, confidence
    
    def yin_pitch_detection(self, signal: np.ndarray) -> Tuple[float, float]:
        """YIN algorithm for robust pitch detection"""
        n = len(signal)
        if n < self.yin_window_size:
            return 0.0, 0.0
            
        signal = signal[:self.yin_window_size]
        n = self.yin_window_size
        
        diff_function = np.zeros(n // 2)
        for tau in range(1, min(self.max_period, n // 2)):
            diff_function[tau] = np.sum((signal[:-tau] - signal[tau:])**2)
        
        cumulative_mean = np.zeros(n // 2)
        cumulative_mean[0] = 1.0
        
        running_sum = 0.0
        for tau in range(1, n // 2):
            running_sum += diff_function[tau]
            cumulative_mean[tau] = diff_function[tau] / (running_sum / tau) if running_sum > 0 else 1.0
        
        tau = self.min_period
        while tau < min(self.max_period, n // 2 - 1):
            if cumulative_mean[tau] < self.yin_threshold:
                if tau + 1 < n // 2 and cumulative_mean[tau] < cumulative_mean[tau - 1] and cumulative_mean[tau] < cumulative_mean[tau + 1]:
                    break
            tau += 1
        
        if tau < min(self.max_period, n // 2 - 1):
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
        cepstral_pitch, cepstral_conf = self.detect_pitch_cepstral(cepstrum)
        
        autocorr_pitch, autocorr_conf = self.detect_pitch_autocorrelation(signal)
        
        yin_pitch, yin_conf = self.yin_pitch_detection(signal)
        
        estimates = []
        if cepstral_conf > 0.2 and self.min_pitch <= cepstral_pitch <= self.max_pitch:
            estimates.append((cepstral_pitch, cepstral_conf * 0.8))  # Slightly lower weight for cepstral
            
        if autocorr_conf > 0.3 and self.min_pitch <= autocorr_pitch <= self.max_pitch:
            estimates.append((autocorr_pitch, autocorr_conf * 0.9))
            
        if yin_conf > 0.4 and self.min_pitch <= yin_pitch <= self.max_pitch:
            estimates.append((yin_pitch, yin_conf * 1.0))  # YIN typically most reliable
        
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
        
        total_weight = sum(conf for _, conf in estimates)
        weighted_pitch = sum(pitch * conf for pitch, conf in estimates) / total_weight
        combined_confidence = total_weight / len(estimates)
        
        note_info = self.pitch_to_note(weighted_pitch)
        
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
            
        A4 = 440.0
        C0 = A4 * (2 ** (-4.75))  # C0 frequency
        
        semitones = 12 * np.log2(pitch / C0)
        rounded_semitones = int(round(semitones))
        cents_offset = int((semitones - rounded_semitones) * 100)
        
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
        
        valid_pitches = [p for p, c in zip(recent_pitches, recent_confidences) if c > 0.5 and p > 0]
        
        if len(valid_pitches) < 5:
            return 0.0
            
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
            
        cepstrum = self.compute_cepstrum(signal)
        
        return self.combine_pitch_estimates(signal, cepstrum)






        self.snare_detector.sensitivity = sensitivity




class ProfessionalLiveAudioAnalyzer:
    """Professional Live Audio Analyzer v4.1 OMEGA with enhanced low-end detail and studio features"""

    def __init__(self, width=2000, height=1080, bars=BARS_DEFAULT, source_name=None):
        self.width = width
        self.height = height
        self.bars = bars

        self.capture = PipeWireMonitorCapture(source_name, SAMPLE_RATE, CHUNK_SIZE)
        self.drum_detector = EnhancedDrumDetector(SAMPLE_RATE)
        self.voice_detector = VoiceDetectionWrapper(SAMPLE_RATE)  # Phase 5: Use wrapper
        self.content_detector = ContentTypeDetector()
        self.audio_pipeline = AudioProcessingPipeline(SAMPLE_RATE, self.bars)

        self.multi_fft = MultiResolutionFFT(SAMPLE_RATE)
        self.vu_meters_panel = VUMetersPanel(SAMPLE_RATE)
        self.bass_zoom_panel = BassZoomPanel(SAMPLE_RATE)
        self.harmonic_analysis_panel = HarmonicAnalysisPanel(SAMPLE_RATE)
        self.pitch_detection_panel = PitchDetectionPanel(SAMPLE_RATE)
        self.chromagram_panel = ChromagramPanel(SAMPLE_RATE)  # OMEGA-1: Musical key detection panel
        self.genre_classification_panel = GenreClassificationPanel(SAMPLE_RATE)  # OMEGA-2: Genre classification panel
        self.phase_analyzer = PhaseCoherenceAnalyzer(SAMPLE_RATE)
        self.transient_analyzer = TransientAnalyzer(SAMPLE_RATE)
        self.room_analyzer = RoomModeAnalyzer(SAMPLE_RATE)
        
        self.input_gain = 4.0  # Default 12dB boost for typical music sources
        self.auto_gain_enabled = True
        self.gain_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.target_lufs = -16.0  # Target for good visualization
        
        self.freq_compensation_enabled = True  # Toggle with 'Q'
        self.psychoacoustic_enabled = True     # Toggle with 'W'
        self.normalization_enabled = True      # Toggle with 'E'
        self.smoothing_enabled = True          # Toggle with 'R'

        self.ring_buffer = np.zeros(FFT_SIZE_BASE * 4, dtype=np.float32)  # Larger buffer
        self.buffer_pos = 0

        self.freqs = np.fft.rfftfreq(FFT_SIZE_BASE, 1 / SAMPLE_RATE)
        
        self.adaptive_allocation_enabled = False  # Disabled for perceptual mapping
        self.current_content_type = 'instrumental'
        self.current_allocation = 0.75
        self.band_indices = self._create_enhanced_band_mapping()

        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Professional Audio Analyzer v4.1 OMEGA-2 - Genre Classification")
        self.clock = pygame.time.Clock()
        
        self.display = SpectrumDisplay(self.screen, width, height, self.bars)

        self.base_width = 2000  # Reference width for font scaling
        self.update_fonts(width)
        
        self.display.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny,
            'grid': self.font_grid
        })
        
        self.professional_meters_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny,
            'meters': self.font_meters
        })
        
        self.vu_meters_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.bass_zoom_panel.set_fonts({
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.harmonic_analysis_panel.set_fonts({
            'medium': self.font_medium,
            'small': self.font_small
        })
        
        self.pitch_detection_panel.set_fonts({
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.chromagram_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.genre_classification_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })

        self.bar_heights = np.zeros(self.bars, dtype=np.float32)
        self.colors = self._generate_professional_colors()

        self.midrange_boost_enabled = True
        self.midrange_boost_factor = 2.0  # 2x boost for 1k-6k range

        self.freq_compensation_gains = self._precalculate_freq_compensation()
        
        self._setup_optimized_band_mapping()
        
        self._setup_frequency_ranges()
        
        if len(self.bar_heights) != self.bars:
            self.bar_heights = np.zeros(self.bars, dtype=np.float32)
        
        
        self._build_frequency_lookup()

        self.last_process_time = 0
        self.peak_latency = 0.0  # Track highest latency
        self.fps_counter = deque(maxlen=30)
        self.frame_counter = 0  # For skipping expensive operations
        
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
        
        self.quality_mode = "quality"  # "quality" or "performance"
        self.dynamic_quality_enabled = True
        self.performance_threshold = 30.0  # ms - switch to performance mode above this
        self.quality_threshold = 25.0  # ms - switch back to quality mode below this
        self.quality_mode_hold_time = 10.0  # seconds - minimum time before switching modes
        self.last_quality_switch_time = 0.0  # timestamp of last mode switch
        
        
        self.show_vu_meters = True  # Toggle with 'J'
        
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
        self.genre_info = {  # OMEGA-2: Initialize genre classification results
            'genres': {},
            'top_genre': 'Unknown',
            'confidence': 0.0,
            'features': {}
        }

        self.kick_flash_time = 0
        self.snare_flash_time = 0
        self.voice_flash_time = 0
        self.singing_flash_time = 0

        self.peak_hold_bars = np.zeros(self.bars, dtype=np.float32)
        self.peak_decay_rate = 0.85  # More aggressive decay - drops to ~0.4% after 20 frames
        self.peak_hold_time = 3.0  # Hold peaks for 3 seconds (reduced from 5)
        self.peak_timestamps = np.zeros(self.bars, dtype=np.float64)  # Use float64 for timestamps
        
        self.bass_peak_hold = {}  # Will store peak values for bass frequencies
        self.bass_peak_timestamps = {}  # Timestamps for bass peaks

        self.sub_bass_energy = 0.0
        self.sub_bass_history = deque(maxlen=100)
        self.sub_bass_warning_active = False
        self.sub_bass_warning_time = 0
        
        self.ultra_sub_bass_energy = 0.0
        self.ultra_sub_bass_history = deque(maxlen=100)

        self.sustained_peaks = {}  # freq -> (magnitude, start_time, x_pos)
        self.room_mode_candidates = []

        self.reference_spectrum = None
        self.comparison_mode = False
        self.reference_stored = False

        self.show_technical_overlay = False
        self.last_technical_analysis = {}

        self.show_frequency_grid = True
        self.show_band_separators = True
        self.show_note_labels = True

        self.show_professional_meters = True
        self.show_harmonic_analysis = True
        self.show_room_analysis = False
        self.show_bass_zoom = True
        self.show_advanced_info = False
        self.show_voice_info = True
        self.show_formants = False
        self.show_pitch_detection = False  # OMEGA: Disabled by default for performance
        self.show_chromagram = False  # OMEGA-1: Disabled by default for performance
        self.show_genre_classification = False  # OMEGA-2: Disabled by default for performance
        self.show_help = False  # Help menu overlay

        self.meter_panel_width = 280  # Increased from 200 to prevent title cutoff
        self.bass_zoom_height = 200  # Increased for 64-bar display
        self.left_border_width = 100  # Left border for SUB level indicator and spacing - increased for more separation
        self.vu_meter_width = 190  # Width for VU meters on the right

        print(f"\n🎵 Professional Audio Analyzer v4.1 OMEGA-2 - Genre Classification")
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
        print("  J: Toggle VU meters")
        print("  R: Toggle room analysis")
        print("  Z: Toggle bass zoom window")
        print("  G: Toggle analysis grid")
        print("  B: Toggle band separators")
        print("  T: Toggle technical overlay")
        print("  P: Toggle pitch detection (OMEGA)")
        print("  C: Store reference / Toggle A/B comparison")
        print("  X: Clear reference spectrum")
        print("  Q: Toggle Fequency Compensation")
        print("  W: Toggle Pyschoacoustic Compensation")
        print("  E: Toggle Normalization")   
        print("  ESC/S/V/F/A: Standard controls")
        print("Window Presets:")
        print("  1: 1400x900 (Compact)     2: 1800x1000 (Standard)")
        print("  3: 2200x1200 (Wide)       4: 2600x1400 (Ultra-Wide)")
        print("  5: 3000x1600 (Cinema)     6: 3400x1800 (Studio)")
        print("  7: 1920x1080 (Full HD)    8: 2560x1440 (2K)")
        print("  9: 3840x2160 (4K)         0: Toggle Fullscreen")

    def update_fonts(self, current_width: int):
        """Update fonts based on current window width for optimal readability"""
        scale_factor = current_width / self.base_width
        scale_factor = max(0.6, min(1.4, scale_factor))  # Clamp between 60% and 140%

        self.font_large = pygame.font.Font(None, max(24, int(36 * scale_factor)))
        self.font_medium = pygame.font.Font(None, max(20, int(28 * scale_factor)))
        self.font_small = pygame.font.Font(None, max(18, int(24 * scale_factor)))
        self.font_tiny = pygame.font.Font(None, max(16, int(20 * scale_factor)))

        self.font_meters = pygame.font.Font(None, max(18, int(22 * scale_factor)))
        self.font_grid = pygame.font.Font(None, max(14, int(18 * scale_factor)))
        self.font_mono = pygame.font.Font(None, max(16, int(20 * scale_factor)))

        self.ui_scale = scale_factor

    def draw_organized_header(self, header_height: int):
        """Draw organized header layout to prevent text overlap"""
        col1_x = 20
        col2_x = int(self.width * 0.35)
        col3_x = int(self.width * 0.65)

        row_height = int(25 * self.ui_scale)
        y_start = 15

        y = y_start
        title = self.font_large.render("Professional Audio Analyzer v4.1 OMEGA-2", True, (255, 255, 255))
        self.screen.blit(title, (col1_x, y))
        y += int(40 * self.ui_scale)

        subtitle_text = (
            "Musical Perceptual Mapping • Professional Metering • Harmonic Analysis • Room Acoustics"
        )
        subtitle = self.font_small.render(subtitle_text, True, (180, 200, 220))
        if subtitle.get_width() > self.width - col1_x - 100:
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

        controls_hint = self.font_tiny.render(
            "🎛️ Gain: +/- keys • 0: Toggle Auto-gain • ESC: Exit", True, (120, 140, 160)
        )
        self.screen.blit(controls_hint, (col1_x, y))

        if col2_x < self.width - 200:  # Only show if there's space
            y = y_start + int(60 * self.ui_scale)

            if hasattr(self, "capture") and self.capture:
                source_text = "🎤 Audio Source: Professional Monitor"
                source_surf = self.font_small.render(source_text, True, (120, 200, 120))
                self.screen.blit(source_surf, (col2_x, y))
                y += row_height

                tech_info = f"📊 {SAMPLE_RATE}Hz • {self.bars} bars"
                tech_surf = self.font_tiny.render(tech_info, True, (140, 160, 180))
                self.screen.blit(tech_surf, (col2_x, y))
                y += row_height

                fft_info = "Multi-FFT: 8192/4096/2048/1024"
                fft_surf = self.font_tiny.render(fft_info, True, (140, 160, 180))
                self.screen.blit(fft_surf, (col2_x, y))

        if col3_x < self.width - 150:  # Only show if there's space
            y = y_start + int(60 * self.ui_scale)

            if hasattr(self, "last_process_time"):
                latency_ms = self.last_process_time * 1000
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

                if hasattr(self, "fps_counter") and len(self.fps_counter) > 0:
                    avg_fps = sum(self.fps_counter) / len(self.fps_counter)
                    fps_color = (120, 200, 120) if avg_fps > 55 else (200, 200, 120)
                    fps_text = f"📈 FPS: {avg_fps:.1f}"
                    fps_surf = self.font_tiny.render(fps_text, True, fps_color)
                    self.screen.blit(fps_surf, (col3_x, y))
                    y += row_height
                
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

            gain_db = 20 * np.log10(self.input_gain)
            gain_color = (120, 200, 120) if self.auto_gain_enabled else (200, 200, 120)
            gain_text = f"🎚️ Gain: +{gain_db:.1f}dB"
            if self.auto_gain_enabled:
                gain_text += " (Auto)"
            gain_surf = self.font_small.render(gain_text, True, gain_color)
            self.screen.blit(gain_surf, (col3_x, y))
            y += row_height

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
        header_height = 280
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        
        
        scale_y = spectrum_bottom + 5

        secondary_frequencies = [30, 40, 80, 150, 300, 700, 1500, 3000, 5000, 8000, 15000]

        spectrum_left = self.left_border_width
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_right = self.width - meter_width - vu_width - 10
        
        scale_bg_rect = pygame.Rect(spectrum_left - 5, scale_y - 5, spectrum_right - spectrum_left + 10, 45)
        pygame.draw.rect(self.screen, (15, 20, 30), scale_bg_rect)
        
        pygame.draw.line(
            self.screen, (120, 120, 140), (spectrum_left, scale_y), (spectrum_right, scale_y), 3
        )
        
        for freq in secondary_frequencies:
            x_pos = int(self.freq_to_x_position(freq))
            if spectrum_left <= x_pos <= spectrum_right:
                tick_height = int(5 * self.ui_scale)
                pygame.draw.line(
                    self.screen, (140, 140, 150), (x_pos, scale_y), (x_pos, scale_y + tick_height), 2
                )
        
        self.last_label_x = -100
        for freq in primary_frequencies:
            x_pos = int(self.freq_to_x_position(freq))

            if x_pos < spectrum_left or x_pos > spectrum_right:
                continue

            tick_height = int(10 * self.ui_scale)
            if freq in [60, 250, 500, 2000, 6000]:
                pygame.draw.line(
                    self.screen, (200, 200, 220), (x_pos, scale_y - 3), (x_pos, scale_y + tick_height + 3), 3
                )
            else:
                pygame.draw.line(
                    self.screen, (170, 170, 180), (x_pos, scale_y), (x_pos, scale_y + tick_height), 2
                )

            min_spacing = max(35, int(40 * self.ui_scale))
            if abs(x_pos - self.last_label_x) < min_spacing:
                continue

            if freq >= 10000:
                freq_text = f"{freq/1000:.0f}k"
            elif freq >= 1000:
                freq_text = f"{freq/1000:.0f}k" if freq % 1000 == 0 else f"{freq/1000:.1f}k"
            else:
                freq_text = f"{freq}"

            label = self.font_grid.render(freq_text, True, (220, 220, 230))
            label_rect = label.get_rect(centerx=x_pos, top=scale_y + tick_height + 3)

            bg_padding = int(2 * self.ui_scale)
            pygame.draw.rect(
                self.screen, (20, 25, 35), label_rect.inflate(bg_padding * 2, bg_padding)
            )
            self.screen.blit(label, label_rect)

            self.last_label_x = x_pos

    def draw_smart_grid_labels(self):
        """Grid with dB labels for mirrored spectrum (0dB at center)"""
        if self.height < 900:
            db_levels = [0, -20, -40, -60]  # Minimal set for small windows
        else:
            db_levels = [0, -10, -20, -30, -40, -50, -60]  # Full set for large windows

        spectrum_left = self.left_border_width
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_right = self.width - meter_width - vu_width - 10
        spectrum_top = 280 + 10  # Header height + gap
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50
        
        center_y = spectrum_top + (spectrum_bottom - spectrum_top) // 2
        spectrum_height = spectrum_bottom - spectrum_top
        
        pygame.draw.line(
            self.screen, (60, 70, 90), (spectrum_left, center_y), (spectrum_right, center_y), 2
        )
        
        label_text = "  0"
        label = self.font_grid.render(label_text, True, (150, 160, 170))
        label_rect = label.get_rect(right=spectrum_left - 5, centery=center_y)
        if label_rect.left >= 5:
            bg_padding = int(2 * self.ui_scale)
            pygame.draw.rect(
                self.screen, (15, 20, 30), label_rect.inflate(bg_padding * 2, bg_padding)
            )
            self.screen.blit(label, label_rect)

        for db in db_levels[1:]:  # Skip 0dB as we already drew it
            normalized_pos = abs(db) / 60.0
            
            upper_y = int(center_y - (spectrum_height / 2) * normalized_pos)
            lower_y = int(center_y + (spectrum_height / 2) * normalized_pos)
            
            if spectrum_top <= upper_y <= center_y:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), (spectrum_left, upper_y), (spectrum_right, upper_y), 1
                )
            if center_y <= lower_y <= spectrum_bottom:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), (spectrum_left, lower_y), (spectrum_right, lower_y), 1
                )
            
            label_text = f"{db:3d}"
            label = self.font_grid.render(label_text, True, (120, 120, 130))
            
            if spectrum_top <= upper_y <= center_y:
                label_rect = label.get_rect(right=spectrum_left - 5, centery=upper_y)
                if label_rect.left >= 5:
                    bg_padding = int(2 * self.ui_scale)
                    pygame.draw.rect(
                        self.screen, (15, 20, 30), label_rect.inflate(bg_padding * 2, bg_padding)
                    )
                    self.screen.blit(label, label_rect)
            
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
        fft_bin_width = self.freqs[1] - self.freqs[0]  # Hz per bin
        min_fft_freq = fft_bin_width  # First non-DC bin
        
        freq_min = max(20, min_fft_freq)
        freq_max = min(MAX_FREQ, self.freqs[-1])

        
        
        total_bars = self.bars
        
        sub_bass_bars = int(total_bars * 0.08)    # 8% for 20-60 Hz
        bass_bars = int(total_bars * 0.17)        # 17% for 60-250 Hz
        low_mid_bars = int(total_bars * 0.15)     # 15% for 250-500 Hz
        mid_bars = int(total_bars * 0.25)         # 25% for 500-2000 Hz (critical range)
        high_mid_bars = int(total_bars * 0.20)    # 20% for 2000-6000 Hz
        high_bars = total_bars - (sub_bass_bars + bass_bars + low_mid_bars + mid_bars + high_mid_bars)  # 15% for 6000-20000 Hz
        
        band_indices = []
        freq_edges = [freq_min]  # Start with minimum frequency
        
        valid_bin_indices = np.where((self.freqs >= freq_min) & (self.freqs <= freq_max))[0]
        num_valid_bins = len(valid_bin_indices)
        
        if num_valid_bins < total_bars:
            print(f"⚠️  Reducing bars from {total_bars} to {num_valid_bins} due to FFT resolution")
            self.bars = num_valid_bins
            total_bars = num_valid_bins
            
            sub_bass_bars = max(1, int(total_bars * 0.08))
            bass_bars = max(1, int(total_bars * 0.17))
            low_mid_bars = max(1, int(total_bars * 0.15))
            mid_bars = max(1, int(total_bars * 0.25))
            high_mid_bars = max(1, int(total_bars * 0.20))
            high_bars = total_bars - (sub_bass_bars + bass_bars + low_mid_bars + mid_bars + high_mid_bars)
        
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
            range_bins = np.where((self.freqs >= range_start) & (self.freqs < range_end))[0]
            
            if len(range_bins) > 0:
                bins_per_bar = max(1, len(range_bins) // num_bars)
                
                for i in range(num_bars):
                    start_idx = i * bins_per_bar
                    end_idx = (i + 1) * bins_per_bar if i < num_bars - 1 else len(range_bins)
                    
                    if start_idx < len(range_bins):
                        bar_bin_indices = range_bins[start_idx:end_idx]
                        if len(bar_bin_indices) > 0:
                            band_indices.append(bar_bin_indices)
                            freq_edges.append(self.freqs[bar_bin_indices[-1]])
        
        if len(band_indices) > total_bars:
            band_indices = band_indices[:total_bars]
            freq_edges = freq_edges[:total_bars + 1]
        
        self.bars = len(band_indices)
        
        self.freq_edges = np.array(freq_edges)

        print(f"\n🎵 Musical perceptual frequency mapping created: {len(band_indices)} bars")
        print(f"Frequency range: {freq_min:.1f}Hz - {freq_max:.1f}Hz")
        print(f"FFT bin resolution: {fft_bin_width:.1f}Hz per bin")
        
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

        if self.buffer_pos + chunk_len <= len(self.ring_buffer):
            self.ring_buffer[self.buffer_pos : self.buffer_pos + chunk_len] = audio_data
        else:
            first_part = len(self.ring_buffer) - self.buffer_pos
            self.ring_buffer[self.buffer_pos :] = audio_data[:first_part]
            self.ring_buffer[: chunk_len - first_part] = audio_data[first_part:]

        self.buffer_pos = (self.buffer_pos + chunk_len) % len(self.ring_buffer)

        audio_buffer = np.zeros(FFT_SIZE_BASE, dtype=np.float32)
        if self.buffer_pos >= FFT_SIZE_BASE:
            audio_buffer[:] = self.ring_buffer[self.buffer_pos - FFT_SIZE_BASE : self.buffer_pos]
        elif self.buffer_pos > 0:
            audio_buffer[-self.buffer_pos :] = self.ring_buffer[: self.buffer_pos]
            audio_buffer[: -self.buffer_pos] = self.ring_buffer[
                -(FFT_SIZE_BASE - self.buffer_pos) :
            ]

        if self.frame_counter % 2 == 0:  # Process every other frame
            multi_fft_results = self.multi_fft.process_multi_resolution(audio_data, self.psychoacoustic_enabled)
            combined_magnitude = self.combine_multi_resolution_results(multi_fft_results)
            self.cached_multi_fft_results = multi_fft_results
            self.cached_combined_magnitude = combined_magnitude
        else:
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
        unified_freqs = np.fft.rfftfreq(FFT_SIZE_BASE, 1 / SAMPLE_RATE)
        unified_magnitude = np.zeros_like(unified_freqs)

        freq_arrays = self.multi_fft.get_frequency_arrays()

        for config_idx, magnitude in multi_fft_results.items():
            config_freqs = freq_arrays[config_idx]
            config = self.multi_fft.fft_configs[config_idx]

            freq_range = config["range"]

            for i, freq in enumerate(unified_freqs):
                if freq_range[0] <= freq <= freq_range[1]:
                    nearest_idx = np.argmin(np.abs(config_freqs - freq))
                    if nearest_idx < len(magnitude):
                        weight = config["weight"]
                        unified_magnitude[i] = max(
                            unified_magnitude[i], magnitude[nearest_idx] * weight
                        )

        return unified_magnitude

    def process_frame(self):
        """Process one frame with enhanced professional analysis"""
        audio_data = self.capture.get_audio_data()
        if audio_data is None:
            self.bar_heights *= 0.9  # Quick decay when no audio
            return False

        start = time.perf_counter()
        self.frame_counter += 1
        
        raw_audio_data = audio_data.copy()
        
        audio_data = audio_data * self.input_gain
        
        if self.auto_gain_enabled and hasattr(self, 'lufs_info'):
            current_lufs = self.lufs_info.get('short_term', -100)
            if current_lufs > -100:  # Valid LUFS reading
                gain_adjustment = 10 ** ((self.target_lufs - current_lufs) / 20)
                self.gain_history.append(gain_adjustment)
                if len(self.gain_history) > 30:  # 0.5 seconds of history
                    smooth_gain = np.median(list(self.gain_history))
                    smooth_gain = np.clip(smooth_gain, 0.5, 2.0)
                    self.input_gain *= smooth_gain
                    self.input_gain = np.clip(self.input_gain, 1.0, 16.0)

        fft_results = self.process_audio_chunk(audio_data)
        combined_fft = fft_results["combined"]

        band_values = self._vectorized_band_mapping(combined_fft)

        if self.freq_compensation_enabled:
            band_values = self.apply_frequency_compensation(band_values)

        if self.normalization_enabled:
            max_val = np.max(band_values)
            if max_val > 0:
                band_values = band_values / max_val

        analysis_skip = getattr(self, 'analysis_skip_frames', 2)
        
            voice_info = self.voice_detector.detect_voice_realtime(audio_data)
            self.voice_info = voice_info
        else:
            voice_info = getattr(self, 'voice_info', self.voice_info)  # Use cached result
            
        if self.frame_counter % (analysis_skip + 1) == 0:
            drum_info = self.drum_detector.process_audio(combined_fft, band_values)
            self.drum_info = drum_info
        else:
            drum_info = getattr(self, 'drum_info', self.drum_info)  # Use cached result
        
        if self.adaptive_allocation_enabled and self.frame_counter % 10 == 0:
            content_type = self.content_detector.analyze_content(voice_info, band_values, self.freq_starts, self.freq_ends)
            new_allocation = self.content_detector.get_allocation_for_content(content_type)
            
            if content_type != self.current_content_type or abs(new_allocation - self.current_allocation) > 0.05:
                self.current_content_type = content_type
                self.current_allocation = new_allocation

            self.professional_meters_panel.update(audio_data)
            panel_results = self.professional_meters_panel.get_results()
            self.lufs_info = panel_results['lufs']
            self.transient_info = panel_results['transient']
        else:
            pass
            
        if self.show_harmonic_analysis and self.frame_counter % 5 == 0:
            self.harmonic_analysis_panel.update(combined_fft, self.freqs)
            harmonic_info = self.harmonic_analyzer.detect_harmonic_series(combined_fft, self.freqs)
            self.harmonic_info = harmonic_info
        else:
            harmonic_info = getattr(self, 'harmonic_info', self.harmonic_info)
            
        if self.show_professional_meters and self.frame_counter % 6 == 0:
            transient_info = self.transient_analyzer.analyze_transients(audio_data)
            self.transient_info = transient_info
        else:
            transient_info = getattr(self, 'transient_info', self.transient_info)
            
        if self.show_room_analysis and self.frame_counter % 8 == 0:
            room_modes = self.room_analyzer.detect_room_modes(combined_fft, self.freqs)
            self.room_modes = room_modes
        else:
            room_modes = getattr(self, 'room_modes', self.room_modes)
            
        if self.show_pitch_detection and self.frame_counter % 3 == 0:
            pitch_window_size = min(self.cepstral_analyzer.yin_window_size * 2, len(self.ring_buffer))
            if self.buffer_pos >= pitch_window_size:
                pitch_audio = self.ring_buffer[self.buffer_pos - pitch_window_size : self.buffer_pos].copy()
            else:
                pitch_audio = np.zeros(pitch_window_size, dtype=np.float32)
                if self.buffer_pos > 0:
                    pitch_audio[-self.buffer_pos:] = self.ring_buffer[:self.buffer_pos]
                    pitch_audio[:-self.buffer_pos] = self.ring_buffer[-(pitch_window_size - self.buffer_pos):]
                else:
                    pitch_audio = self.ring_buffer[-pitch_window_size:]
            
            self.pitch_detection_panel.update(pitch_audio)
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
            
        if self.show_chromagram and self.frame_counter % 4 == 0:
            self.chromagram_panel.update(combined_fft, self.freqs)
            chromagram_info = self.chromagram_panel.get_results()
        else:
            chromagram_info = self.chromagram_info
            
        if self.show_genre_classification and self.frame_counter % 5 == 0:
            self.genre_classification_panel.update(
                combined_fft, audio_chunk, drum_info, harmonic_info
            )
            genre_info = self.genre_classification_panel.get_results()
        else:
            genre_info = self.genre_info

        if drum_info["kick"]["kick_detected"]:
            self.kick_flash_time = time.time()
        if drum_info["snare"]["snare_detected"]:
            self.snare_flash_time = time.time()
        if voice_info["has_voice"] and voice_info["voice_confidence"] > 0.7:
            self.voice_flash_time = time.time()
        if voice_info.get("is_singing", False):
            self.singing_flash_time = time.time()

        self.drum_info = drum_info
        self.voice_info = voice_info
        self.lufs_info = lufs_info
        self.harmonic_info = harmonic_info
        self.transient_info = transient_info
        self.room_modes = room_modes
        self.pitch_info = pitch_info  # OMEGA: Store pitch detection results
        self.chromagram_info = chromagram_info  # OMEGA-1: Store chromagram results
        self.genre_info = genre_info  # OMEGA-2: Store genre classification results

        if self.smoothing_enabled:
            self._vectorized_smoothing(band_values, drum_info, voice_info)
        else:
            self.bar_heights = band_values
        
        if self.frame_counter % 4 == 0:
            self.bass_zoom_panel.update(audio_data, drum_info)
        
        self.last_process_time = time.perf_counter() - start
        
        if self.show_vu_meters:
            self.vu_meters_panel.update(raw_audio_data, self.last_process_time)
        
        if self.dynamic_quality_enabled:
            latency_ms = self.last_process_time * 1000
            current_time = time.time()
            time_since_switch = current_time - self.last_quality_switch_time
            
            if time_since_switch >= self.quality_mode_hold_time:
                if self.quality_mode == "quality" and latency_ms > self.performance_threshold:
                    self.quality_mode = "performance"
                    self._switch_to_performance_mode()
                    self.last_quality_switch_time = current_time
                    print(f"⚡ Switched to performance mode (latency: {latency_ms:.1f}ms)")
                
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
                if center_freq < 60:
                    gain = 0.15   # Aggressive reduction for sub-bass
                elif center_freq < 100:
                    gain = 0.15    # Aggressive reduction for bass
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
                    gain = 3.0    # Moderate boost for highs
                elif center_freq < 12000:
                    gain = 0.8    # Slight reduction for upper highs
                else:
                    gain = 0.5    # Moderate reduction for air

                if self.midrange_boost_enabled and 1000 <= center_freq <= 6000:
                    gain *= self.midrange_boost_factor

                gains[i] = gain

        return gains

    def _setup_optimized_band_mapping(self):
        """Pre-compute band mapping arrays for faster processing"""
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
        
        self.band_starts = np.array(self.band_starts, dtype=np.int32)
        self.band_ends = np.array(self.band_ends, dtype=np.int32)
        self.band_counts = np.array(self.band_counts, dtype=np.float32)
    
    def _vectorized_band_mapping(self, fft_data: np.ndarray) -> np.ndarray:
        """Optimized band mapping using pre-computed indices"""
        band_values = np.zeros(self.bars, dtype=np.float32)
        
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
        
        for i in range(self.bars):
            freq_range = self.get_frequency_range_for_bar(i)
            if freq_range[0] > 0:
                for freq in range(int(freq_range[0]), int(freq_range[1]) + 1, 10):
                    self.freq_to_bar[freq] = i
                    
        formant_freqs = [700, 800, 1220, 1400, 2600, 2800, 3400, 3600]
        for freq in formant_freqs:
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
        lookup_freq = int(round(freq / 10) * 10)
        return self.freq_to_bar.get(lookup_freq, 0)
    
    def _setup_bass_frequency_mapping(self):
        """Create frequency mapping for bass detail panel using actual FFT resolution"""
        bass_fft_size = 8192  # Larger FFT for better bass resolution
        bass_freqs = np.fft.rfftfreq(bass_fft_size, 1 / SAMPLE_RATE)
        
        bass_min_freq = 20.0
        bass_max_freq = 200.0
        valid_bins = np.where((bass_freqs >= bass_min_freq) & (bass_freqs <= bass_max_freq))[0]
        
        if len(valid_bins) == 0:
            self.bass_freq_ranges = [(20, 200)]
            self.bass_bin_mapping = [[]]
            self.bass_detail_bars = 1
            return
        
        self.bass_freq_ranges = []
        self.bass_bin_mapping = []
        
        bins_per_bar = max(1, len(valid_bins) // 31)  # Target ~31 bars for good visual density
        
        for i in range(0, len(valid_bins), bins_per_bar):
            end_idx = min(i + bins_per_bar, len(valid_bins))
            bin_group = valid_bins[i:end_idx]
            
            if len(bin_group) > 0:
                f_start = bass_freqs[bin_group[0]]
                f_end = bass_freqs[bin_group[-1]]
                
                if len(self.bass_freq_ranges) > 0:
                    f_start = max(f_start, self.bass_freq_ranges[-1][1])
                
                self.bass_freq_ranges.append((f_start, f_end))
                self.bass_bin_mapping.append(bin_group)
        
        self.bass_detail_bars = len(self.bass_freq_ranges)
        
        self.bass_bar_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_timestamps = np.zeros(self.bass_detail_bars, dtype=np.float64)
    
    def _vectorized_smoothing(self, band_values: np.ndarray, drum_info: Dict, voice_info: Dict):
        """Vectorized smoothing for better performance"""
        attack = np.zeros(self.bars, dtype=np.float32)
        release = np.zeros(self.bars, dtype=np.float32)
        
        ultra_low_mask = self.freq_starts <= 100
        low_mask = (self.freq_starts > 100) & (self.freq_starts <= 500)
        mid_mask = (self.freq_starts > 500) & (self.freq_starts <= 3000)
        high_mask = self.freq_starts > 3000
        
        kick_detected = drum_info["kick"]["kick_detected"]
        snare_detected = drum_info["snare"]["snare_detected"]
        has_voice = voice_info["has_voice"]
        
        attack[ultra_low_mask] = 0.98 if kick_detected else 0.75
        release[ultra_low_mask] = 0.06
        
        attack[low_mask] = 0.95 if (kick_detected or snare_detected) else 0.80
        release[low_mask] = 0.10
        
        voice_attack = 0.95 if has_voice else 0.85  # Faster when voice detected
        attack[mid_mask] = voice_attack
        release[mid_mask] = 0.15
        
        attack[high_mask] = 0.75
        release[high_mask] = 0.25
        
        rising_mask = band_values > self.bar_heights
        falling_mask = ~rising_mask
        
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
    


    
    def _switch_to_performance_mode(self):
        """Switch to performance mode for lower latency"""
        self.target_bars = 512  # Reduce from 698
        self.skip_expensive_analysis = True
        self.analysis_skip_frames = 4  # Was 2
        
    def _switch_to_quality_mode(self):
        """Switch back to quality mode"""
        self.target_bars = 698
        self.skip_expensive_analysis = False
        self.analysis_skip_frames = 2  # Back to normal

    def show_help_controls(self):
        """Display all available keyboard controls"""
        print("\n" + "="*90)
        print("🎛️  PROFESSIONAL AUDIO ANALYZER v4.1 OMEGA-2 - COMPLETE KEYBOARD CONTROLS")
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
                freq_start = self.freqs[indices[0]]
                freq_end = self.freqs[indices[-1]]
                
                if bar_index == 0:
                    freq_start = max(20.0, freq_start)

                if len(indices) == 1:
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
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_width = self.width - meter_width - vu_width - self.left_border_width - 10
        bar_width = spectrum_width / self.bars
        
        if frequency <= 20:
            return self.left_border_width
        if frequency >= 20000:
            return self.left_border_width + spectrum_width
        
        if hasattr(self, 'freq_edges'):
            bar_idx = -1
            for i in range(len(self.freq_edges) - 1):
                if self.freq_edges[i] <= frequency < self.freq_edges[i + 1]:
                    bar_idx = i
                    break
            
            if bar_idx == -1:
                if frequency >= self.freq_edges[-1]:
                    bar_idx = len(self.freq_edges) - 2
                else:
                    differences = np.abs(self.freq_edges[:-1] - frequency)
                    bar_idx = np.argmin(differences)
            
            freq_start = self.freq_edges[bar_idx]
            freq_end = self.freq_edges[bar_idx + 1]
            
            if freq_end > freq_start:
                log_position = (np.log10(frequency) - np.log10(freq_start)) / (np.log10(freq_end) - np.log10(freq_start))
                log_position = np.clip(log_position, 0, 1)
            else:
                log_position = 0.5
            
            bar_start_x = self.left_border_width + bar_idx * bar_width
            
            return bar_start_x + log_position * bar_width
        
        log_freq = np.log10(frequency)
        log_min = np.log10(20)
        log_max = np.log10(20000)
        normalized_pos = (log_freq - log_min) / (log_max - log_min)
        return self.left_border_width + normalized_pos * spectrum_width

    def update_peak_holds(self):
        """Update peak hold indicators with time-based decay"""
        current_time = time.time()
        num_bars = min(len(self.bar_heights), len(self.peak_hold_bars), len(self.peak_timestamps))
        
        for i in range(num_bars):
            if self.bar_heights[i] > self.peak_hold_bars[i] + 0.02:  # Small threshold to prevent jitter
                self.peak_hold_bars[i] = self.bar_heights[i]
                self.peak_timestamps[i] = current_time
            
            if self.peak_timestamps[i] > 0:  # Has a valid timestamp
                time_since_peak = current_time - self.peak_timestamps[i]
                
                if time_since_peak > self.peak_hold_time:
                    decay_time = time_since_peak - self.peak_hold_time
                    decay_factor = 0.5 ** (decay_time / 0.3)
                    
                    new_value = self.peak_hold_bars[i] * decay_factor
                    
                    if new_value < 0.01:
                        self.peak_hold_bars[i] = 0.0
                        self.peak_timestamps[i] = 0.0  # Mark as cleared
                    else:
                        self.peak_hold_bars[i] = new_value
            
            elif self.peak_hold_bars[i] > 0 and self.peak_timestamps[i] == 0:
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

        meter_height = 200
        meter_width = 25  # Slightly wider for better visibility
        x_pos = (self.left_border_width - meter_width) // 2  # Center in left border
        y_pos = self.height // 2 - meter_height // 2

        pygame.draw.rect(self.screen, (30, 30, 40), (x_pos, y_pos, meter_width, meter_height))
        pygame.draw.rect(self.screen, (100, 100, 120), (x_pos, y_pos, meter_width, meter_height), 1)

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

        label_text = self.font_tiny.render("SUB", True, (150, 150, 170))
        self.screen.blit(label_text, (x_pos - 5, y_pos - 20))

        if self.sub_bass_warning_active:
            warning_text = self.font_small.render("SUB!", True, (255, 100, 100))
            self.screen.blit(warning_text, (x_pos - 10, y_pos - 45))

    def draw_adaptive_allocation_indicator(self):
        """Draw indicator for adaptive frequency allocation"""
        indicator_width = 150
        indicator_height = 60
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        x_pos = self.width - meter_width - vu_width - indicator_width - 20
        y_pos = 80
        
        bg_color = (20, 25, 35) if self.adaptive_allocation_enabled else (35, 25, 20)
        pygame.draw.rect(self.screen, bg_color, (x_pos, y_pos, indicator_width, indicator_height))
        pygame.draw.rect(self.screen, (80, 80, 100), (x_pos, y_pos, indicator_width, indicator_height), 1)
        
        status_color = (100, 255, 150) if self.adaptive_allocation_enabled else (150, 150, 150)
        status_text = "ADAPTIVE" if self.adaptive_allocation_enabled else "FIXED"
        status_surface = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surface, (x_pos + 5, y_pos + 5))
        
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

            pygame.draw.line(
                self.screen, color + (128,), (x_pos, spectrum_top), (x_pos, spectrum_bottom), 2
            )

            label_surface = self.font_tiny.render(label, True, color)
            self.screen.blit(label_surface, (x_pos + 5, spectrum_top + 5))

    def detect_and_display_room_modes(self):
        """Detect potential room modes in the bass region"""
        current_time = time.time()
        room_mode_threshold = 0.6
        min_duration = 0.3  # 300ms minimum

        for i in range(min(150, self.bars)):  # Check first 150 bars (roughly 20-200Hz)
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2

            if 30 <= center_freq <= 300 and self.bar_heights[i] > room_mode_threshold:
                if center_freq not in self.sustained_peaks:
                    self.sustained_peaks[center_freq] = (
                        self.bar_heights[i],
                        current_time,
                        int(self.freq_to_x_position(center_freq)),
                    )
                else:
                    magnitude, start_time, x_pos = self.sustained_peaks[center_freq]
                    self.sustained_peaks[center_freq] = (
                        max(magnitude, self.bar_heights[i]),
                        start_time,
                        x_pos,
                    )

        self.room_mode_candidates = []
        to_remove = []

        for freq, (magnitude, start_time, x_pos) in self.sustained_peaks.items():
            duration = current_time - start_time

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

            if duration > min_duration:
                self.room_mode_candidates.append((freq, magnitude, duration, x_pos))

        for freq in to_remove:
            del self.sustained_peaks[freq]

        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50
        for freq, magnitude, duration, x_pos in self.room_mode_candidates:
            pygame.draw.circle(self.screen, (255, 100, 100, 180), (x_pos, spectrum_bottom + 20), 8)

            mode_text = self.font_tiny.render(f"{freq:.1f}Hz", True, (255, 150, 150))
            self.screen.blit(mode_text, (x_pos - 25, spectrum_bottom + 35))

    def draw_enhanced_analysis_grid(self):
        """Draw enhanced grid lines for easier reading"""
        if not self.show_frequency_grid:
            return

        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        spectrum_left = self.left_border_width
        spectrum_right = self.width - meter_width - vu_width - 10
        spectrum_top = 280 + 10  # Header height + gap
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        spectrum_bottom = self.height - bass_zoom_height - 50

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
                

    def draw_peak_hold_indicators(self):
        """Draw peak hold lines for all frequency bars (mirrored for spectrum)"""
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        vis_width = self.width - meter_width - vu_width - self.left_border_width - 10
        
        header_height = 280
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        vis_height = self.height - header_height - bass_zoom_height - 50
        vis_start_x = self.left_border_width
        vis_start_y = header_height + 10
        
        center_y = vis_start_y + vis_height // 2
        max_bar_height = (vis_height // 2) - 10
        bar_width = vis_width / self.bars

        num_bars_to_draw = min(len(self.peak_hold_bars), self.bars)
        
        for i in range(num_bars_to_draw):
            peak = self.peak_hold_bars[i]
            if peak > 0.05:  # Only draw significant peaks
                x = vis_start_x + i * bar_width
                clamped_peak = min(peak, 1.0)
                peak_height = int(clamped_peak * max_bar_height)
                
                upper_y = center_y - peak_height
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255, 180),
                    (int(x), upper_y),
                    (int(x + bar_width - 1), upper_y),
                    1,
                )
                
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
        
        pitch = self.voice_info.get("pitch", 150)
        formants = female_formants if pitch > 200 else male_formants
        
        colors = {
            "F1": (255, 100, 100),  # Red for F1
            "F2": (100, 255, 100),  # Green for F2  
            "F3": (100, 100, 255),  # Blue for F3
            "F4": (255, 255, 100)   # Yellow for F4
        }
        
        for formant_name, (freq, description) in formants.items():
            bar_pos = self.get_bar_for_frequency(freq)
            
            if bar_pos >= 0:  # Valid bar position (0 is valid)
                bar_width = vis_width / self.bars
                x = vis_start_x + int(bar_pos * bar_width)
                color = colors[formant_name]
                
                pygame.draw.line(
                    self.screen, color,
                    (x, vis_start_y), 
                    (x, vis_start_y + vis_height),
                    2
                )
                
                label_text = f"{formant_name}: {freq}Hz"
                label_surface = self.font_tiny.render(label_text, True, color)
                label_y = vis_start_y + 10 + (list(formants.keys()).index(formant_name) * 15)
                self.screen.blit(label_surface, (x + 5, label_y))
                
                current_bar_height = self.bar_heights[bar_pos] if bar_pos < len(self.bar_heights) else 0
                if current_bar_height > 0.005:  # Lower threshold
                    max_bar_height = (vis_height // 2) - 10
                    clamped_height = min(current_bar_height, 1.0)
                    height = int(clamped_height * max_bar_height)
                    marker_y = center_y - height - 8
                    
                    pygame.draw.circle(self.screen, color, (x + 3, marker_y), 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + 3, marker_y), 6, 2)
                    pygame.draw.circle(self.screen, (255, 255, 255), (x + 3, marker_y), 2)

    def draw_technical_overlay(self):
        """Draw technical analysis overlay"""
        if not self.show_technical_overlay:
            return

        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 320  # Match harmonic panel width
        overlay_height = 350
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
            overlay_y = 280 + 10 + 200 + 10 + 320 + 10  # Updated for new pitch panel height
        else:
            overlay_y = 280 + 10 + 200 + 10

        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (overlay_x, overlay_y))

        pygame.draw.rect(
            self.screen, (90, 90, 110), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )

        y_offset = overlay_y + 20
        line_height = 25

        title_text = self.font_medium.render("Technical Analysis", True, (200, 200, 220))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 40

        bass_energy = self.calculate_band_energy(20, 250)
        mid_energy = self.calculate_band_energy(250, 2000)
        high_energy = self.calculate_band_energy(2000, 20000)

        self.draw_overlay_text(f"Bass (20-250Hz): {bass_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height
        self.draw_overlay_text(f"Mids (250-2kHz): {mid_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height
        self.draw_overlay_text(f"Highs (2k-20kHz): {high_energy:.1%}", overlay_x + 20, y_offset)
        y_offset += line_height * 1.5

        tilt = self.calculate_spectral_tilt()
        tilt_description = "Bright" if tilt > 0 else "Dark" if tilt < -3 else "Balanced"
        self.draw_overlay_text(
            f"Spectral Tilt: {tilt:.1f}dB/oct ({tilt_description})", overlay_x + 20, y_offset
        )
        y_offset += line_height * 1.5

        crest_factor = self.calculate_crest_factor()
        self.draw_overlay_text(f"Crest Factor: {crest_factor:.1f}dB", overlay_x + 20, y_offset)
        y_offset += line_height

        dynamic_range = max(self.bar_heights) - np.mean(self.bar_heights[self.bar_heights > 0.1])
        self.draw_overlay_text(f"Dynamic Range: {dynamic_range:.1f}dB", overlay_x + 20, y_offset)
        y_offset += line_height * 1.5

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
            
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 320  # Match harmonic panel width
        overlay_height = 220  # Slightly taller for gender detection info
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
            overlay_y = 280 + 10 + 200 + 10 + 320 + 10 + 350 + 10  # Updated for new pitch panel height
        else:
            overlay_y = 280 + 10 + 200 + 10 + 350 + 10
        
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        pygame.draw.rect(
            self.screen, (80, 90, 110), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )
        
        y_offset = overlay_y + 20
        line_height = 25
        
        title_text = self.font_medium.render("Voice Detection", True, (200, 200, 220))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 40
        
        has_voice = self.voice_info.get("has_voice", False)
        pitch = self.voice_info.get("pitch", 0)
        
        total_energy = 0.0
        voice_band_count = 0
        
        for i in range(min(self.bars, len(self.bar_heights))):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2
            
            if 80 <= center_freq <= 4000:
                speech_energy += self.bar_heights[i]
                voice_band_count += 1
            total_energy += self.bar_heights[i]
        
        if voice_band_count > 0 and total_energy > 0:
            speech_ratio = speech_energy / total_energy
            pitch_confidence = 1.0 if pitch > 80 else 0.0
            confidence = (speech_ratio * 0.6 + pitch_confidence * 0.4)
            if has_voice and confidence < 0.3:
                confidence = 0.3
        else:
            confidence = 0.0
        
        self.voice_info["confidence"] = confidence
        
        if has_voice:
            status_color = (100, 255, 100)
            status_text = "VOICE DETECTED"
        else:
            status_color = (150, 150, 150)
            status_text = "No Voice"
            
        status_surf = self.font_small.render(status_text, True, status_color)
        self.screen.blit(status_surf, (overlay_x + 20, y_offset))
        y_offset += line_height
        
        conf_text = f"Confidence: {confidence:.1%}"
        conf_color = (
            (255, 100, 100) if confidence < 0.3 else
            (255, 200, 100) if confidence < 0.7 else
            (100, 255, 100)
        )
        conf_surf = self.font_small.render(conf_text, True, conf_color)
        self.screen.blit(conf_surf, (overlay_x + 20, y_offset))
        y_offset += line_height
        
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
                
                y_offset += 5  # Small spacing
                if pitch > 0:
                    
                    high_energy = 0.0  # 1000-4000 Hz
                    
                    for i in range(min(self.bars, len(self.bar_heights))):
                        freq_start, freq_end = self.get_frequency_range_for_bar(i)
                        center_freq = (freq_start + freq_end) / 2
                        
                        if 80 <= center_freq <= 250:
                            low_energy += self.bar_heights[i]
                        elif 1000 <= center_freq <= 4000:
                            high_energy += self.bar_heights[i]
                    
                    spectral_tilt = low_energy / (high_energy + 0.001)
                    
                        gender = "Male"
                        tilt_factor = min(1.0, spectral_tilt / 2.0)  # Males typically have tilt > 2
                        gender_confidence = min(1.0, (160 - pitch) / 75 * 0.7 + tilt_factor * 0.3)
                        gender_color = (100, 150, 255)  # Blue
                    elif pitch > 190:  # Lowered threshold from 200
                        gender = "Female"
                        tilt_factor = min(1.0, (2.0 - spectral_tilt) / 2.0)
                        gender_confidence = min(1.0, (pitch - 190) / 65 * 0.7 + tilt_factor * 0.3)
                        gender_color = (255, 150, 200)  # Pink
                    else:
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
                    
                    if self.show_advanced_info:
                        tilt_text = f"Spectral tilt: {spectral_tilt:.2f}"
                        tilt_surf = self.font_tiny.render(tilt_text, True, (150, 150, 170))
                        self.screen.blit(tilt_surf, (overlay_x + 40, y_offset))
                        y_offset += 20
        
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
    
            return
            
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0
        overlay_width = 320  # Increased by 40px for better layout
        overlay_height = 360  # Increased by 60px for fixed graph position
        overlay_x = self.width - meter_width - vu_width - overlay_width - 10
        
        
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(230)
        overlay.fill((25, 20, 35))  # Slightly purple tint
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        pygame.draw.rect(
            self.screen, (110, 90, 140), (overlay_x, overlay_y, overlay_width, overlay_height), 2
        )
        
        y_offset = overlay_y + 20
        line_height = 26 # from 22 to 26
        
        title_text = self.font_medium.render("OMEGA Pitch Detection", True, (220, 200, 255))
        self.screen.blit(title_text, (overlay_x + 20, y_offset))
        y_offset += 35
        
        pitch = self.pitch_info.get('pitch', 0.0)
        confidence = self.pitch_info.get('confidence', 0.0)
        note = self.pitch_info.get('note', '')
        octave = self.pitch_info.get('octave', 0)
        cents = self.pitch_info.get('cents_offset', 0)
        stability = self.pitch_info.get('stability', 0.0)
        
        if pitch > 0 and confidence > 0.3:
            pitch_color = (
                (255, 150, 100) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            
            freq_text = f"Frequency: {pitch:.1f} Hz"
            freq_surf = self.font_small.render(freq_text, True, pitch_color)
            self.screen.blit(freq_surf, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            if note:
                note_text = f"Note: {note}{octave}"
                if cents != 0:
                    note_text += f" {cents:+d}¢"
                note_surf = self.font_medium.render(note_text, True, (200, 220, 255))
                self.screen.blit(note_surf, (overlay_x + 20, y_offset))
                y_offset += line_height + 5
            
            conf_text = f"Confidence: {confidence:.0%}"
            conf_surf = self.font_small.render(conf_text, True, (180, 180, 200))
            self.screen.blit(conf_surf, (overlay_x + 20, y_offset))
            y_offset += line_height
            
            bar_width = overlay_width - 40
            bar_height = 8
            bar_x = overlay_x + 20
            bar_y = y_offset
            
            pygame.draw.rect(self.screen, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height))
            fill_width = int(bar_width * confidence)
            conf_color = (
                (200, 100, 100) if confidence < 0.5 else
                (200, 200, 100) if confidence < 0.7 else
                (100, 200, 100)
            )
            pygame.draw.rect(self.screen, conf_color, (bar_x, bar_y, fill_width, bar_height))
            pygame.draw.rect(self.screen, (100, 100, 120), (bar_x, bar_y, bar_width, bar_height), 1)
            y_offset += bar_height + 10
            
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
            no_pitch_text = "No pitch detected"
            no_pitch_surf = self.font_small.render(no_pitch_text, True, (150, 150, 150))
            self.screen.blit(no_pitch_surf, (overlay_x + 20, y_offset))
            y_offset += line_height + 10
        
        methods = self.pitch_info.get('methods', {})
        if methods:
            pygame.draw.line(
                self.screen, (80, 80, 100), 
                (overlay_x + 20, y_offset), 
                (overlay_x + overlay_width - 20, y_offset), 1
            )
            y_offset += 10
            
            methods_text = self.font_small.render("Detection Methods:", True, (180, 180, 200))
            self.screen.blit(methods_text, (overlay_x + 20, y_offset))
            y_offset += line_height
            
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
        
        if hasattr(self.cepstral_analyzer, 'pitch_history') and len(self.cepstral_analyzer.pitch_history) > 10:
            graph_height = 40
            graph_width = overlay_width - 40
            graph_x = overlay_x + 20
            graph_y = overlay_y + overlay_height - graph_height - 20  # 20px from bottom
            
            pygame.draw.rect(self.screen, (30, 30, 40), (graph_x, graph_y, graph_width, graph_height))
            
            history = list(self.cepstral_analyzer.pitch_history)
            if history:
                valid_pitches = [p for p in history if p > 0]
                if valid_pitches:
                    min_pitch = min(valid_pitches) * 0.9
                    max_pitch = max(valid_pitches) * 1.1
                    
                    points = []
                    for i, pitch in enumerate(history[-50:]):  # Last 50 values
                        if pitch > 0:
                            x = graph_x + int(i * graph_width / 50)
                            y = graph_y + graph_height - int((pitch - min_pitch) / (max_pitch - min_pitch) * graph_height)
                            points.append((x, y))
                    
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, (150, 150, 255), False, points, 2)
            
            pygame.draw.rect(self.screen, (80, 80, 100), (graph_x, graph_y, graph_width, graph_height), 1)
            
            hist_label = self.font_tiny.render("Pitch History", True, (150, 150, 170))
            self.screen.blit(hist_label, (graph_x, graph_y - 15))
                
    def freq_to_note(self, freq: float) -> str:
        """Convert frequency to musical note"""
        if freq <= 0:
            return None
            
        A4 = 440
        C0 = A4 * pow(2, -4.75)
        
        if freq < C0:
            return None
            
        h = 12 * np.log2(freq / C0)
        octave = int(h / 12)
        n = int(h % 12)
        
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        return f"{notes[n]}{octave}"
    

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
    


    def draw_help_menu(self):
        """Draw keyboard shortcuts help overlay"""
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(240)
        overlay.fill((20, 20, 30))
        self.screen.blit(overlay, (0, 0))
        
        col_width = 400
        start_x = 50
        start_y = 50
        line_height = 25
        section_gap = 35
        
        title_text = self.font_large.render("OMEGA-2 Audio Analyzer - Keyboard Shortcuts", True, (255, 255, 255))
        title_rect = title_text.get_rect(centerx=self.width // 2, y=start_y)
        self.screen.blit(title_text, title_rect)
        
        help_sections = [
            ("Display Controls", [
                ("ESC", "Exit analyzer"),
                ("S", "Save screenshot"),
                ("M", "Toggle professional meters"),
                ("H", "Toggle harmonic analysis"),
                ("R", "Toggle room analysis"),
                ("Z", "Toggle bass zoom window"),
                ("J", "Toggle VU meters"),
                ("V", "Toggle voice info"),
                ("F", "Toggle formants display"),
                ("A", "Toggle advanced info"),
                ("G", "Toggle frequency grid"),
                ("B", "Toggle band separators"),
                ("T", "Toggle technical overlay"),
            ]),
            ("Audio Processing", [
                ("Q", "Toggle frequency compensation"),
                ("W", "Toggle psychoacoustic weighting"),
                ("E", "Toggle normalization"),
                ("U", "Toggle smoothing"),
                ("Y", "Toggle dynamic quality mode"),
                ("I", "Toggle midrange boost"),
                ("O", "Increase midrange boost"),
                ("L", "Decrease midrange boost"),
            ]),
            ("OMEGA Features", [
                ("P", "Toggle pitch detection (OMEGA)"),
                ("K", "Toggle chromagram & key detection (OMEGA-1)"),
                ("N", "Toggle genre classification (OMEGA-2)"),
            ]),
            ("Input Controls", [
                ("+ / =", "Increase input gain"),
                ("-", "Decrease input gain"),
                ("0", "Reset gain to +12dB"),
                ("C", "Store reference / Toggle A/B comparison"),
                ("X", "Clear reference spectrum"),
                ("D", "Print debug info"),
                (";", "Toggle adaptive allocation"),
            ]),
            ("Window Presets", [
                ("1", "1400x900 (Compact)"),
                ("2", "1800x1000 (Standard)"),
                ("3", "2200x1200 (Wide)"),
                ("4", "2600x1400 (Ultra-Wide)"),
                ("5", "3000x1600 (Cinema)"),
                ("6", "3400x1800 (Studio)"),
                ("7", "1920x1080 (Full HD)"),
                ("8", "2560x1440 (2K)"),
                ("9", "3840x2160 (4K)"),
                ("SHIFT+0", "Toggle fullscreen"),
            ]),
        ]
        
        total_items = sum(len(section[1]) + 2 for section in help_sections)  # +2 for title and gap
        items_per_col = (total_items + 2) // 3  # Distribute across 3 columns
        
        col = 0
        x = start_x
        y = start_y + 80
        items_in_col = 0
        
        for section_title, shortcuts in help_sections:
            if items_in_col > 0 and items_in_col + len(shortcuts) + 2 > items_per_col:
                col += 1
                x = start_x + col * col_width
                y = start_y + 80
                items_in_col = 0
            
            section_surf = self.font_medium.render(section_title, True, (200, 200, 255))
            self.screen.blit(section_surf, (x, y))
            y += line_height + 5
            items_in_col += 1
            
            for key, desc in shortcuts:
                key_surf = self.font_small.render(f"{key:>8}", True, (255, 200, 100))
                desc_surf = self.font_small.render(f" - {desc}", True, (200, 200, 200))
                self.screen.blit(key_surf, (x, y))
                self.screen.blit(desc_surf, (x + 80, y))
                y += line_height
                items_in_col += 1
            
            y += section_gap - line_height
            items_in_col += 1
        
        footer_text = self.font_medium.render("Press ? or / to close help", True, (150, 150, 200))
        footer_rect = footer_text.get_rect(centerx=self.width // 2, bottom=self.height - 30)
        self.screen.blit(footer_text, footer_rect)

    def draw_ab_comparison(self):
        """Draw A/B comparison between current and reference spectrum"""
        if self.reference_spectrum is None:
            return

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

                pygame.draw.line(
                    self.screen,
                    (200, 200, 200, 150),
                    (int(x), center_y - ref_height),
                    (int(x + bar_width), center_y - ref_height),
                    2,
                )

                pygame.draw.line(
                    self.screen,
                    (150, 150, 150, 120),
                    (int(x), center_y),
                    (int(x + bar_width), center_y + ref_height),
                    2,
                )

        if len(self.bar_heights) == len(self.reference_spectrum):
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

                label_surface = self.font_small.render("A/B COMPARISON", True, (200, 200, 220))
                self.screen.blit(label_surface, (self.width - 200, 120))

    def print_debug_output(self):
        """Print detailed debug information to terminal"""
        print("\n" + "=" * 80)
        print(f"PROFESSIONAL AUDIO ANALYZER V4 - SPECTRUM SNAPSHOT - {self.bars} bars")
        print("Musical Perceptual Frequency Mapping")
        print("=" * 80)

        current_fps = self.fps_counter[-1] if self.fps_counter else 0
        avg_fps = np.mean(self.fps_counter) if len(self.fps_counter) > 0 else 0

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
        
        low_end_bars = int(self.bars * self.current_allocation)
        print(f"\n[PERCEPTUAL FREQUENCY MAPPING]")
        print(f"Mode: Musical Perceptual (Bark-scale inspired)")
        print(f"Distribution:")
        print(f"  Bass (20-250Hz): 25% of bars")
        print(f"  Low-mid (250-500Hz): 15% of bars") 
        print(f"  Mid (500-2kHz): 25% of bars")
        print(f"  High-mid (2k-6kHz): 20% of bars")
        print(f"  High (6k-20kHz): 15% of bars")

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

        height = 16
        width = 80  # Terminal width

        ascii_bars = []
        for _ in range(height):
            ascii_bars.append([" " for _ in range(width)])

        for pos in range(width):
            freq_position = pos / (width - 1)
            
            
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
            
            bar_height = self.bar_heights[best_bar_idx] if best_bar_idx < len(self.bar_heights) else 0
            bar_pixels = int(bar_height * height)

            for j in range(bar_pixels):
                if j < height:
                    if j == 0:
                        ascii_bars[height - 1 - j][pos] = "▄"
                    elif j == bar_pixels - 1 and bar_pixels < height:
                        ascii_bars[height - 1 - j][pos] = "▄"
                    else:
                        ascii_bars[height - 1 - j][pos] = "█"

        for row in ascii_bars:
            print("".join(row))

        print()
        print("-" * 80)
        
        
        
        freq_line = [' '] * 80
        freq_line[0:3] = list("20 ")
        freq_line[19:23] = list("250 ")
        freq_line[31:35] = list("500 ")
        freq_line[50:53] = list("2k ")
        freq_line[66:69] = list("6k ")
        freq_line[76:80] = list("20k ")
        
        print("".join(freq_line) + "Hz")
        print()

        non_zero_bars = self.bar_heights[self.bar_heights > 0.01]
        if len(non_zero_bars) > 0:
            max_val = np.max(self.bar_heights)
            avg_val = np.mean(non_zero_bars)
            min_val = np.min(non_zero_bars)
            print(f"Stats: Max={max_val:.2f}, Avg={avg_val:.2f}, Min={min_val:.2f}")

        for idx in range(len(self.bar_heights)):
            if self.bar_heights[idx] > 0.1:
                freq_start, freq_end = self.get_frequency_range_for_bar(idx)
                center_freq = (freq_start + freq_end) / 2

                freq_key = round(center_freq / 10) * 10

                if freq_key not in freq_peaks or self.bar_heights[idx] > freq_peaks[freq_key][1]:
                    freq_peaks[freq_key] = (center_freq, self.bar_heights[idx])

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
            
            if hasattr(self, 'ring_buffer'):
                audio_rms = np.sqrt(np.mean(self.ring_buffer**2))
                print(f"\nAudio Buffer RMS: {audio_rms:.6f}")
                print(f"Audio Buffer Max: {np.max(np.abs(self.ring_buffer)):.6f}")
                
            if hasattr(self, 'cepstral_analyzer'):
                print(f"YIN window size: {self.cepstral_analyzer.yin_window_size}")
                print(f"Pitch history length: {len(self.cepstral_analyzer.pitch_history)}")
        else:
            print("Pitch info not initialized")
            
        print()
        print("[GAP ANALYSIS]")
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
        
        print()
        print("[PROFESSIONAL METERING - ITU-R BS.1770-4]")
        if not hasattr(self, 'lufs_info') or not self.lufs_info:
            if hasattr(self, 'ring_buffer'):
                audio_for_lufs = self.ring_buffer[-SAMPLE_RATE:] if len(self.ring_buffer) >= SAMPLE_RATE else self.ring_buffer
                panel_results = self.professional_meters_panel.get_results()
                self.lufs_info = panel_results['lufs']
                self.transient_info = panel_results['transient']
        
        if hasattr(self, 'lufs_info') and self.lufs_info:
            lufs = self.lufs_info
            print(f"Momentary LUFS (400ms): {lufs['momentary']:+6.1f} LUFS")
            print(f"Short-term LUFS (3s):   {lufs['short_term']:+6.1f} LUFS")
            print(f"Integrated LUFS:        {lufs['integrated']:+6.1f} LU")
            print(f"Loudness Range (LRA):   {lufs['range']:6.1f} LU")
            
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

        if hasattr(self, "capture") and hasattr(self.capture, "chunk_size"):
            current_chunk = self.capture.chunk_size
            if dynamic_range_db < 20:
                print(f"Auto-tune: Consider increasing chunk size from {current_chunk}")
            elif dynamic_range_db > 40:
                print(f"Auto-tune: Consider decreasing chunk size from {current_chunk}")
            else:
                print(f"Auto-tune: Chunk size {current_chunk} is optimal")

        print()
        

        print("=" * 80)
        print("BASS DETAIL VISUALIZATION (20-200Hz)")
        print("=" * 80)

        bass_width = 80
        bass_freq_min = 20
        bass_freq_max = 200

        freq_markers = [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200]
        marker_line = [" "] * bass_width
        label_line = [" "] * bass_width

        for freq in freq_markers:
            pos = int(
                (np.log10(freq) - np.log10(bass_freq_min))
                / (np.log10(bass_freq_max) - np.log10(bass_freq_min))
                * (bass_width - 1)
            )
            if 0 <= pos < bass_width:
                marker_line[pos] = "|"
                if freq in [20, 40, 60, 100, 150, 200]:
                    label_str = str(freq)
                    if pos + len(label_str) > bass_width:
                        pos = max(0, bass_width - len(label_str))
                    for i, char in enumerate(label_str):
                        if pos + i < bass_width:
                            label_line[pos + i] = char

        bass_bars = []
        for row in range(10):  # 10 rows for visualization
            bass_bars.append([" "] * bass_width)

        for i in range(self.bars):
            freq_start, freq_end = self.get_frequency_range_for_bar(i)
            center_freq = (freq_start + freq_end) / 2

            if bass_freq_min <= center_freq <= bass_freq_max:
                pos = int(
                    (np.log10(center_freq) - np.log10(bass_freq_min))
                    / (np.log10(bass_freq_max) - np.log10(bass_freq_min))
                    * (bass_width - 1)
                )

                if 0 <= pos < bass_width:
                    height = int(self.bar_heights[i] * 10)

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

        for row in bass_bars:
            print("".join(row))

        print("".join(marker_line))
        print("".join(label_line) + " Hz")

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

            peak_idx = bass_bars_indices[np.argmax(bass_values)]
            freq_start, freq_end = self.get_frequency_range_for_bar(peak_idx)
            peak_freq = (freq_start + freq_end) / 2
            print(f"Peak bass frequency: {peak_freq:.1f}Hz")

        print()
        
        if self.show_vu_meters:
            print("=" * 80)
            print("VU METERS (0 VU = -20 dBFS)")
            print("=" * 80)
            
            left_db = self.vu_left_display if hasattr(self, 'vu_left_display') else -20.0
            right_db = self.vu_right_display if hasattr(self, 'vu_right_display') else -20.0
            left_peak = self.vu_left_peak_db if hasattr(self, 'vu_left_peak_db') else -20.0
            right_peak = self.vu_right_peak_db if hasattr(self, 'vu_right_peak_db') else -20.0
            
            meter_width = 60
            scale_min = -20.0
            scale_max = 3.0
            
            scale_line = [' '] * meter_width
            label_line = [' '] * meter_width
            
            scale_positions = [(-20, "-20"), (-10, "-10"), (-7, "-7"), (-5, "-5"), 
                             (-3, "-3"), (0, "0"), (+3, "+3")]
            
            for db_val, label in scale_positions:
                pos = int((db_val - scale_min) / (scale_max - scale_min) * (meter_width - 1))
                if 0 <= pos < meter_width:
                    scale_line[pos] = '|'
                    if len(label) <= 3:
                        for i, char in enumerate(label):
                            if pos + i < meter_width:
                                label_line[pos + i] = char
            
            def draw_vu_meter(name, value, peak):
                value_norm = (value - scale_min) / (scale_max - scale_min)
                peak_norm = (peak - scale_min) / (scale_max - scale_min)
                
                value_norm = max(0.0, min(1.0, value_norm))
                peak_norm = max(0.0, min(1.0, peak_norm))
                
                value_pos = int(value_norm * (meter_width - 1))
                peak_pos = int(peak_norm * (meter_width - 1))
                
                meter = [' '] * meter_width
                
                for i in range(value_pos + 1):
                    if value > 0 and i >= int((0 - scale_min) / (scale_max - scale_min) * (meter_width - 1)):
                        meter[i] = '█'  # Red zone
                    elif value > -3 and i >= int((-3 - scale_min) / (scale_max - scale_min) * (meter_width - 1)):
                        meter[i] = '▓'  # Yellow zone
                    else:
                        meter[i] = '▒'  # Green zone
                
                if 0 <= peak_pos < meter_width:
                    meter[peak_pos] = '┃'
                
                print(f"{name:5} [{(''.join(meter)).ljust(meter_width)}] {value:+5.1f} dB (peak: {peak:+5.1f} dB)")
            
            print("      " + "".join(scale_line))
            print("      " + "".join(label_line) + " dB")
            print()
            
            draw_vu_meter("LEFT", left_db, left_peak)
            draw_vu_meter("RIGHT", right_db, right_peak)
            
            if left_db > 0 or right_db > 0:
                print("\n⚠️  WARNING: Signal exceeding 0 VU!")
            elif left_db > -3 or right_db > -3:
                print("\n⚠️  Hot signal - watch levels")
            else:
                print("\n✅ Signal levels OK")
        
        print()


















    
        pygame.draw.rect(self.screen, (15, 25, 40), zoom_rect)
        pygame.draw.rect(self.screen, (60, 80, 120), zoom_rect, 2)

        title = self.font_small.render("BASS DETAIL (20-200 Hz)", True, (200, 220, 255))
        self.screen.blit(title, (x + 5, y + 5))

        scale_height = 35
        visualization_height = height - 40 - scale_height  # Title space + scale space

        if hasattr(self, 'bass_bar_values'):
            total_width = width - 20
            bar_width = total_width / self.bass_detail_bars  # Exact division
            max_height = visualization_height
            current_time = time.time()
            bar_bottom = y + height - scale_height - 5  # Leave space for scale

            for j in range(self.bass_detail_bars):
                amplitude = min(self.bass_bar_values[j], 1.0)
                freq_range = self.bass_freq_ranges[j]
                
                bar_height = min(int(amplitude * max_height), max_height)
                bar_x = x + 10 + j * bar_width
                bar_y = max(y + 25, bar_bottom - bar_height)
                
                freq_center = (freq_range[0] + freq_range[1]) / 2
                
                bass_min = 20.0
                bass_max = 200.0
                freq_position = (freq_center - bass_min) / (bass_max - bass_min)
                freq_position = max(0.0, min(1.0, freq_position))  # Clamp to 0-1
                
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

                if (
                    hasattr(self, "drum_info")
                    and self.drum_info["kick"].get("display_strength", 0) > 0.1
                    and freq_center <= 120
                ):
                    color = tuple(min(255, int(c * 1.3)) for c in color)

                actual_bar_width = int(bar_width) + 1 if j < self.bass_detail_bars - 1 else int(bar_width)
                pygame.draw.rect(
                    self.screen, color, (int(bar_x), bar_y, actual_bar_width, bar_height)
                )

                if j < len(self.bass_peak_values) and self.bass_peak_values[j] > 0.05:
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

            scale_y = y + height - scale_height
            pygame.draw.line(self.screen, (100, 100, 120), 
                           (x + 10, scale_y), 
                           (x + width - 10, scale_y), 2)
            
            scale_bg_rect = pygame.Rect(x + 5, scale_y - 2, width - 10, scale_height - 3)
            pygame.draw.rect(self.screen, (10, 15, 25), scale_bg_rect)
            
            key_freqs = [20, 30, 40, 60, 80, 100, 150, 200]
            for target_freq in key_freqs:
                log_pos = (np.log10(target_freq) - np.log10(20)) / (np.log10(200) - np.log10(20))
                if 0 <= log_pos <= 1:
                    label_x = x + 10 + log_pos * (width - 20)
                    
                    pygame.draw.line(self.screen, (150, 150, 160),
                                       (int(label_x), scale_y - 2),
                                       (int(label_x), scale_y + 6), 2)
                    
                    freq_text = f"{target_freq}"
                    freq_surf = self.font_tiny.render(freq_text, True, (200, 220, 240))
                    text_rect = freq_surf.get_rect(centerx=int(label_x), top=scale_y + 8)
                    self.screen.blit(freq_surf, text_rect)
            
            hz_surf = self.font_tiny.render("Hz", True, (200, 220, 240))
            self.screen.blit(hz_surf, (x + width - 30, scale_y + 8))

    def draw_frame(self):
        """Draw the enhanced professional visualization"""
        self.display.clear_screen()

        header_height = 280  # Doubled from 140 to give more room
        meter_width = self.meter_panel_width if self.show_professional_meters else 0
        bass_zoom_height = self.bass_zoom_height if self.show_bass_zoom else 0
        vu_width = self.vu_meter_width if self.show_vu_meters else 0

        header_rect = pygame.Rect(0, 0, self.width, header_height)
        pygame.draw.rect(self.screen, (18, 22, 32), header_rect)

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
        
        if self.show_vu_meters:
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
            
        avg_fps = 60.0
        if hasattr(self, "fps_counter") and len(self.fps_counter) > 0:
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            
        quality_mode = getattr(self, 'quality_mode', 'quality')
        
        header_data = {
            'title': 'Professional Audio Analyzer v4.1 OMEGA-2',
            'subtitle': 'Musical Perceptual Mapping • Professional Metering • Harmonic Analysis • Room Acoustics',
            'features': [
                '🎯 Peak Hold • ⚠️ Sub-Bass Monitor • 📊 Band Separators',
                '🔬 Technical Overlay • ⚖️ A/B Comparison • 📏 Analysis Grid',
                '🎛️ Gain: +/- keys • 0: Toggle Auto-gain • ESC: Exit'
            ],
            'audio_source': 'Professional Monitor',
            'sample_rate': SAMPLE_RATE,
            'bars': self.bars,
            'fft_info': 'Multi-FFT: 8192/4096/2048/1024',
            'latency': self.peak_latency,
            'fps': avg_fps,
            'quality_mode': quality_mode,
            'gain_db': 20 * np.log10(self.input_gain),
            'auto_gain': self.auto_gain_enabled,
            'active_features': active_features,
            'ui_scale': self.ui_scale
        }
        
        self.display.draw_header(header_data, header_height)

        current_time = time.time()

            meter_x = self.width - self.meter_panel_width - vu_width
            self.professional_meters_panel.draw(
                self.screen,
                meter_x,
                header_height,
                self.meter_panel_width,
                self.height - header_height - bass_zoom_height,
                self.ui_scale
            )
        
        if self.show_harmonic_analysis:
            harmonic_width = 320
            harmonic_x = self.width - meter_width - vu_width - harmonic_width - 10
            self.harmonic_analysis_panel.draw(self.screen, harmonic_x, header_height + 10, harmonic_width, 200, self.ui_scale)

        vis_width = self.width - meter_width - vu_width - self.left_border_width - 10
        vis_height = self.height - header_height - bass_zoom_height - 50
        vis_start_x = self.left_border_width
        vis_start_y = header_height + 10

        if self.show_vu_meters:
            vu_x = self.width - self.vu_meter_width
            vu_y = vis_start_y
            if self.show_bass_zoom:
                vu_height = (self.height - bass_zoom_height) - vu_y + bass_zoom_height
            else:
                vu_height = self.height - vu_y - 10
            
            self.vu_meters_panel.draw(
                self.screen,
                vu_x,
                vu_y,
                self.vu_meter_width,
                vu_height,
                self.ui_scale
            )

        if self.show_bass_zoom:
            bass_y = self.height - bass_zoom_height
            bass_width = spectrum_end_x - self.left_border_width
            
            self.bass_zoom_panel.draw(
                self.screen,
                self.left_border_width,
                bass_y,
                bass_width,
                bass_zoom_height - 10,
                self.ui_scale
            )

        center_y = vis_start_y + vis_height // 2
        max_bar_height = (vis_height // 2) - 10

        self.update_peak_holds()

            'spectrum_left': self.left_border_width,
            'spectrum_right': self.width - meter_width - vu_width - 10,
            'spectrum_top': vis_start_y,
            'spectrum_bottom': vis_start_y + vis_height,
            'center_y': center_y
        }
        self.display.draw_grid_and_labels(grid_params, self.ui_scale)
        
        
            'spectrum_top': vis_start_y,
            'spectrum_bottom': vis_start_y + vis_height,
            'spectrum_left': self.left_border_width,
            'spectrum_right': self.width - meter_width - vu_width - 10
        }
        self.display.draw_frequency_band_separators(band_params, self.freq_to_x_position, self.show_band_separators)
        
        

        vis_params = {
            'vis_start_x': vis_start_x,
            'vis_start_y': vis_start_y,
            'vis_width': vis_width,
            'vis_height': vis_height,
            'center_y': center_y,
            'max_bar_height': max_bar_height
        }
        self.display.draw_spectrum_bars(self.bar_heights, vis_params)
        
        """
        bar_width = vis_width / self.bars

        for i in range(self.bars):
            if self.bar_heights[i] > 0.01:
                clamped_height = min(self.bar_heights[i], 1.0)
                height = int(clamped_height * max_bar_height)
                x = vis_start_x + int(i * bar_width)
                width = max(1, int(bar_width))

                color = self.colors[i]

                freq_range = self.get_frequency_range_for_bar(i)

                if hasattr(self, "voice_info") and self.voice_info["has_voice"]:
                    pitch = self.voice_info.get("pitch", 0)
                    if pitch > 0 and freq_range[0] <= pitch <= freq_range[1]:
                        boost = 1.4
                        color = tuple(min(255, int(c * boost)) for c in color)

                if (
                    hasattr(self, "harmonic_info")
                    and self.harmonic_info.get("dominant_fundamental", 0) > 0
                ):
                    fund_freq = self.harmonic_info["dominant_fundamental"]
                    for n in range(1, 8):
                        harmonic_freq = fund_freq * n
                        if abs(freq_range[0] - harmonic_freq) < 20:
                            boost = 1.3
                            color = tuple(min(255, int(c * boost)) for c in color)

                if hasattr(self, "drum_info"):
                    kick_strength = self.drum_info["kick"].get("display_strength", 0)
                    snare_strength = self.drum_info["snare"].get("display_strength", 0)

                    if freq_range[0] <= 120 and kick_strength > 0.1:
                        boost = 1.0 + kick_strength * 1.2
                        color = tuple(min(255, int(c * boost)) for c in color)
                    elif 150 <= freq_range[0] <= 800 and snare_strength > 0.1:
                        boost = 1.0 + snare_strength * 0.8
                        color = tuple(min(255, int(c * boost)) for c in color)

                pygame.draw.rect(self.screen, color, (x, center_y - height, width, height))

                lower_color = tuple(int(c * 0.75) for c in color)
                pygame.draw.rect(self.screen, lower_color, (x, center_y, width, height))
        """


        spectrum_right = self.width - meter_width - vu_width - 10
        spectrum_bottom = vis_start_y + vis_height
        scale_y = spectrum_bottom + 5
        
        freq_scale_params = {
            'spectrum_left': spectrum_left,
            'spectrum_right': spectrum_right,
            'spectrum_bottom': spectrum_bottom,
            'scale_y': scale_y
        }
        self.display.draw_frequency_scale(freq_scale_params, self.freq_to_x_position, self.ui_scale)
        
            'x': (self.left_border_width - 25) // 2,  # Center in left border
            'y': self.height // 2 - 100,
            'width': 25,
            'height': 200
        }
        sub_bass_energy = self.calculate_sub_bass_energy()
        self.display.draw_sub_bass_indicator(sub_bass_energy, self.sub_bass_warning_active, sub_bass_position)
            'x': self.width - meter_width - vu_width - 150 - 20,
            'y': 80,
            'width': 150,
            'height': 60
        }
        self.display.draw_adaptive_allocation_indicator(
            self.adaptive_allocation_enabled,
            self.current_content_type,
            self.current_allocation,
            adaptive_position
        )

        if self.show_room_analysis:
            self.detect_and_display_room_modes()
            
        if self.show_pitch_detection:
            meter_width = self.meter_panel_width if self.show_professional_meters else 0
            vu_width = self.vu_meter_width if self.show_vu_meters else 0
            overlay_width = 320  # Increased by 40px for better layout
            overlay_height = 360  # Increased by 60px for fixed graph position
            overlay_x = self.width - meter_width - vu_width - overlay_width - 10
            
            
            self.pitch_detection_panel.draw(self.screen, overlay_x, overlay_y, overlay_width, overlay_height, self.ui_scale)
            
        if self.show_chromagram:
            self.chromagram_panel.draw(
                self.screen, x=int(380 * self.ui_scale), y=int(440 * self.ui_scale),
                width=int(300 * self.ui_scale), height=int(260 * self.ui_scale),
                ui_scale=self.ui_scale
            )
            
        if self.show_genre_classification:
            vu_width = self.vu_meter_width if self.show_vu_meters else 0
            overlay_x = int((self.width - meter_width - vu_width - 290 - 10) * self.ui_scale)
            
            base_y = int((280 + 10 + 200 + 10) * self.ui_scale)  # Header + gap + harmonic + gap
            if self.show_pitch_detection:
                base_y += int((320 + 10) * self.ui_scale)  # Pitch detection height + gap
            if self.show_chromagram:
                base_y += int((260 + 10) * self.ui_scale)  # Chromagram height + gap
            
            self.genre_classification_panel.draw(
                self.screen, x=overlay_x, y=base_y,
                width=int(290 * self.ui_scale), height=int(240 * self.ui_scale),
                ui_scale=self.ui_scale
            )

            bass_energy = self.calculate_band_energy(20, 250)
            mid_energy = self.calculate_band_energy(250, 2000)
            high_energy = self.calculate_band_energy(2000, 20000)
            
            tilt = self.calculate_spectral_tilt()
            tilt_description = "Bright" if tilt > 0 else "Dark" if tilt < -3 else "Balanced"
            
            crest_factor = self.calculate_crest_factor()
            dynamic_range = max(self.bar_heights) - np.mean(self.bar_heights[self.bar_heights > 0.1])
            
            room_modes = []
            for freq, _, duration, _ in self.room_mode_candidates[:3]:
                room_modes.append({'freq': freq, 'duration': duration})
            
            tech_data = {
                'bass_energy': bass_energy,
                'mid_energy': mid_energy,
                'high_energy': high_energy,
                'spectral_tilt': tilt,
                'tilt_description': tilt_description,
                'crest_factor': crest_factor,
                'dynamic_range': dynamic_range,
                'room_modes': room_modes
            }
            
            overlay_width = 320
            overlay_height = 350
            overlay_x = self.width - meter_width - vu_width - overlay_width - 10
            if self.show_pitch_detection:
                overlay_y = 280 + 10 + 200 + 10 + 320 + 10
            else:
                overlay_y = 280 + 10 + 200 + 10
                
            tech_position = {
                'x': overlay_x,
                'y': overlay_y,
                'width': overlay_width,
                'height': overlay_height
            }
            
            self.display.draw_technical_overlay(tech_data, tech_position, True)
        
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
            
            if voice_band_count > 0 and total_energy > 0:
                speech_ratio = speech_energy / total_energy
                pitch = self.voice_info.get("pitch", 0)
                pitch_confidence = 1.0 if pitch > 80 else 0.0
                confidence = (speech_ratio * 0.6 + pitch_confidence * 0.4)
                if self.voice_info.get("has_voice", False) and confidence < 0.3:
                    confidence = 0.3
            else:
                confidence = 0.0
            
            self.voice_info["confidence"] = confidence
            
            if self.voice_info.get("has_voice", False) and self.voice_info.get("pitch", 0) > 0:
                pitch = self.voice_info["pitch"]
                
                low_energy = 0.0
                high_energy = 0.0
                for i in range(min(self.bars, len(self.bar_heights))):
                    freq_start, freq_end = self.get_frequency_range_for_bar(i)
                    center_freq = (freq_start + freq_end) / 2
                    
                    if 80 <= center_freq <= 250:
                        low_energy += self.bar_heights[i]
                    elif 1000 <= center_freq <= 4000:
                        high_energy += self.bar_heights[i]
                
                spectral_tilt = low_energy / (high_energy + 0.001)
                
                if pitch < 160:
                    gender = "Male"
                    tilt_factor = min(1.0, spectral_tilt / 2.0)
                    gender_confidence = min(1.0, (160 - pitch) / 75 * 0.7 + tilt_factor * 0.3)
                elif pitch > 190:
                    gender = "Female"
                    tilt_factor = min(1.0, (2.0 - spectral_tilt) / 2.0)
                    gender_confidence = min(1.0, (pitch - 190) / 65 * 0.7 + tilt_factor * 0.3)
                else:
                    if spectral_tilt > 1.5:
                        gender = "Likely Male"
                        gender_confidence = 0.5 + (spectral_tilt - 1.5) * 0.3
                    else:
                        gender = "Likely Female"
                        gender_confidence = 0.5 + (1.5 - spectral_tilt) * 0.3
                
                self.voice_info["gender"] = gender
                self.voice_info["gender_confidence"] = gender_confidence
                self.voice_info["frequency_characteristics"] = {
                    "brightness": spectral_tilt,
                    "warmth": low_energy / (total_energy + 0.001)
                }
            
            overlay_width = 320
            overlay_height = 220
            overlay_x = self.width - meter_width - vu_width - overlay_width - 10
            
            if self.show_pitch_detection:
                overlay_y = 280 + 10 + 200 + 10 + 320 + 10 + 350 + 10
            else:
                overlay_y = 280 + 10 + 200 + 10 + 350 + 10
                
            voice_position = {
                'x': overlay_x,
                'y': overlay_y,
                'width': overlay_width,
                'height': overlay_height
            }
            
            self.display.draw_voice_info(self.voice_info, voice_position)
            
        if self.show_formants:
            self.draw_formant_overlays(vis_start_x, vis_start_y, vis_width, vis_height, center_y)

            self.display.draw_ab_comparison(self.reference_spectrum, self.bar_heights, vis_params)


        if self.show_room_analysis and hasattr(self, "room_modes") and len(self.room_modes) > 0:
            warning_y = 110
            for mode in self.room_modes[:2]:  # Show top 2 room modes
                severity_color = (255, int(200 * (1 - mode["severity"])), 100)
                mode_text = f"Room Mode: {mode['frequency']:.0f}Hz (Q={mode['q_factor']:.1f})"
                mode_surf = self.font_tiny.render(mode_text, True, severity_color)
                self.screen.blit(mode_surf, (10, warning_y))
                warning_y += 15
                
            help_sections = [
                ("Display Controls", [
                    ("ESC", "Exit analyzer"),
                    ("S", "Save screenshot"),
                    ("M", "Toggle professional meters"),
                    ("H", "Toggle harmonic analysis"),
                    ("R", "Toggle room analysis"),
                    ("Z", "Toggle bass zoom window"),
                    ("J", "Toggle VU meters"),
                    ("V", "Toggle voice info"),
                    ("F", "Toggle formants display"),
                    ("A", "Toggle advanced info"),
                    ("G", "Toggle frequency grid"),
                    ("B", "Toggle band separators"),
                    ("T", "Toggle technical overlay"),
                ]),
                ("Audio Processing", [
                    ("Q", "Toggle frequency compensation"),
                    ("W", "Toggle psychoacoustic weighting"),
                    ("E", "Toggle normalization"),
                    ("U", "Toggle smoothing"),
                    ("Y", "Toggle dynamic quality mode"),
                    ("I", "Toggle midrange boost"),
                    ("O", "Increase midrange boost"),
                    ("L", "Decrease midrange boost"),
                ]),
                ("OMEGA Features", [
                    ("P", "Toggle pitch detection (OMEGA)"),
                    ("K", "Toggle chromagram & key detection (OMEGA-1)"),
                    ("N", "Toggle genre classification (OMEGA-2)"),
                ]),
                ("Input Controls", [
                    ("+ / =", "Increase input gain"),
                    ("-", "Decrease input gain"),
                    ("0", "Reset gain to +12dB"),
                    ("C", "Store reference / Toggle A/B comparison"),
                    ("X", "Clear reference spectrum"),
                    ("D", "Print debug info"),
                    (";", "Toggle adaptive allocation"),
                ]),
                ("Window Presets", [
                    ("1", "1400x900 (Compact)"),
                    ("2", "1800x1000 (Standard)"),
                    ("3", "2200x1200 (Wide)"),
                    ("4", "2600x1400 (Ultra-Wide)"),
                    ("5", "3000x1600 (Cinema)"),
                    ("6", "3400x1800 (Studio)"),
                    ("7", "1920x1080 (Full HD)"),
                    ("8", "2560x1440 (2K)"),
                    ("9", "3840x2160 (4K)"),
                    ("SHIFT+0", "Toggle fullscreen"),
                ]),
            ]
            self.display.draw_help_menu(help_sections, True)

    def resize_window(self, new_width: int, new_height: int):
        """Resize window with professional aspect ratios and update fonts"""
        self.width = new_width
        self.height = new_height
        self.screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
        
        self.display.update_dimensions(new_width, new_height)

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
                            self.show_pitch_detection = not self.show_pitch_detection
                            print(f"🎵 Pitch Detection (OMEGA): {'ON' if self.show_pitch_detection else 'OFF'}")
                        elif event.key == pygame.K_k:
                            self.show_chromagram = not self.show_chromagram
                            print(f"🎹 Chromagram & Key Detection (OMEGA-1): {'ON' if self.show_chromagram else 'OFF'}")
                        elif event.key == pygame.K_n:
                            self.show_genre_classification = not self.show_genre_classification
                            print(f"🎸 Genre Classification (OMEGA-2): {'ON' if self.show_genre_classification else 'OFF'}")
                        elif event.key == pygame.K_y:
                            self.dynamic_quality_enabled = not self.dynamic_quality_enabled
                            print(f"🚀 Dynamic Quality: {'ON' if self.dynamic_quality_enabled else 'OFF'}")
                            if not self.dynamic_quality_enabled and self.quality_mode == "performance":
                                self.quality_mode = "quality"
                                self._switch_to_quality_mode()
                        elif event.key == pygame.K_c:
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
                            self.reference_spectrum = None
                            self.reference_stored = False
                            self.comparison_mode = False
                            print("🗑️ Reference spectrum cleared")
                        elif event.key == pygame.K_SLASH or event.key == pygame.K_QUESTION:
                            self.show_help_controls()
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
                            pygame.display.toggle_fullscreen()
                        elif event.key == pygame.K_d:
                            self.print_debug_output()
                            print("🐛 Debug output printed to terminal")
                        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                            self.input_gain *= 1.5
                            self.input_gain = min(self.input_gain, 16.0)  # Max +24dB
                            gain_db = 20 * np.log10(self.input_gain)
                            print(f"🔊 Input gain: +{gain_db:.1f}dB")
                        elif event.key == pygame.K_MINUS:
                            self.input_gain /= 1.5
                            self.input_gain = max(self.input_gain, 0.25)  # Min -12dB
                            gain_db = 20 * np.log10(self.input_gain)
                            print(f"🔊 Input gain: {gain_db:+.1f}dB")
                        elif event.key == pygame.K_0:
                            self.input_gain = 4.0
                            print(f"🔊 Input gain reset to +12.0dB")
                        elif event.key == pygame.K_SEMICOLON:
                            self.adaptive_allocation_enabled = not self.adaptive_allocation_enabled
                            if not self.adaptive_allocation_enabled:
                                self.current_allocation = 0.75
                                self.current_content_type = 'instrumental'
                            print(f"🎛️ Adaptive allocation: {'ON' if self.adaptive_allocation_enabled else 'OFF'}")
                        elif event.key == pygame.K_i:
                            self.midrange_boost_enabled = not self.midrange_boost_enabled
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost (1k-6k): {'ON' if self.midrange_boost_enabled else 'OFF'}")
                        elif event.key == pygame.K_o:
                            self.midrange_boost_factor = min(self.midrange_boost_factor + 0.5, 10.0)
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost factor: {self.midrange_boost_factor:.1f}x")
                        elif event.key == pygame.K_l:
                            self.midrange_boost_factor = max(self.midrange_boost_factor - 0.5, 1.0)
                            self.freq_compensation_gains = self._precalculate_freq_compensation()
                            print(f"🎵 Midrange boost factor: {self.midrange_boost_factor:.1f}x")
                        elif event.key == pygame.K_QUESTION or event.key == pygame.K_SLASH:
                            self.show_help = not self.show_help

                self.process_frame()
                self.draw_frame()

                pygame.display.flip()
                self.clock.tick(60)

                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps_counter.append(frame_count)
                    frame_count = 0
                    fps_timer = time.time()

        finally:
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
