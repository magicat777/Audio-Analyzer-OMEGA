"""
Harmonic Analysis Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract harmonic analysis as self-contained module
Enhanced with THD calculation, formant analysis, and expanded instrument profiles
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from scipy import signal as scipy_signal
from ..analyzers.harmonic_metrics import HarmonicMetrics


class HarmonicAnalyzer:
    """Advanced harmonic analysis and instrument identification"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2

        # Extended instrument harmonic profiles
        self.instrument_profiles = {
            # Drums
            "kick_drum": {
                "fundamental_range": (40, 120),
                "harmonic_pattern": [1.0, 0.3, 0.1, 0.05],
                "characteristics": "low_fundamental_strong",
                "inharmonicity": 0.02
            },
            "snare_drum": {
                "fundamental_range": (150, 400),
                "harmonic_pattern": [0.8, 1.0, 0.6, 0.4, 0.2],
                "characteristics": "broadband_with_peaks",
                "inharmonicity": 0.15
            },
            "hi_hat": {
                "fundamental_range": (3000, 8000),
                "harmonic_pattern": [1.0, 0.8, 0.6, 0.4],
                "characteristics": "metallic_noise",
                "inharmonicity": 0.25
            },
            "tom_tom": {
                "fundamental_range": (80, 300),
                "harmonic_pattern": [1.0, 0.5, 0.3, 0.1],
                "characteristics": "pitched_drum",
                "inharmonicity": 0.05
            },
            
            # String instruments
            "bass_guitar": {
                "fundamental_range": (41, 200),
                "harmonic_pattern": [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
                "characteristics": "strong_harmonics",
                "inharmonicity": 0.001
            },
            "electric_guitar": {
                "fundamental_range": (82, 880),
                "harmonic_pattern": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1],
                "characteristics": "rich_harmonics",
                "inharmonicity": 0.001
            },
            "acoustic_guitar": {
                "fundamental_range": (82, 880),
                "harmonic_pattern": [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1],
                "characteristics": "warm_harmonics",
                "inharmonicity": 0.002
            },
            "violin": {
                "fundamental_range": (196, 3520),
                "harmonic_pattern": [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.2],
                "characteristics": "bright_harmonics",
                "inharmonicity": 0.0005
            },
            "cello": {
                "fundamental_range": (65, 520),
                "harmonic_pattern": [1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
                "characteristics": "rich_low_harmonics",
                "inharmonicity": 0.0008
            },
            
            # Keyboard instruments
            "piano": {
                "fundamental_range": (27, 4186),
                "harmonic_pattern": [1.0, 0.5, 0.3, 0.2, 0.1, 0.05],
                "characteristics": "clean_harmonics",
                "inharmonicity": 0.003
            },
            "electric_piano": {
                "fundamental_range": (27, 4186),
                "harmonic_pattern": [1.0, 0.6, 0.3, 0.15, 0.08],
                "characteristics": "bell_like",
                "inharmonicity": 0.01
            },
            "organ": {
                "fundamental_range": (32, 8000),
                "harmonic_pattern": [1.0, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
                "characteristics": "additive_synthesis",
                "inharmonicity": 0.0
            },
            "harpsichord": {
                "fundamental_range": (27, 4186),
                "harmonic_pattern": [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1],
                "characteristics": "plucked_strings",
                "inharmonicity": 0.002
            },
            
            # Wind instruments
            "flute": {
                "fundamental_range": (262, 2093),
                "harmonic_pattern": [1.0, 0.2, 0.1, 0.05],
                "characteristics": "pure_tone",
                "inharmonicity": 0.0
            },
            "clarinet": {
                "fundamental_range": (147, 1568),
                "harmonic_pattern": [1.0, 0.1, 0.7, 0.05, 0.4],
                "characteristics": "odd_harmonics",
                "inharmonicity": 0.0
            },
            "saxophone": {
                "fundamental_range": (58, 880),
                "harmonic_pattern": [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
                "characteristics": "conical_bore",
                "inharmonicity": 0.001
            },
            "trumpet": {
                "fundamental_range": (165, 988),
                "harmonic_pattern": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
                "characteristics": "bright_brass",
                "inharmonicity": 0.0
            },
            "trombone": {
                "fundamental_range": (73, 587),
                "harmonic_pattern": [1.0, 0.85, 0.7, 0.6, 0.5, 0.4],
                "characteristics": "warm_brass",
                "inharmonicity": 0.0
            },
            
            # Voices
            "voice_female": {
                "fundamental_range": (150, 500),
                "harmonic_pattern": [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
                "characteristics": "formant_enhanced",
                "inharmonicity": 0.0
            },
            "voice_male": {
                "fundamental_range": (85, 300),
                "harmonic_pattern": [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
                "characteristics": "formant_enhanced",
                "inharmonicity": 0.0
            },
            "voice_child": {
                "fundamental_range": (200, 600),
                "harmonic_pattern": [1.0, 0.5, 0.3, 0.2, 0.1],
                "characteristics": "formant_enhanced",
                "inharmonicity": 0.0
            },
            
            # Synthesizers
            "synth_saw": {
                "fundamental_range": (20, 20000),
                "harmonic_pattern": [1.0, 0.5, 0.33, 0.25, 0.2, 0.17, 0.14],
                "characteristics": "1/n_harmonics",
                "inharmonicity": 0.0
            },
            "synth_square": {
                "fundamental_range": (20, 20000),
                "harmonic_pattern": [1.0, 0.0, 0.33, 0.0, 0.2, 0.0, 0.14],
                "characteristics": "odd_harmonics_only",
                "inharmonicity": 0.0
            },
            "synth_triangle": {
                "fundamental_range": (20, 20000),
                "harmonic_pattern": [1.0, 0.0, 0.11, 0.0, 0.04, 0.0, 0.02],
                "characteristics": "odd_harmonics_squared",
                "inharmonicity": 0.0
            }
        }

        # Harmonic detection history
        self.harmonic_history = deque(maxlen=30)
        self.instrument_confidence = {}
        
        # Harmonic metrics calculator
        self.metrics = HarmonicMetrics(sample_rate)
        
        # History for spectral metrics
        self.spectral_centroid_history = deque(maxlen=60)
        self.thd_history = deque(maxlen=60)
        self.previous_fft = None

    def detect_harmonic_series(self, fft_data: np.ndarray, freqs: np.ndarray, 
                              audio_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        
        # Calculate additional metrics if we have a dominant fundamental
        thd = 0.0
        thd_n = 0.0
        spectral_centroid = 0.0
        spectral_spread = 0.0
        spectral_flux = 0.0
        spectral_rolloff = 0.0
        inharmonicity = 0.0
        formants = []
        hnr = 0.0
        
        if harmonics_found:
            dominant_fund = harmonics_found[0]["fundamental"]
            
            # THD and THD+N
            thd = self.metrics.calculate_thd(fft_data, freqs, dominant_fund)
            thd_n = self.metrics.calculate_thd_plus_noise(fft_data, freqs, dominant_fund)
            
            # Spectral metrics
            spectral_centroid = self.metrics.calculate_spectral_centroid(fft_data, freqs)
            spectral_spread = self.metrics.calculate_spectral_spread(fft_data, freqs, spectral_centroid)
            spectral_rolloff = self.metrics.calculate_spectral_rolloff(fft_data, freqs)
            
            # Spectral flux (if we have previous FFT)
            if self.previous_fft is not None:
                spectral_flux = self.metrics.calculate_spectral_flux(fft_data, self.previous_fft)
            
            # Inharmonicity
            inharmonicity = self.metrics.calculate_inharmonicity(fft_data, freqs, dominant_fund)
            
            # Harmonic to noise ratio
            hnr = self.metrics.calculate_harmonic_to_noise_ratio(fft_data, freqs, dominant_fund)
            
            # Formant detection (if audio data provided)
            if audio_data is not None and len(audio_data) > 14:
                formants = self.metrics.detect_formants_lpc(audio_data)
        
        # Update histories
        self.spectral_centroid_history.append(spectral_centroid)
        self.thd_history.append(thd)
        self.previous_fft = fft_data.copy()

        return {
            "harmonic_series": harmonics_found,
            "instrument_matches": instrument_matches,
            "dominant_fundamental": harmonics_found[0]["fundamental"] if harmonics_found else 0,
            "thd": thd,
            "thd_plus_noise": thd_n,
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
            "spectral_flux": spectral_flux,
            "spectral_rolloff": spectral_rolloff,
            "inharmonicity": inharmonicity,
            "formants": formants,
            "hnr_db": hnr
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
                    strength_diff = abs(expected_strength - actual_strength)

                    # Score based on similarity (closer = higher score)
                    similarity = 1.0 - min(strength_diff, 1.0)
                    pattern_score += similarity
                    pattern_matches += 1
        
        # Bonus for matching inharmonicity characteristics
        if "inharmonicity" in profile and len(harmonics) > 3:
            measured_inharmonicity = self._calculate_simple_inharmonicity(harmonics)
            expected_inharmonicity = profile["inharmonicity"]
            
            if abs(measured_inharmonicity - expected_inharmonicity) < 0.05:
                score *= 1.2  # 20% bonus for matching inharmonicity

            if pattern_matches > 0:
                score = pattern_score / pattern_matches
            else:
                score = 0.0

        return score
    
    def _calculate_simple_inharmonicity(self, harmonics: List[Dict]) -> float:
        """Simple inharmonicity calculation from harmonic data"""
        if len(harmonics) < 2:
            return 0.0
        
        deviations = []
        fundamental = harmonics[0]["actual_freq"] if harmonics else 0
        
        for h in harmonics[1:]:
            expected = fundamental * h["number"]
            actual = h["actual_freq"]
            if expected > 0:
                deviation = abs(actual - expected) / expected
                deviations.append(deviation)
        
        return np.mean(deviations) if deviations else 0.0


class HarmonicAnalysisPanel:
    """Harmonic analysis panel for visual display of harmonic content"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.analyzer = HarmonicAnalyzer(sample_rate)
        
        # Display state
        self.harmonic_info = {
            "dominant_fundamental": 0,
            "harmonic_series": [],
            "instrument_matches": [],
            "thd": 0.0,
            "thd_plus_noise": 0.0,
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_flux": 0.0,
            "spectral_rolloff": 0.0,
            "inharmonicity": 0.0,
            "formants": [],
            "hnr_db": 0.0
        }
        
        # Store audio data for formant analysis
        self.last_audio_data = None
        
        # Fonts will be set by main app
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_large = fonts.get('large')
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        self.font_tiny = fonts.get('tiny')
        
    def update(self, fft_data: np.ndarray, freqs: np.ndarray, audio_data: Optional[np.ndarray] = None):
        """Update harmonic analysis with new FFT data"""
        if fft_data is not None and len(fft_data) > 0:
            self.last_audio_data = audio_data
            self.harmonic_info = self.analyzer.detect_harmonic_series(fft_data, freqs, audio_data)
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw enhanced harmonic analysis panel"""
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Draw background
        draw_panel_background(screen, x, y, width, height)
        
        # Draw centered header
        if self.font_medium:
            current_y = draw_panel_header(screen, "Harmonic Analysis", self.font_medium,
                                        x, y, width)
        else:
            current_y = y + 35
        
        current_y += int(5 * ui_scale)  # Small gap after header
        
        # Divide panel into sections
        left_width = int(width * 0.6)
        right_width = width - left_width

        # Left side - Fundamentals and instruments
        left_y = current_y
        
        # Dominant fundamental (always show)
        if self.font_large:
            fund_freq = self.harmonic_info["dominant_fundamental"]
            if fund_freq > 0:
                fund_text = f"{fund_freq:.1f} Hz"
                text_color = (255, 220, 120)
            else:
                fund_text = "-- Hz"
                text_color = (150, 150, 150)
            fund_surf = self.font_large.render(fund_text, True, text_color)
            screen.blit(fund_surf, (x + int(10 * ui_scale), left_y))
            left_y += int(35 * ui_scale)
        
        if self.font_small:
            # THD and THD+N (always show with fixed positions)
            thd = self.harmonic_info.get('thd', 0)
            thd_n = self.harmonic_info.get('thd_plus_noise', 0)
            
            if thd > 0 or thd_n > 0:
                thd_color = (
                    (100, 255, 100) if thd < 1 else
                    (255, 255, 100) if thd < 5 else
                    (255, 100, 100)
                )
                thd_text = f"THD: {thd:.2f}%  THD+N: {thd_n:.2f}%"
            else:
                thd_color = (150, 150, 150)
                thd_text = "THD: --%  THD+N: --%"
            
            thd_surf = self.font_small.render(thd_text, True, thd_color)
            screen.blit(thd_surf, (x + int(10 * ui_scale), left_y))
            left_y += int(20 * ui_scale)
            
            # Inharmonicity (always show)
            inharm = self.harmonic_info.get('inharmonicity', 0)
            if inharm > 0:
                inharm_text = f"Inharmonicity: {inharm:.3f}"
                inharm_color = (
                    (100, 255, 100) if inharm < 0.01 else
                    (255, 255, 100) if inharm < 0.05 else
                    (255, 100, 100)
                )
            else:
                inharm_text = "Inharmonicity: --"
                inharm_color = (150, 150, 150)
                
            inharm_surf = self.font_small.render(inharm_text, True, inharm_color)
            screen.blit(inharm_surf, (x + int(10 * ui_scale), left_y))
            left_y += int(25 * ui_scale)

        # Instrument matches (always show title and fixed bar positions)
        if self.font_small:
            # Always show title
            inst_title = self.font_small.render("Detected Instruments:", True, (180, 200, 220))
            screen.blit(inst_title, (x + int(10 * ui_scale), left_y))
            left_y += int(20 * ui_scale)
            
            instruments = self.harmonic_info.get("instrument_matches", [])
            
            # Always show 5 instrument bar slots
            for i in range(5):
                bar_height = 15
                bar_max_width = int(left_width * 0.7)
                
                # Always draw the outline
                pygame.draw.rect(screen, (100, 100, 120), 
                               (x + int(10 * ui_scale), left_y, bar_max_width, bar_height), 1)
                
                if i < len(instruments):
                    # Draw filled bar for detected instruments
                    match = instruments[i]
                    instrument = match["instrument"].replace("_", " ").title()
                    confidence = match["confidence"]

                    # Color based on confidence
                    color_intensity = int(100 + confidence * 155)
                    color = (color_intensity, color_intensity, 255)

                    # Draw confidence bar
                    bar_width = int(bar_max_width * confidence)
                    pygame.draw.rect(screen, color, 
                                   (x + int(10 * ui_scale), left_y, bar_width, bar_height))
                    
                    # Instrument name (black text for better contrast on colored bars)
                    match_text = f"{instrument}: {confidence:.0%}"
                    match_surf = self.font_small.render(match_text, True, (0, 0, 0))
                    screen.blit(match_surf, (x + int(15 * ui_scale), left_y))
                else:
                    # Draw placeholder text for empty slots
                    placeholder_text = f"--"
                    placeholder_surf = self.font_small.render(placeholder_text, True, (100, 100, 120))
                    screen.blit(placeholder_surf, (x + int(15 * ui_scale), left_y))
                
                left_y += int(18 * ui_scale)
        
        # Right side - Spectral metrics and formants
        right_x = x + left_width
        right_y = current_y
        
        if self.font_small:
            # Spectral centroid (brightness) - always show
            centroid = self.harmonic_info.get('spectral_centroid', 0)
            
            # Always show brightness label
            if centroid > 0:
                brightness = min(centroid / 2000, 1.0)  # Normalize to 0-1
                brightness_text = f"Brightness: {brightness:.0%}"
                brightness_color = (
                    int(255 * brightness),
                    int(200 * (1 - brightness)),
                    100
                )
            else:
                brightness = 0
                brightness_text = "Brightness: --%"
                brightness_color = (150, 150, 150)
            
            bright_surf = self.font_small.render(brightness_text, True, brightness_color)
            screen.blit(bright_surf, (right_x + 10, right_y))
            right_y += 20
            
            # Always draw brightness meter outline
            meter_width = right_width - 20
            meter_height = 10
            pygame.draw.rect(screen, (50, 50, 60), 
                           (right_x + 10, right_y, meter_width, meter_height))
            
            # Fill meter if we have data
            if brightness > 0:
                pygame.draw.rect(screen, brightness_color,
                               (right_x + 10, right_y, int(meter_width * brightness), meter_height))
            
            right_y += 15
            
            # HNR (voice quality) - Fixed position
            hnr = self.harmonic_info.get('hnr_db', 0)
            if hnr > 0:
                hnr_text = f"HNR: {hnr:.1f} dB"
                hnr_quality = "Excellent" if hnr > 20 else "Good" if hnr > 15 else "Fair" if hnr > 10 else "Poor"
                hnr_surf = self.font_small.render(f"{hnr_text} ({hnr_quality})", True, (200, 220, 200))
            else:
                hnr_surf = self.font_small.render("HNR: --", True, (150, 150, 150))
            screen.blit(hnr_surf, (right_x + 10, right_y))
            right_y += 25
            
            # Formants (fixed position) - always show title
            form_title = self.font_small.render("Formants:", True, (180, 200, 220))
            screen.blit(form_title, (right_x + 10, right_y))
            right_y += 20
            
            # Show formant frequencies or placeholders
            formants = self.harmonic_info.get('formants', [])
            for i in range(4):  # Always show F1-F4
                if i < len(formants):
                    form_text = f"F{i+1}: {formants[i]:.0f} Hz"
                    text_color = (150, 200, 255)
                else:
                    form_text = f"F{i+1}: --"
                    text_color = (100, 100, 120)
                    
                form_surf = self.font_small.render(form_text, True, text_color)
                screen.blit(form_surf, (right_x + 10, right_y))
                right_y += 18
            
            # Vowel detection (fixed position)
            vowel_detected = False
            if len(formants) >= 2:
                vowel, vowel_conf = self.analyzer.metrics.detect_vowel_from_formants(formants)
                if vowel != 'unknown' and vowel_conf > 0.5:
                    vowel_text = f"Vowel: [{vowel}] {vowel_conf:.0%}"
                    vowel_color = (255, 200, 100)
                    vowel_detected = True
            
            if not vowel_detected:
                vowel_text = "Vowel: --"
                vowel_color = (150, 150, 150)
                
            vowel_surf = self.font_small.render(vowel_text, True, vowel_color)
            screen.blit(vowel_surf, (right_x + 10, right_y))
            right_y += 20
        
        # Bottom section - Harmonic series visualization
        self._draw_harmonic_series(screen, x + 10, y + height - 100, 
                                 width - 20, 90, ui_scale)
    
    def _draw_harmonic_series(self, screen: pygame.Surface, x: int, y: int, 
                             width: int, height: int, ui_scale: float):
        """Draw harmonic series visualization"""
        # Background
        pygame.draw.rect(screen, (15, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (50, 60, 80), (x, y, width, height), 1)
        
        series = self.harmonic_info.get('harmonic_series', [])
        if not series or not self.font_tiny:
            return
        
        # Get the strongest harmonic series
        main_series = series[0] if series else None
        if not main_series:
            return
        
        harmonics = main_series.get('harmonics', [])
        if not harmonics:
            return
        
        # Title
        title = self.font_tiny.render("Harmonic Series", True, (150, 170, 200))
        screen.blit(title, (x + 5, y + 2))
        
        # Draw harmonics
        bar_width = width // min(16, len(harmonics) + 1)
        max_height = height - 25
        
        for i, harmonic in enumerate(harmonics[:16]):
            if i * bar_width > width - bar_width:
                break
            
            # Bar height based on relative strength
            rel_strength = harmonic['relative_strength']
            bar_height = int(max_height * min(rel_strength, 1.0))
            
            bar_x = x + i * bar_width + 5
            bar_y = y + height - bar_height - 15
            
            # Color based on harmonic number
            color = (
                255 - i * 10,
                200 - i * 8,
                100 + i * 5
            )
            
            # Draw bar
            if bar_height > 0:
                pygame.draw.rect(screen, color, 
                               (bar_x, bar_y, bar_width - 2, bar_height))
            
            # Harmonic number
            num_surf = self.font_tiny.render(str(harmonic['number']), True, (180, 180, 200))
            num_rect = num_surf.get_rect(centerx=bar_x + bar_width//2, 
                                       bottom=y + height - 2)
            screen.blit(num_surf, num_rect)
    
    def get_results(self) -> Dict[str, Any]:
        """Get current harmonic analysis results"""
        return {
            'harmonic_info': self.harmonic_info.copy(),
            'dominant_fundamental': self.harmonic_info.get('dominant_fundamental', 0),
            'instrument_matches': self.harmonic_info.get('instrument_matches', []).copy(),
            'thd': self.harmonic_info.get('thd', 0),
            'spectral_centroid': self.harmonic_info.get('spectral_centroid', 0),
            'formants': self.harmonic_info.get('formants', [])
        }