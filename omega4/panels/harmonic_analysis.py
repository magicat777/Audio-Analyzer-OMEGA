"""
Harmonic Analysis Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract harmonic analysis as self-contained module
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
from scipy import signal as scipy_signal


class HarmonicAnalyzer:
    """Advanced harmonic analysis and instrument identification"""

    def __init__(self, sample_rate: int = 48000):
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
                    strength_diff = abs(expected_strength - actual_strength)

                    # Score based on similarity (closer = higher score)
                    similarity = 1.0 - min(strength_diff, 1.0)
                    pattern_score += similarity
                    pattern_matches += 1

            if pattern_matches > 0:
                score = pattern_score / pattern_matches
            else:
                score = 0.0

        return score


class HarmonicAnalysisPanel:
    """Harmonic analysis panel for visual display of harmonic content"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.analyzer = HarmonicAnalyzer(sample_rate)
        
        # Display state
        self.harmonic_info = {
            "dominant_fundamental": 0,
            "harmonic_series": [],
            "instrument_matches": []
        }
        
        # Fonts will be set by main app
        self.font_medium = None
        self.font_small = None
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        
    def update(self, fft_data: np.ndarray, freqs: np.ndarray):
        """Update harmonic analysis with new FFT data"""
        if fft_data is not None and len(fft_data) > 0:
            self.harmonic_info = self.analyzer.detect_harmonic_series(fft_data, freqs)
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw harmonic analysis panel"""
        # Background
        analysis_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (25, 20, 35), analysis_rect)
        pygame.draw.rect(screen, (70, 60, 90), analysis_rect, 2)

        current_y = y + int(10 * ui_scale)

        # Title
        if self.font_medium:
            title = self.font_medium.render("Harmonic Analysis", True, (255, 255, 255))
            screen.blit(title, (x + int(10 * ui_scale), current_y))
            current_y += int(55 * ui_scale)

        # Dominant fundamental
        if self.harmonic_info["dominant_fundamental"] > 0 and self.font_small:
            fund_text = f"Fund: {self.harmonic_info['dominant_fundamental']:.1f} Hz"
            fund_surf = self.font_small.render(fund_text, True, (255, 220, 120))
            screen.blit(fund_surf, (x + int(10 * ui_scale), current_y))
            current_y += int(25 * ui_scale)

        # Instrument matches
        if self.font_small:
            instruments = self.harmonic_info.get("instrument_matches", [])
            for i, match in enumerate(instruments[:3]):  # Top 3 matches
                instrument = match["instrument"].replace("_", " ").title()
                confidence = match["confidence"]

                color_intensity = int(100 + confidence * 155)
                color = (color_intensity, color_intensity, 255)

                match_text = f"{instrument}: {confidence:.1%}"
                match_surf = self.font_small.render(match_text, True, color)
                screen.blit(match_surf, (x + int(10 * ui_scale), current_y))
                current_y += int(20 * ui_scale)
    
    def get_results(self) -> Dict[str, Any]:
        """Get current harmonic analysis results"""
        return {
            'harmonic_info': self.harmonic_info.copy(),
            'dominant_fundamental': self.harmonic_info.get('dominant_fundamental', 0),
            'instrument_matches': self.harmonic_info.get('instrument_matches', []).copy()
        }