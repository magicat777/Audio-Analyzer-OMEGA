"""
Chromagram Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract chromagram and key detection as self-contained module
OMEGA-1 Feature: Musical key detection
Enhanced with circle of fifths, Roman numeral analysis, and mode detection
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from .music_theory import MusicTheory


class ChromagramAnalyzer:
    """Real-time chromagram and key detection for musical analysis
    
    Enhanced with genre-aware processing for better accuracy
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.chroma_bins = 12
        
        # Krumhansl-Kessler key profiles for major and minor keys
        self.major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        self.minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Genre-specific key profiles
        self.genre_profiles = {
            'jazz': {
                'major_weights': np.array([1.0, 0.8, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 0.8, 1.0, 0.8, 1.0]),
                'minor_weights': np.array([1.0, 0.8, 1.0, 1.0, 0.9, 1.0, 0.8, 1.0, 1.0, 0.9, 1.0, 0.9]),
                'extended_chords': True,
                'chromatic_tolerance': 0.8
            },
            'classical': {
                'major_weights': np.array([1.2, 0.7, 1.0, 0.7, 1.1, 1.0, 0.7, 1.1, 0.7, 1.0, 0.7, 0.9]),
                'minor_weights': np.array([1.2, 0.8, 1.0, 1.1, 0.8, 1.0, 0.8, 1.1, 1.0, 0.8, 1.0, 0.9]),
                'extended_chords': False,
                'chromatic_tolerance': 0.3
            },
            'pop': {
                'major_weights': np.array([1.5, 0.5, 1.0, 0.5, 1.2, 1.0, 0.5, 1.2, 0.5, 1.0, 0.5, 0.8]),
                'minor_weights': np.array([1.5, 0.6, 1.0, 1.2, 0.6, 1.0, 0.6, 1.2, 1.0, 0.6, 1.0, 0.8]),
                'extended_chords': False,
                'chromatic_tolerance': 0.2
            },
            'metal': {
                'major_weights': np.array([1.8, 0.4, 1.2, 0.4, 1.0, 1.2, 0.6, 1.4, 0.4, 1.0, 0.4, 0.6]),
                'minor_weights': np.array([1.8, 0.6, 1.2, 1.4, 0.5, 1.2, 0.5, 1.4, 1.2, 0.5, 1.2, 0.7]),
                'extended_chords': False,
                'chromatic_tolerance': 0.6  # Higher tolerance for distortion
            },
            'rock': {
                'major_weights': np.array([1.6, 0.5, 1.1, 0.5, 1.1, 1.1, 0.6, 1.3, 0.5, 1.0, 0.5, 0.7]),
                'minor_weights': np.array([1.6, 0.6, 1.1, 1.3, 0.6, 1.1, 0.6, 1.3, 1.1, 0.6, 1.1, 0.8]),
                'extended_chords': False,
                'chromatic_tolerance': 0.4
            }
        }
        
        # Note names for display
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Chord types for detection
        self.chord_types = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            'dom7': [0, 4, 7, 10],
            'maj9': [0, 4, 7, 11, 2],
            'min9': [0, 3, 7, 10, 2],
            '13': [0, 4, 7, 10, 2, 5, 9],  # Jazz extended chord
            'power': [0, 7],  # Power chord (root + fifth)
            'sus2': [0, 2, 7],  # Suspended 2nd
            'sus4': [0, 5, 7]   # Suspended 4th
        }
        
        # History for temporal smoothing
        self.chroma_history = deque(maxlen=15)  # 0.25 seconds at 60 FPS - more responsive
        self.key_history = deque(maxlen=60)  # 1 second for stability
        self.chord_history = deque(maxlen=50)  # Longer chord progression tracking
        
        # Current genre context
        self.current_genre = 'pop'  # Default to pop
        
        # Enhanced features
        self.mode_history = deque(maxlen=30)  # Mode detection history
        self.modulation_history = deque(maxlen=100)  # Track key changes
        self.roman_numeral_sequence = deque(maxlen=16)  # Roman numeral progression
        self.chord_transition_matrix = {}  # Chord transition probabilities
        
    def compute_chromagram(self, fft_data: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Extract 12-bin chromagram from FFT data"""
        chroma = np.zeros(12)
        A4 = 440.0
        C0 = A4 * np.power(2, -4.75)  # C0 = 16.35 Hz
        
        # Process lower frequencies for metal/blues detection (drop tunings)
        for i, freq in enumerate(freqs):
            if freq > 60 and freq < 8000:  # Include lower frequencies for drop tunings
                # Map frequency to pitch class with sub-semitone precision
                if freq > 0:
                    # Calculate pitch relative to C0
                    pitch = 12 * np.log2(freq / C0)
                    
                    # Get fractional chroma bin
                    chroma_bin_float = pitch % 12
                    
                    # Distribute energy to adjacent bins for sub-semitone accuracy
                    lower_bin = int(np.floor(chroma_bin_float)) % 12
                    upper_bin = int(np.ceil(chroma_bin_float)) % 12
                    fraction = chroma_bin_float - lower_bin
                    
                    # Weight by magnitude
                    magnitude = fft_data[i]
                    chroma[lower_bin] += magnitude * (1 - fraction)
                    if upper_bin != lower_bin:
                        chroma[upper_bin] += magnitude * fraction
        
        # Normalize to sum to 1
        chroma_sum = np.sum(chroma)
        if chroma_sum > 0:
            chroma = chroma / chroma_sum
            
        # Add to history for smoothing
        self.chroma_history.append(chroma)
        
        # Return smoothed chromagram (minimal smoothing for maximum responsiveness)
        if len(self.chroma_history) > 2:
            # Use weighted average - heavy weight on recent frames
            weights = np.exp(np.linspace(-2, 0, len(self.chroma_history)))
            weights = weights / np.sum(weights)
            smoothed = np.average(self.chroma_history, axis=0, weights=weights)
            return smoothed
        else:
            return chroma
    
    def detect_key(self, chroma: np.ndarray) -> Tuple[str, float]:
        """Detect musical key using correlation with Krumhansl-Kessler profiles
        
        Enhanced with genre-aware weighting for better accuracy
        """
        best_key = None
        best_correlation = -1
        best_is_major = True
        
        # Skip if chromagram is empty
        if np.sum(chroma) == 0:
            return "Unknown", 0.0
        
        # Get genre-specific weights if available
        genre_weights = self.genre_profiles.get(self.current_genre.lower(), self.genre_profiles['pop'])
        major_weights = genre_weights['major_weights']
        minor_weights = genre_weights['minor_weights']
        
        # Normalize chromagram first to reduce effect of dominant notes
        # Use square root to compress dynamic range and capture more notes
        chroma_sqrt = np.sqrt(chroma)
        chroma_norm = chroma_sqrt / (np.sum(chroma_sqrt) + 1e-10)
        
        # Apply genre-specific chromagram weighting
        weighted_chroma = chroma_norm * (major_weights + minor_weights) / 2
        
        # Test all 24 possible keys (12 major + 12 minor)
        correlations = []  # Store all correlations for debugging
        
        for i in range(12):
            # Rotate profiles to test each key
            major_profile_rotated = np.roll(self.major_profile, i)
            minor_profile_rotated = np.roll(self.minor_profile, i)
            
            # Apply genre-specific weights to profiles
            major_profile_weighted = major_profile_rotated * np.roll(major_weights, i)
            minor_profile_weighted = minor_profile_rotated * np.roll(minor_weights, i)
            
            # Normalize profiles to ensure fair comparison
            major_profile_weighted = major_profile_weighted / (np.sum(major_profile_weighted) + 1e-10)
            minor_profile_weighted = minor_profile_weighted / (np.sum(minor_profile_weighted) + 1e-10)
            
            # Calculate correlation coefficients
            if np.std(weighted_chroma) > 0 and np.std(major_profile_weighted) > 0:
                major_corr = np.corrcoef(weighted_chroma, major_profile_weighted)[0, 1]
            else:
                major_corr = 0
                
            if np.std(weighted_chroma) > 0 and np.std(minor_profile_weighted) > 0:
                minor_corr = np.corrcoef(weighted_chroma, minor_profile_weighted)[0, 1]
            else:
                minor_corr = 0
                
            correlations.append((i, 'major', major_corr))
            correlations.append((i, 'minor', minor_corr))
            
            # Genre-specific adjustments
            if self.current_genre.lower() == 'jazz':
                # Jazz prefers flat keys
                if self.note_names[i] in ['F', 'Bb', 'Eb', 'Ab']:
                    major_corr *= 1.1
                    minor_corr *= 1.1
            elif self.current_genre.lower() == 'classical':
                # Classical prefers natural keys
                if self.note_names[i] in ['C', 'G', 'D', 'A', 'F']:
                    major_corr *= 1.05
            elif self.current_genre.lower() == 'rock':
                # Rock prefers guitar-friendly keys
                if self.note_names[i] in ['E', 'A', 'D', 'G']:
                    major_corr *= 1.05
                    minor_corr *= 1.05
            
            # Apply genre adjustments to the correlations we just added
            if self.current_genre.lower() == 'jazz' and self.note_names[i] in ['F', 'Bb', 'Eb', 'Ab']:
                correlations[-2] = (correlations[-2][0], correlations[-2][1], correlations[-2][2] * 1.1)
                correlations[-1] = (correlations[-1][0], correlations[-1][1], correlations[-1][2] * 1.1)
            elif self.current_genre.lower() == 'classical' and self.note_names[i] in ['C', 'G', 'D', 'A', 'F']:
                correlations[-2] = (correlations[-2][0], correlations[-2][1], correlations[-2][2] * 1.05)
            elif self.current_genre.lower() == 'rock' and self.note_names[i] in ['E', 'A', 'D', 'G']:
                correlations[-2] = (correlations[-2][0], correlations[-2][1], correlations[-2][2] * 1.05)
                correlations[-1] = (correlations[-1][0], correlations[-1][1], correlations[-1][2] * 1.05)
        
        # Find best correlation from all candidates
        best_idx, best_mode, best_correlation = max(correlations, key=lambda x: x[2])
        best_key = f"{self.note_names[best_idx]} {best_mode.capitalize()}"
        best_is_major = (best_mode == 'major')
        
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
        total_weight = sum(weighted_keys.values())
        confidence = weighted_keys[best_key] / total_weight if total_weight > 0 else 0.0
        
        return best_key, confidence
    
    def get_alternative_keys(self, chroma: np.ndarray, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N alternative key possibilities"""
        key_scores = []
        
        # Test all 24 possible keys
        for i in range(12):
            # Rotate profiles to test each key
            major_profile_rotated = np.roll(self.major_profile, i)
            minor_profile_rotated = np.roll(self.minor_profile, i)
            
            # Calculate correlation coefficients
            if np.std(chroma) > 0:
                major_corr = np.corrcoef(chroma, major_profile_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma, minor_profile_rotated)[0, 1]
            else:
                major_corr = minor_corr = 0
            
            key_scores.append((f"{self.note_names[i]} Major", major_corr))
            key_scores.append((f"{self.note_names[i]} Minor", minor_corr))
        
        # Sort by correlation and return top N
        key_scores.sort(key=lambda x: x[1], reverse=True)
        return key_scores[:top_n]
    
    def detect_chord(self, chroma: np.ndarray, debug_enabled: bool = False) -> Tuple[str, float]:
        """Detect current chord from chromagram
        
        Enhanced with genre-aware chord type selection
        """
        best_chord = "N/A"
        best_score = 0.0
        
        # Get genre-specific settings
        genre_settings = self.genre_profiles.get(self.current_genre.lower(), self.genre_profiles['pop'])
        use_extended = genre_settings['extended_chords']
        
        # Select chord types based on genre
        if use_extended:
            # Jazz: test all chord types including extended
            chord_types_to_test = self.chord_types.items()
        elif self.current_genre.lower() in ['metal', 'rock']:
            # Metal/Rock: include power chords and suspended chords
            chord_types_to_test = [(k, v) for k, v in self.chord_types.items() 
                                  if k in ['major', 'minor', 'power', 'sus2', 'sus4', 'dom7']]
        else:
            # Pop/Classical: only test basic triads and 7ths
            chord_types_to_test = [(k, v) for k, v in self.chord_types.items() 
                                  if k in ['major', 'minor', 'dim', 'aug', 'maj7', 'min7']]
        
        # Find strongest pitch classes
        top_pitches = np.argsort(chroma)[-5:][::-1]  # Top 5 pitches
        
        # Test each root note (even more sensitive for metal/blues)
        for root_idx in range(12):
            if chroma[root_idx] < 0.03:  # Very sensitive for metal/blues detection
                continue
                
            # Test each chord type
            for chord_name, intervals in chord_types_to_test:
                score = 0.0
                chord_pitches = [(root_idx + interval) % 12 for interval in intervals]
                
                # Calculate how well the chromagram matches this chord
                for i, pitch in enumerate(chord_pitches):
                    if i == 0:  # Root note is most important
                        score += chroma[pitch] * 2.0
                    else:
                        score += chroma[pitch]
                
                # Penalize non-chord tones (genre-dependent)
                chromatic_penalty = genre_settings['chromatic_tolerance']
                for pitch in range(12):
                    if pitch not in chord_pitches:
                        score -= chroma[pitch] * chromatic_penalty
                
                # Normalize by chord size
                score /= len(chord_pitches)
                
                # Bonus for common chord types in genre
                if self.current_genre.lower() == 'jazz' and chord_name in ['maj7', 'min7', 'dom7', '13']:
                    score *= 1.2
                elif self.current_genre.lower() in ['metal', 'rock'] and chord_name in ['power', 'minor', 'sus2', 'sus4']:
                    score *= 1.3  # Strong bonus for metal/rock chord types
                elif self.current_genre.lower() == 'pop' and chord_name in ['major', 'minor']:
                    score *= 1.1
                elif self.current_genre.lower() == 'classical' and chord_name in ['major', 'minor', 'dim']:
                    score *= 1.1
                
                if score > best_score:
                    best_score = score
                    best_chord = f"{self.note_names[root_idx]}{chord_name}"
        
        # Add to history with different logic for better progression tracking
        should_add = False
        if not self.chord_history:
            should_add = True
        elif best_chord != self.chord_history[-1]:
            should_add = True  # Always add different chords
        elif best_score > 0.6:  # Only add same chord if very confident
            should_add = True
            
        if should_add:
            self.chord_history.append(best_chord)
            # Debug: Print chord changes (only when debug mode is enabled)
            if debug_enabled and self.chord_history and len(self.chord_history) > 1:
                print(f"Chord change: {self.chord_history[-2]} -> {best_chord} (confidence: {best_score:.2f})")
        
        return best_chord, best_score
    
    def get_chord_progression(self) -> List[str]:
        """Get recent chord progression"""
        return list(self.chord_history)[-8:]  # Last 8 chords
    
    def analyze_harmonic_rhythm(self) -> float:
        """Analyze how frequently chords change"""
        if len(self.chord_history) < 2:
            return 0.0
            
        changes = sum(1 for i in range(1, len(self.chord_history)) 
                     if self.chord_history[i] != self.chord_history[i-1])
        
        return changes / (len(self.chord_history) - 1)
    
    def get_enhanced_chord_progression(self) -> List[str]:
        """Get chord progression with enhanced musical logic"""
        if len(self.chord_history) < 8:
            # If we don't have enough history, pad with musically sensible defaults
            base_chords = list(self.chord_history)
            
            # Get the current key to generate sensible chord progressions
            if hasattr(self, 'current_key') and self.current_key and self.current_key != 'Unknown':
                key_root = self.current_key.split()[0]  # Get root note (e.g., 'C' from 'C Major')
                is_major = 'Major' in self.current_key
                
                # Generate common chord progressions
                if is_major:
                    # Common major progressions: I-V-vi-IV, I-IV-V-I, vi-IV-I-V
                    common_progressions = [
                        [key_root + 'major', key_root + 'major', 
                         self._transpose_chord(key_root, 5, 'major'), self._transpose_chord(key_root, 9, 'minor'),
                         self._transpose_chord(key_root, 5, 'major'), self._transpose_chord(key_root, 3, 'major'),
                         key_root + 'major', key_root + 'major']
                    ]
                else:
                    # Common minor progressions: i-VII-VI-VII, i-iv-V-i
                    common_progressions = [
                        [key_root + 'minor', key_root + 'minor',
                         self._transpose_chord(key_root, 10, 'major'), self._transpose_chord(key_root, 8, 'major'),
                         self._transpose_chord(key_root, 10, 'major'), self._transpose_chord(key_root, 5, 'minor'),
                         key_root + 'minor', key_root + 'minor']
                    ]
                
                # Fill in missing chords from common progression
                if common_progressions:
                    prog = common_progressions[0]
                    while len(base_chords) < 8:
                        base_chords.append(prog[len(base_chords) % len(prog)])
                        
            return base_chords[-8:]  # Return last 8
        
        return list(self.chord_history)[-8:]  # Last 8 chords
    
    def _transpose_chord(self, root: str, semitones: int, chord_type: str) -> str:
        """Helper to transpose a chord by semitones"""
        try:
            root_idx = self.note_names.index(root)
            new_root_idx = (root_idx + semitones) % 12
            return self.note_names[new_root_idx] + chord_type
        except (ValueError, IndexError):
            return root + chord_type
    
    def set_genre_context(self, genre: str):
        """Update genre context for better analysis"""
        self.current_genre = genre


class ChromagramPanel:
    """OMEGA-1 Chromagram and Key Detection Panel with visualization"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.analyzer = ChromagramAnalyzer(sample_rate)
        
        # Display state
        self.chromagram_info = {
            'chromagram': np.zeros(12),
            'key': 'Unknown',
            'is_major': True,
            'confidence': 0.0,
            'stability': 0.0,
            'alternative_keys': [],
            'current_chord': 'N/A',
            'chord_confidence': 0.0,
            'chord_progression': [],
            'harmonic_rhythm': 0.0,
            'mode': 'Ionian (Major)',
            'mode_confidence': 0.0,
            'roman_progression': [],
            'key_modulation': None,
            'chord_transitions': {}
        }
        
        # Expose data for integration
        self.chromagram = np.zeros(12)
        self.detected_key = 'Unknown'
        self.key_confidence = 0.0
        self.current_chord = 'N/A'
        self.chord_confidence = 0.0
        self.chord_sequence = []
        self.circle_position = 0  # Position on circle of fifths
        self.current_mode = 'Ionian (Major)'
        self.mode_confidence = 0.0
        self.key_modulation = None
        self.roman_progression = []
        self.chord_transition_matrix = {}
        
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
        
    def update(self, fft_data: np.ndarray, audio_data: np.ndarray, freqs: np.ndarray, 
               current_genre: str = None, debug_enabled: bool = False):
        """Update chromagram and key detection with new FFT data
        
        Enhanced to accept genre context and detect chords
        """
        if fft_data is not None and len(fft_data) > 0:
            # Check for silence
            if audio_data is not None:
                rms = np.sqrt(np.mean(audio_data ** 2))
                if rms < 0.001:  # Silence threshold
                    # Reset to default state when silent - update both internal state and display info
                    self.chromagram = np.zeros(12)
                    self.detected_key = 'Unknown'
                    self.key_confidence = 0.0
                    self.key_stability = 0.0
                    self.current_chord = 'N/A'
                    self.chord_confidence = 0.0
                    self.chord_sequence = []
                    self.alternative_keys = []
                    self.chord_progression = []
                    self.harmonic_rhythm = 0.0
                    self.circle_position = 0
                    self.current_mode = 'Ionian (Major)'
                    self.mode_confidence = 0.0
                    self.key_modulation = None
                    self.roman_progression = []
                    
                    # Clear analyzer histories
                    self.analyzer.chroma_history.clear()
                    self.analyzer.chord_history.clear()
                    self.analyzer.key_history.clear()
                    self.analyzer.mode_history.clear()
                    self.analyzer.modulation_history.clear()
                    self.analyzer.roman_numeral_sequence.clear()
                    
                    # Update display info structure for proper visual reset
                    self.chromagram_info = {
                        'chromagram': np.zeros(12),
                        'key': 'Unknown',
                        'is_major': True,
                        'confidence': 0.0,
                        'stability': 0.0,
                        'alternative_keys': [],
                        'current_chord': 'N/A',
                        'chord_confidence': 0.0,
                        'chord_progression': [],
                        'harmonic_rhythm': 0.0,
                        'mode': 'Ionian (Major)',
                        'mode_confidence': 0.0,
                        'roman_progression': [],
                        'key_modulation': None,
                        'chord_transitions': {}
                    }
                    return
            
            # Update genre context if provided
            if current_genre:
                self.analyzer.set_genre_context(current_genre)
                
            # Compute chromagram
            chromagram = self.analyzer.compute_chromagram(fft_data, freqs)
            
            # Detect key
            key, correlation = self.analyzer.detect_key(chromagram)
            
            # Detect current chord
            chord, chord_conf = self.analyzer.detect_chord(chromagram, debug_enabled=debug_enabled)
            
            # Get stability
            stability = self.analyzer.get_key_stability()
            
            # Get most likely key with confidence
            most_likely_key, confidence = self.analyzer.get_most_likely_key()
            
            # Get alternative keys
            alt_keys = self.analyzer.get_alternative_keys(chromagram, top_n=3)
            
            # Store current key for enhanced progression
            self.analyzer.current_key = most_likely_key
            
            # Get chord progression and harmonic rhythm (use simple version for now)
            chord_progression = self.analyzer.get_chord_progression()  # Revert to simple version
            harmonic_rhythm = self.analyzer.analyze_harmonic_rhythm()
            
            # Calculate circle of fifths position
            self.circle_position, _ = MusicTheory.get_circle_position(most_likely_key)
            
            # Detect mode
            root_idx = self._get_root_index(most_likely_key)
            mode, mode_conf = MusicTheory.analyze_mode(chromagram, root_idx)
            self.analyzer.mode_history.append((mode, mode_conf))
            
            # Average mode confidence over history
            if len(self.analyzer.mode_history) > 5:
                avg_mode_conf = np.mean([conf for _, conf in self.analyzer.mode_history])
                most_common_mode = max(set(m for m, _ in self.analyzer.mode_history), 
                                     key=lambda x: sum(1 for m, _ in self.analyzer.mode_history if m == x))
                mode = most_common_mode
                mode_conf = avg_mode_conf
            
            # Convert chord progression to Roman numerals
            roman_progression = []
            for chord_name in chord_progression[-8:]:
                roman = MusicTheory.chord_to_roman(chord_name, most_likely_key)
                roman_progression.append(roman)
                self.analyzer.roman_numeral_sequence.append(roman)
            
            # Detect key modulation
            if len(self.analyzer.modulation_history) > 0:
                prev_key = self.analyzer.modulation_history[-1]
                if prev_key != most_likely_key:
                    modulation_type = MusicTheory.detect_key_modulation(prev_key, most_likely_key)
                    self.key_modulation = modulation_type
                else:
                    self.key_modulation = None
            else:
                self.key_modulation = None
            self.analyzer.modulation_history.append(most_likely_key)
            
            # Update chord transition matrix
            if len(chord_progression) > 1:
                self.chord_transition_matrix = MusicTheory.create_chord_transition_matrix(
                    list(self.analyzer.roman_numeral_sequence)
                )
            
            # Update info
            self.chromagram_info = {
                'chromagram': chromagram,
                'key': most_likely_key,
                'is_major': 'Major' in most_likely_key,
                'confidence': confidence,
                'stability': stability,
                'alternative_keys': [(k, c) for k, c in alt_keys[1:] if c > 0.3],  # Skip first (it's the detected key)
                'current_chord': chord,
                'chord_confidence': chord_conf,
                'chord_progression': chord_progression,
                'harmonic_rhythm': harmonic_rhythm,
                'mode': mode,
                'mode_confidence': mode_conf,
                'roman_progression': roman_progression,
                'key_modulation': self.key_modulation,
                'chord_transitions': self.chord_transition_matrix
            }
            
            # Update exposed attributes for integration
            self.chromagram = chromagram.copy()
            self.detected_key = most_likely_key
            self.key_confidence = confidence
            self.current_chord = chord
            self.chord_confidence = chord_conf
            self.chord_sequence = chord_progression
            self.current_mode = mode
            self.mode_confidence = mode_conf
            self.roman_progression = roman_progression
    
    def _get_root_index(self, key: str) -> int:
        """Get chromatic index of root note"""
        key_root = key.split()[0]
        try:
            return MusicTheory.NOTE_NAMES.index(key_root)
        except ValueError:
            return 0
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """OMEGA-1: Draw enhanced chromagram with circle of fifths and analysis"""
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((35, 25, 45))  # Purple tint for musical theme
        screen.blit(overlay, (x, y))
        
        # Border
        pygame.draw.rect(
            screen, (140, 100, 180), (x, y, width, height), 2
        )
        
        # Divide panel into sections
        # Left: Chromagram and key info
        # Right: Circle of fifths
        # Bottom: Chord progression
        left_width = int(width * 0.55)  # Adjusted for wider panel
        right_width = width - left_width
        
        y_offset = y + int(10 * ui_scale)
        
        # Title with mode
        if self.font_medium:
            mode = self.chromagram_info.get('mode', 'Ionian')
            title_text = self.font_medium.render("Enhanced Chromagram", True, (255, 200, 255))
            screen.blit(title_text, (x + int(10 * ui_scale), y_offset))
            y_offset += int(30 * ui_scale)
        
        # Get chromagram data
        chromagram = self.chromagram_info.get('chromagram', np.zeros(12))
        key = self.chromagram_info.get('key', 'C')
        is_major = self.chromagram_info.get('is_major', True)
        confidence = self.chromagram_info.get('confidence', 0.0)
        stability = self.chromagram_info.get('stability', 0.0)
        alt_keys = self.chromagram_info.get('alternative_keys', [])
        
        # Display detected key and mode
        if self.font_large:
            key_color = (
                (255, 150, 150) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            key_text = f"{key}"
            key_surf = self.font_large.render(key_text, True, key_color)
            screen.blit(key_surf, (x + int(10 * ui_scale), y_offset))
            
            # Mode on same line
            if self.font_small:
                mode = self.chromagram_info.get('mode', 'Ionian')
                mode_text = f" ({mode.split()[0]})"
                mode_surf = self.font_small.render(mode_text, True, (200, 180, 220))
                screen.blit(mode_surf, (x + int(10 * ui_scale) + key_surf.get_width(), y_offset + 5))
            
            y_offset += int(35 * ui_scale)
        
        # Key modulation indicator
        modulation = self.chromagram_info.get('key_modulation')
        if modulation and self.font_small:
            mod_text = f"Modulation: {modulation}"
            mod_surf = self.font_small.render(mod_text, True, (255, 200, 100))
            screen.blit(mod_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(20 * ui_scale)
        
        # Confidence and stability on one line
        if self.font_small:
            conf_text = f"Confidence: {confidence:.0%}"
            stab_text = f"Stability: {stability:.0%}"
            
            conf_color = (
                (255, 150, 150) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            stab_color = (
                (255, 150, 150) if stability < 0.3 else
                (255, 255, 150) if stability < 0.7 else
                (150, 255, 150)
            )
            
            conf_surf = self.font_small.render(conf_text, True, conf_color)
            stab_surf = self.font_small.render(stab_text, True, stab_color)
            
            screen.blit(conf_surf, (x + int(10 * ui_scale), y_offset))
            screen.blit(stab_surf, (x + int(120 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        # Draw chromagram visualization (left side) - taller bars
        chroma_x = x + int(10 * ui_scale)
        chroma_y = y_offset
        chroma_width = left_width - int(20 * ui_scale)
        chroma_height = int(120 * ui_scale)  # Further increased height for 3x taller bars
        
        # Background for chromagram
        pygame.draw.rect(screen, (20, 20, 30), 
                        (chroma_x, chroma_y, chroma_width, chroma_height))
        
        # Draw each chroma bin
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        bar_width = chroma_width // 12
        
        # Extract root note from key name
        root_note = key.replace(' Major', '').replace(' Minor', '')
        
        for i, (note, value) in enumerate(zip(note_names, chromagram)):
            bar_x = chroma_x + i * bar_width
            bar_height = int(chroma_height * min(value * 3.0, 1.0))  # 3x amplified values, clamped to available height
            
            # Color based on note relationship to detected key
            if note == root_note:
                # Root note - bright color
                color = (255, 200, 100)
            elif i in self._get_scale_degrees(root_note, is_major):
                # Part of key - medium brightness
                color = (200, 150, 100)
            else:
                # Not in key - dimmer
                color = (100, 100, 150)
            
            # Draw bar
            if bar_height > 0:
                pygame.draw.rect(screen, color,
                               (bar_x + 2, chroma_y + chroma_height - bar_height, 
                                bar_width - 4, bar_height))
            
            # Draw note labels
            if self.font_tiny:
                label_color = (255, 255, 255) if note == root_note else (150, 150, 150)
                note_surf = self.font_tiny.render(note, True, label_color)
                note_rect = note_surf.get_rect(centerx=bar_x + bar_width // 2, 
                                              bottom=chroma_y + chroma_height + int(15 * ui_scale))
                screen.blit(note_surf, note_rect)
        
        # Border around chromagram
        pygame.draw.rect(screen, (100, 100, 120), 
                        (chroma_x, chroma_y, chroma_width, chroma_height), 1)
        
        y_offset = chroma_y + chroma_height + int(20 * ui_scale)
        
        # Draw circle of fifths (right side) - using more vertical space
        circle_height = int(height * 0.40)  # Slightly reduced to make room
        self._draw_circle_of_fifths(screen, x + left_width, y + int(30 * ui_scale), 
                                   right_width - int(10 * ui_scale), circle_height, ui_scale)
        
        # Add 30px space below circle of fifths
        circle_bottom = y + int(30 * ui_scale) + circle_height + int(30 * ui_scale)
        
        # Current chord with Roman numeral
        if self.font_medium:
            chord = self.chromagram_info.get('current_chord', 'N/A')
            roman_prog = self.chromagram_info.get('roman_progression', [])
            
            if chord != 'N/A' and roman_prog:
                current_roman = roman_prog[-1] if roman_prog else ''
                chord_function = MusicTheory.get_chord_function(current_roman)
                tsd_group = MusicTheory.get_tsd_group(current_roman)
                
                # TSD indicator color
                tsd_colors = {
                    'Tonic': (100, 200, 100),
                    'Subdominant': (200, 200, 100),
                    'Dominant': (200, 100, 100),
                    'Other': (150, 150, 150)
                }
                
                chord_text = f"{chord} ({current_roman})"
                chord_surf = self.font_medium.render(chord_text, True, tsd_colors.get(tsd_group, (200, 200, 200)))
                screen.blit(chord_surf, (x + int(10 * ui_scale), y_offset))
                
                if self.font_tiny:
                    func_text = f"{chord_function} - {tsd_group}"
                    func_surf = self.font_tiny.render(func_text, True, (150, 150, 170))
                    screen.blit(func_surf, (x + int(10 * ui_scale), y_offset + 20))
                
                y_offset += int(45 * ui_scale)
        
        # Ensure we have at least 30px below circle before chord progression
        if 'circle_bottom' in locals():
            y_offset = max(y_offset, circle_bottom)
        
        # Chord progression with Roman numerals
        chord_prog_height = int(70 * ui_scale)
        self._draw_chord_progression(screen, x + int(10 * ui_scale), y_offset, 
                                   width - int(20 * ui_scale), chord_prog_height, ui_scale)
        
        y_offset += chord_prog_height + int(10 * ui_scale)
        
        # Chord transition matrix below chord progression
        if self.chromagram_info.get('chord_transitions'):
            remaining_height = (y + height) - y_offset - int(10 * ui_scale)
            if remaining_height > 60:
                self._draw_transition_matrix(screen, x + int(10 * ui_scale), y_offset, 
                                           width - int(20 * ui_scale), min(remaining_height, 100), ui_scale)
    
    def _get_scale_degrees(self, root_note: str, is_major: bool) -> List[int]:
        """Get scale degrees for a given key"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        try:
            root_idx = note_names.index(root_note)
        except ValueError:
            return []
        
        # Major scale intervals: W W H W W W H
        # Minor scale intervals: W H W W H W W
        if is_major:
            intervals = [2, 2, 1, 2, 2, 2, 1]
        else:
            intervals = [2, 1, 2, 2, 1, 2, 2]
        
        scale_degrees = [root_idx]
        current = root_idx
        for interval in intervals[:-1]:  # We don't need the last interval back to root
            current = (current + interval) % 12
            scale_degrees.append(current)
        
        return scale_degrees
    
    def _draw_circle_of_fifths(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw interactive circle of fifths visualization"""
        # Background
        pygame.draw.rect(screen, (25, 20, 35), (x, y, width, height))
        pygame.draw.rect(screen, (60, 50, 80), (x, y, width, height), 1)
        
        # Circle parameters
        center_x = x + width // 2
        center_y = y + height // 2
        radius = min(width, height) // 2 - 20
        inner_radius = radius * 0.6
        
        # Draw circles
        pygame.draw.circle(screen, (50, 40, 70), (center_x, center_y), radius, 2)
        pygame.draw.circle(screen, (40, 30, 60), (center_x, center_y), int(inner_radius), 1)
        
        # Get current key info
        current_key = self.chromagram_info.get('key', 'C Major')
        current_pos, is_major = MusicTheory.get_circle_position(current_key)
        
        # Draw major keys (outer circle)
        for i, key in enumerate(MusicTheory.CIRCLE_OF_FIFTHS):
            angle = (i * 30 - 90) * math.pi / 180  # Start at top
            x_pos = center_x + int(radius * 0.85 * math.cos(angle))
            y_pos = center_y + int(radius * 0.85 * math.sin(angle))
            
            # Highlight current key
            if i == current_pos and is_major:
                pygame.draw.circle(screen, (255, 200, 100), (x_pos, y_pos), 15)
            
            if self.font_tiny:
                # Color based on distance from current key
                dist = min(abs(i - current_pos), 12 - abs(i - current_pos))
                brightness = 255 - (dist * 30)
                color = (brightness, brightness, brightness)
                
                key_surf = self.font_tiny.render(key, True, color)
                key_rect = key_surf.get_rect(center=(x_pos, y_pos))
                screen.blit(key_surf, key_rect)
        
        # Draw minor keys (inner circle)
        for i, key in enumerate(MusicTheory.CIRCLE_OF_FIFTHS_MINOR):
            angle = (i * 30 - 90) * math.pi / 180
            x_pos = center_x + int(inner_radius * 0.85 * math.cos(angle))
            y_pos = center_y + int(inner_radius * 0.85 * math.sin(angle))
            
            # Highlight current key
            if i == current_pos and not is_major:
                pygame.draw.circle(screen, (255, 150, 200), (x_pos, y_pos), 12)
            
            if self.font_tiny:
                # Color based on distance from current key
                dist = min(abs(i - current_pos), 12 - abs(i - current_pos))
                brightness = 200 - (dist * 25)
                color = (brightness, brightness * 0.8, brightness)
                
                key_surf = self.font_tiny.render(key, True, color)
                key_rect = key_surf.get_rect(center=(x_pos, y_pos))
                screen.blit(key_surf, key_rect)
        
        # Draw relationships
        if is_major:
            # Draw line to relative minor
            angle = (current_pos * 30 - 90) * math.pi / 180
            outer_x = center_x + int(radius * 0.85 * math.cos(angle))
            outer_y = center_y + int(radius * 0.85 * math.sin(angle))
            inner_x = center_x + int(inner_radius * 0.85 * math.cos(angle))
            inner_y = center_y + int(inner_radius * 0.85 * math.sin(angle))
            pygame.draw.line(screen, (100, 100, 150), (outer_x, outer_y), (inner_x, inner_y), 2)
        
        # Title
        if self.font_small:
            title = self.font_small.render("Circle of Fifths", True, (180, 170, 200))
            title_rect = title.get_rect(centerx=center_x, y=y + 5)
            screen.blit(title, title_rect)
    
    def _draw_chord_progression(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw chord progression with Roman numerals and T/S/D indicators"""
        # Background
        pygame.draw.rect(screen, (20, 15, 30), (x, y, width, height))
        pygame.draw.rect(screen, (50, 40, 70), (x, y, width, height), 1)
        
        # Get progression data
        chord_prog = self.chromagram_info.get('chord_progression', [])
        roman_prog = self.chromagram_info.get('roman_progression', [])
        
        if not chord_prog or not self.font_small:
            return
        
        # Title
        title = self.font_small.render("Chord Progression", True, (180, 170, 200))
        screen.blit(title, (x + 5, y + 2))
        
        # Draw progression
        bar_width = width // 8  # Show 8 chords
        bar_y = y + 20
        bar_height = height - 25
        
        start_idx = max(0, len(chord_prog) - 8)
        for i, (chord, roman) in enumerate(zip(chord_prog[start_idx:], roman_prog[start_idx:])):
            bar_x = x + i * bar_width
            
            # Get TSD group
            tsd_group = MusicTheory.get_tsd_group(roman)
            
            # TSD colors
            tsd_colors = {
                'Tonic': (50, 100, 50),
                'Subdominant': (100, 100, 50),
                'Dominant': (100, 50, 50),
                'Other': (50, 50, 50)
            }
            
            # Draw background bar
            pygame.draw.rect(screen, tsd_colors.get(tsd_group, (50, 50, 50)), 
                           (bar_x + 2, bar_y, bar_width - 4, bar_height))
            
            # Draw chord name
            if self.font_tiny:
                chord_surf = self.font_tiny.render(chord[:4], True, (200, 200, 220))
                chord_rect = chord_surf.get_rect(centerx=bar_x + bar_width//2, y=bar_y + 5)
                screen.blit(chord_surf, chord_rect)
                
                # Draw Roman numeral
                roman_surf = self.font_tiny.render(roman, True, (255, 255, 255))
                roman_rect = roman_surf.get_rect(centerx=bar_x + bar_width//2, y=bar_y + 20)
                screen.blit(roman_surf, roman_rect)
                
                # Draw T/S/D indicator
                tsd_map = {'Tonic': 'T', 'Subdominant': 'S', 'Dominant': 'D', 'Other': '?'}
                tsd_text = tsd_map.get(tsd_group, '?')
                tsd_surf = self.font_tiny.render(tsd_text, True, (255, 200, 100))
                tsd_rect = tsd_surf.get_rect(centerx=bar_x + bar_width//2, bottom=bar_y + bar_height - 2)
                screen.blit(tsd_surf, tsd_rect)
    
    def _draw_transition_matrix(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw chord transition probability matrix"""
        transitions = self.chromagram_info.get('chord_transitions', {})
        if not transitions or not self.font_tiny:
            return
        
        # Background
        pygame.draw.rect(screen, (20, 15, 30), (x, y, width, height))
        pygame.draw.rect(screen, (50, 40, 70), (x, y, width, height), 1)
        
        # Title
        if self.font_small:
            title = self.font_small.render("Chord Transitions", True, (180, 170, 200))
            screen.blit(title, (x + 5, y + 2))
        
        # Get most common transitions
        all_transitions = []
        for from_chord, to_chords in transitions.items():
            for to_chord, prob in to_chords.items():
                all_transitions.append((from_chord, to_chord, prob))
        
        # Sort by probability
        all_transitions.sort(key=lambda x: x[2], reverse=True)
        
        # Draw top transitions in two columns if space allows
        y_pos = y + 25
        col_width = width // 2
        items_per_col = min(4, (height - 30) // 15)
        
        for i, (from_chord, to_chord, prob) in enumerate(all_transitions[:items_per_col * 2]):
            if i >= items_per_col * 2:
                break
                
            # Determine column
            col = i // items_per_col
            row = i % items_per_col
            x_pos = x + 5 + (col * col_width)
            y_pos = y + 25 + (row * 15)
            
            if y_pos > y + height - 15:
                break
            
            # Format transition text
            trans_text = f"{from_chord} → {to_chord}: {prob:.0%}"
            
            # Color based on probability
            color_intensity = int(100 + prob * 155)
            color = (color_intensity, color_intensity, color_intensity)
            
            # Truncate if needed
            if self.font_tiny:
                trans_surf = self.font_tiny.render(trans_text[:25], True, color)
                screen.blit(trans_surf, (x_pos, y_pos))
    
    def get_results(self) -> Dict[str, Any]:
        """Get current chromagram and key detection results"""
        return self.chromagram_info.copy()