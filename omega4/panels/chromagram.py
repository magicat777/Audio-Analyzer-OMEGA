"""
Chromagram Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract chromagram and key detection as self-contained module
OMEGA-1 Feature: Musical key detection
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque


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
            '13': [0, 4, 7, 10, 2, 5, 9]  # Jazz extended chord
        }
        
        # History for temporal smoothing
        self.chroma_history = deque(maxlen=30)  # 0.5 seconds at 60 FPS
        self.key_history = deque(maxlen=60)  # 1 second for stability
        self.chord_history = deque(maxlen=30)  # Chord progression tracking
        
        # Current genre context
        self.current_genre = 'pop'  # Default to pop
        
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
        """Detect musical key using correlation with Krumhansl-Kessler profiles
        
        Enhanced with genre-aware weighting for better accuracy
        """
        best_key = None
        best_correlation = -1
        best_is_major = True
        
        # Get genre-specific weights if available
        genre_weights = self.genre_profiles.get(self.current_genre.lower(), self.genre_profiles['pop'])
        major_weights = genre_weights['major_weights']
        minor_weights = genre_weights['minor_weights']
        
        # Apply genre-specific chromagram weighting
        weighted_chroma = chroma * (major_weights + minor_weights) / 2
        
        # Test all 24 possible keys (12 major + 12 minor)
        for i in range(12):
            # Rotate profiles to test each key
            major_profile_rotated = np.roll(self.major_profile, i)
            minor_profile_rotated = np.roll(self.minor_profile, i)
            
            # Apply genre-specific weights to profiles
            major_profile_weighted = major_profile_rotated * np.roll(major_weights, i)
            minor_profile_weighted = minor_profile_rotated * np.roll(minor_weights, i)
            
            # Calculate correlation coefficients
            if np.std(weighted_chroma) > 0:  # Avoid division by zero
                major_corr = np.corrcoef(weighted_chroma, major_profile_weighted)[0, 1]
                minor_corr = np.corrcoef(weighted_chroma, minor_profile_weighted)[0, 1]
            else:
                major_corr = minor_corr = 0
            
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
    
    def detect_chord(self, chroma: np.ndarray) -> Tuple[str, float]:
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
        else:
            # Pop/Classical: only test basic triads and 7ths
            chord_types_to_test = [(k, v) for k, v in self.chord_types.items() 
                                  if k in ['major', 'minor', 'dim', 'aug', 'maj7', 'min7']]
        
        # Find strongest pitch classes
        top_pitches = np.argsort(chroma)[-5:][::-1]  # Top 5 pitches
        
        # Test each root note
        for root_idx in range(12):
            if chroma[root_idx] < 0.1:  # Skip weak root notes
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
                elif self.current_genre.lower() == 'pop' and chord_name in ['major', 'minor']:
                    score *= 1.1
                elif self.current_genre.lower() == 'classical' and chord_name in ['major', 'minor', 'dim']:
                    score *= 1.1
                
                if score > best_score:
                    best_score = score
                    best_chord = f"{self.note_names[root_idx]}{chord_name}"
        
        # Add to history
        self.chord_history.append(best_chord)
        
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
            'key': 'C Major',
            'is_major': True,
            'confidence': 0.0,
            'stability': 0.0,
            'alternative_keys': [],
            'current_chord': 'N/A',
            'chord_confidence': 0.0,
            'chord_progression': [],
            'harmonic_rhythm': 0.0
        }
        
        # Expose data for integration
        self.chromagram = np.zeros(12)
        self.detected_key = 'C Major'
        self.key_confidence = 0.0
        self.current_chord = 'N/A'
        self.chord_confidence = 0.0
        self.chord_sequence = []
        self.circle_position = 0  # Position on circle of fifths
        
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
               current_genre: str = None):
        """Update chromagram and key detection with new FFT data
        
        Enhanced to accept genre context and detect chords
        """
        if fft_data is not None and len(fft_data) > 0:
            # Update genre context if provided
            if current_genre:
                self.analyzer.set_genre_context(current_genre)
                
            # Compute chromagram
            chromagram = self.analyzer.compute_chromagram(fft_data, freqs)
            
            # Detect key
            key, correlation = self.analyzer.detect_key(chromagram)
            
            # Detect current chord
            chord, chord_conf = self.analyzer.detect_chord(chromagram)
            
            # Get stability
            stability = self.analyzer.get_key_stability()
            
            # Get most likely key with confidence
            most_likely_key, confidence = self.analyzer.get_most_likely_key()
            
            # Get alternative keys
            alt_keys = self.analyzer.get_alternative_keys(chromagram, top_n=3)
            
            # Get chord progression and harmonic rhythm
            chord_progression = self.analyzer.get_chord_progression()
            harmonic_rhythm = self.analyzer.analyze_harmonic_rhythm()
            
            # Calculate circle of fifths position
            self.circle_position = self._calculate_circle_position(most_likely_key)
            
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
                'harmonic_rhythm': harmonic_rhythm
            }
            
            # Update exposed attributes for integration
            self.chromagram = chromagram.copy()
            self.detected_key = most_likely_key
            self.key_confidence = confidence
            self.current_chord = chord
            self.chord_confidence = chord_conf
            self.chord_sequence = chord_progression
    
    def _calculate_circle_position(self, key: str) -> int:
        """Calculate position on circle of fifths (0-11)"""
        circle_order = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        # Extract root note from key
        key_root = key.split()[0]
        
        # Handle enharmonic equivalents
        enharmonics = {'Gb': 'F#', 'Db': 'C#', 'Ab': 'G#', 'Eb': 'D#', 'Bb': 'A#'}
        if key_root in enharmonics:
            key_root = enharmonics[key_root]
            
        try:
            return circle_order.index(key_root)
        except ValueError:
            return 0
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """OMEGA-1: Draw chromagram and musical key detection information"""
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((35, 25, 45))  # Purple tint for musical theme
        screen.blit(overlay, (x, y))
        
        # Border
        pygame.draw.rect(
            screen, (140, 100, 180), (x, y, width, height), 2
        )
        
        y_offset = y + int(20 * ui_scale)
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("OMEGA-1 Key Detection", True, (255, 200, 255))
            screen.blit(title_text, (x + int(20 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Get chromagram data
        chromagram = self.chromagram_info.get('chromagram', np.zeros(12))
        key = self.chromagram_info.get('key', 'C')
        is_major = self.chromagram_info.get('is_major', True)
        confidence = self.chromagram_info.get('confidence', 0.0)
        stability = self.chromagram_info.get('stability', 0.0)
        alt_keys = self.chromagram_info.get('alternative_keys', [])
        
        # Display detected key
        if self.font_large:
            key_color = (
                (255, 150, 150) if confidence < 0.5 else
                (255, 255, 150) if confidence < 0.7 else
                (150, 255, 150)
            )
            key_text = f"Key: {key}"
            key_surf = self.font_large.render(key_text, True, key_color)
            screen.blit(key_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Confidence and stability
        if self.font_small:
            conf_text = f"Confidence: {confidence:.0%}"
            conf_surf = self.font_small.render(conf_text, True, (200, 200, 220))
            screen.blit(conf_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(20 * ui_scale)
            
            stab_text = f"Stability: {stability:.0%}"
            stab_color = (
                (255, 150, 150) if stability < 0.3 else
                (255, 255, 150) if stability < 0.7 else
                (150, 255, 150)
            )
            stab_surf = self.font_small.render(stab_text, True, stab_color)
            screen.blit(stab_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        # Draw chromagram visualization
        chroma_x = x + int(20 * ui_scale)
        chroma_y = y_offset
        chroma_width = width - int(40 * ui_scale)
        chroma_height = int(80 * ui_scale)
        
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
            bar_height = int(chroma_height * min(value, 1.0))
            
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
        
        y_offset = chroma_y + chroma_height + int(25 * ui_scale)
        
        # Alternative keys if confidence is low
        if confidence < 0.7 and alt_keys and self.font_tiny:
            alt_text = "Alternative keys:"
            alt_surf = self.font_tiny.render(alt_text, True, (150, 150, 170))
            screen.blit(alt_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(15 * ui_scale)
            
            for alt_key, alt_conf in alt_keys[:2]:  # Show top 2 alternatives
                alt_key_text = f"  {alt_key}: {alt_conf:.0%}"
                alt_key_surf = self.font_tiny.render(alt_key_text, True, (130, 130, 150))
                screen.blit(alt_key_surf, (x + int(20 * ui_scale), y_offset))
                y_offset += int(15 * ui_scale)
    
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
    
    def get_results(self) -> Dict[str, Any]:
        """Get current chromagram and key detection results"""
        return self.chromagram_info.copy()