"""
Genre Classification Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract genre classification as self-contained module
OMEGA-2 Feature: Real-time genre detection
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import deque
from .hip_hop_detector import HipHopDetector


class GenreClassifier:
    """Real-time genre classification using audio feature analysis"""
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Genre definitions with characteristic features
        self.genres = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic', 
                      'Hip-Hop', 'Metal', 'Country', 'R&B', 'Folk']
        
        # Feature history for temporal analysis
        self.feature_buffer = deque(maxlen=43)  # ~1 second at 43 FPS
        self.genre_history = deque(maxlen=30)  # Smooth genre detection
        
        # Initialize dedicated hip-hop detector
        self.hip_hop_detector = HipHopDetector(sample_rate)
        
        # Genre-specific feature patterns (enhanced with harmonic features)
        self.genre_patterns = {
            'Rock': {
                'tempo_range': (100, 140),
                'spectral_centroid': (1500, 3000),
                'zero_crossing_rate': (0.05, 0.15),
                'spectral_rolloff': (3000, 5000),
                'dynamic_range': (0.4, 0.8),
                'percussion_strength': (0.6, 1.0),
                # New harmonic features
                'harmonic_complexity': (0.2, 0.5),
                'chord_change_rate': (0.3, 0.6),
                'key_stability': (0.6, 0.9),
                'pitch_class_concentration': (0.7, 0.9)  # Power chords = concentrated
            },
            'Pop': {
                'tempo_range': (100, 130),
                'spectral_centroid': (2000, 3500),
                'zero_crossing_rate': (0.08, 0.18),
                'spectral_rolloff': (3500, 6000),
                'dynamic_range': (0.3, 0.6),
                'percussion_strength': (0.5, 0.8),
                # New harmonic features
                'harmonic_complexity': (0.1, 0.4),  # Simple triads
                'chord_change_rate': (0.4, 0.7),    # Regular changes
                'key_stability': (0.8, 1.0),        # Very stable
                'pitch_class_concentration': (0.6, 0.8)  # Diatonic
            },
            'Jazz': {
                'tempo_range': (60, 180),
                'spectral_centroid': (1000, 2500),
                'zero_crossing_rate': (0.03, 0.12),
                'spectral_rolloff': (2500, 4500),
                'dynamic_range': (0.5, 0.9),
                'percussion_strength': (0.3, 0.7),
                # New harmonic features
                'harmonic_complexity': (0.6, 1.0),  # Extended chords
                'chord_change_rate': (0.5, 0.9),    # Frequent changes
                'key_stability': (0.2, 0.6),        # Modulations common
                'pitch_class_concentration': (0.3, 0.6)  # Chromatic
            },
            'Classical': {
                'tempo_range': (40, 180),
                'spectral_centroid': (1200, 3000),  # Higher than hip-hop
                'zero_crossing_rate': (0.02, 0.08),
                'spectral_rolloff': (3000, 6000),   # Higher frequency content
                'dynamic_range': (0.6, 1.0),
                'percussion_strength': (0.0, 0.2),   # Less percussion
                # New harmonic features
                'harmonic_complexity': (0.4, 0.8),   # More complex harmony
                'chord_change_rate': (0.2, 0.5),    # Moderate changes
                'key_stability': (0.7, 0.95),       # Mostly stable
                'pitch_class_concentration': (0.4, 0.7)  # More chromatic
            },
            'Electronic': {
                'tempo_range': (120, 140),
                'spectral_centroid': (2000, 4000),
                'zero_crossing_rate': (0.1, 0.25),
                'spectral_rolloff': (4000, 8000),
                'dynamic_range': (0.2, 0.5),
                'percussion_strength': (0.7, 1.0),
                # New harmonic features
                'harmonic_complexity': (0.0, 0.3),  # Minimal harmony
                'chord_change_rate': (0.0, 0.3),    # Static harmony
                'key_stability': (0.9, 1.0),        # Very stable/drone
                'pitch_class_concentration': (0.8, 1.0)  # Limited pitch set
            },
            'Hip-Hop': {
                'tempo_range': (60, 110),  # Expanded for trap and boom bap
                'spectral_centroid': (800, 2500),  # Lower center for bass emphasis
                'zero_crossing_rate': (0.04, 0.20),  # Wider range for vocals
                'spectral_rolloff': (2000, 5000),  # Adjusted for bass-heavy mix
                'dynamic_range': (0.4, 0.8),  # Higher for punchy drums
                'percussion_strength': (0.85, 1.0),  # Very strong beats
                # New harmonic features
                'harmonic_complexity': (0.1, 0.3),  # Simple harmony/loops
                'chord_change_rate': (0.0, 0.3),    # Static or slow changes
                'key_stability': (0.9, 1.0),        # Extremely stable
                'pitch_class_concentration': (0.8, 1.0),  # Very limited notes
                # Hip-hop specific features
                'bass_emphasis': (0.7, 1.0),  # Strong sub-bass
                'beat_regularity': (0.8, 1.0),  # Very regular beat patterns
                'vocal_presence': (0.6, 1.0),  # Prominent vocals/rap
            },
            'Metal': {
                'tempo_range': (80, 200),
                'spectral_centroid': (2000, 4000),  # Lowered - metal can be darker
                'zero_crossing_rate': (0.12, 0.25),  # Slightly lowered
                'spectral_rolloff': (3500, 6500),   # Lowered a bit
                'dynamic_range': (0.2, 0.5),        # Increased upper range
                'percussion_strength': (0.7, 1.0),
                # New harmonic features
                'harmonic_complexity': (0.2, 0.6),  # Power chords to complex riffs
                'chord_change_rate': (0.5, 0.9),    # Slightly lower minimum
                'key_stability': (0.5, 0.8),        # Modal shifts
                'pitch_class_concentration': (0.6, 0.8)  # Minor/modal scales
            },
            'Country': {
                'tempo_range': (80, 120),
                'spectral_centroid': (1200, 2500),
                'zero_crossing_rate': (0.04, 0.12),
                'spectral_rolloff': (2500, 4000),
                'dynamic_range': (0.4, 0.7),
                'percussion_strength': (0.4, 0.7),
                # New harmonic features
                'harmonic_complexity': (0.1, 0.4),  # Simple triads
                'chord_change_rate': (0.3, 0.6),    # Standard progressions
                'key_stability': (0.8, 1.0),        # Very stable
                'pitch_class_concentration': (0.6, 0.8)  # Major scales
            },
            'R&B': {
                'tempo_range': (60, 100),
                'spectral_centroid': (1500, 3000),
                'zero_crossing_rate': (0.05, 0.12),
                'spectral_rolloff': (3000, 5000),
                'dynamic_range': (0.3, 0.6),
                'percussion_strength': (0.5, 0.8),
                # New harmonic features
                'harmonic_complexity': (0.3, 0.7),  # 7th chords common
                'chord_change_rate': (0.3, 0.6),    # Smooth progressions
                'key_stability': (0.6, 0.9),        # Mostly stable
                'pitch_class_concentration': (0.5, 0.7)  # Soul harmonies
            },
            'Folk': {
                'tempo_range': (70, 120),
                'spectral_centroid': (1000, 2000),
                'zero_crossing_rate': (0.03, 0.10),
                'spectral_rolloff': (2000, 3500),
                'dynamic_range': (0.5, 0.8),
                'percussion_strength': (0.2, 0.5),
                # New harmonic features
                'harmonic_complexity': (0.1, 0.3),  # Very simple
                'chord_change_rate': (0.2, 0.5),    # Traditional patterns
                'key_stability': (0.9, 1.0),        # Extremely stable
                'pitch_class_concentration': (0.7, 0.9)  # Pentatonic/major
            }
        }
        
    def extract_features(self, fft_data: np.ndarray, audio_chunk: np.ndarray,
                        freqs: np.ndarray, drum_info: Dict, harmonic_info: Dict,
                        chromagram_data: np.ndarray = None,
                        chord_history: List[str] = None,
                        key_history: List[str] = None) -> Dict[str, float]:
        """Extract comprehensive audio features for genre classification
        
        Enhanced with harmonic features from chromagram analysis
        """
        features = {}
        
        # Use provided frequency array
        magnitude = np.abs(fft_data)
        features['spectral_centroid'] = self.compute_spectral_centroid(freqs, magnitude)
        
        # Spectral rolloff (high frequency content)
        features['spectral_rolloff'] = self.compute_spectral_rolloff(freqs, magnitude)
        
        # Zero crossing rate (percussiveness)
        features['zero_crossing_rate'] = self.compute_zero_crossing_rate(audio_chunk)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = self.compute_spectral_bandwidth(freqs, magnitude, features['spectral_centroid'])
        
        # Dynamic range estimation
        features['dynamic_range'] = self.estimate_dynamic_range(audio_chunk)
        
        # Percussion strength from drum detection
        kick_mag = drum_info.get('kick', {}).get('magnitude', 0)
        snare_mag = drum_info.get('snare', {}).get('magnitude', 0)
        features['percussion_strength'] = max(kick_mag, snare_mag)
        
        # Enhanced harmonic complexity calculation
        harmonic_series = harmonic_info.get('harmonic_series', [])
        if harmonic_series:
            # Consider both number and distribution of harmonics
            num_harmonics = len(harmonic_series)
            # Hip-hop typically has fewer, simpler harmonics
            # Classical has more complex harmonic structures
            if num_harmonics <= 3:
                features['harmonic_complexity'] = 0.1  # Very simple (hip-hop/electronic)
            elif num_harmonics <= 5:
                features['harmonic_complexity'] = 0.3  # Simple (pop/rock)
            elif num_harmonics <= 8:
                features['harmonic_complexity'] = 0.5  # Moderate (jazz/classical)
            else:
                features['harmonic_complexity'] = 0.7  # Complex (classical/jazz)
        else:
            # Calculate harmonic complexity from FFT data if harmonic series not available
            if fft_data is not None and len(fft_data) > 10:
                # Use spectral flatness as proxy for harmonic complexity
                magnitude = np.abs(fft_data)
                geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
                arithmetic_mean = np.mean(magnitude)
                if arithmetic_mean > 0:
                    spectral_flatness = geometric_mean / arithmetic_mean
                    # Convert to complexity score (flatter = simpler)
                    features['harmonic_complexity'] = 1.0 - spectral_flatness
                else:
                    features['harmonic_complexity'] = 0.1
            else:
                features['harmonic_complexity'] = 0.1  # Default to simple
        
        # Tempo estimation (simplified)
        features['estimated_tempo'] = self.estimate_tempo_from_transients(drum_info)
        
        # NEW: Enhanced harmonic features from chromagram
        if chromagram_data is not None:
            features['pitch_class_concentration'] = self.compute_pitch_class_concentration(chromagram_data)
            features['pitch_class_entropy'] = self.compute_pitch_class_entropy(chromagram_data)
        else:
            features['pitch_class_concentration'] = 0.5
            features['pitch_class_entropy'] = 0.5
            
        # NEW: Chord change rate from chord history
        if chord_history is not None and len(chord_history) > 1:
            features['chord_change_rate'] = self.compute_chord_change_rate(chord_history)
        else:
            features['chord_change_rate'] = 0.0
            
        # NEW: Key stability from key history
        if key_history is not None and len(key_history) > 1:
            features['key_stability'] = self.compute_key_stability(key_history)
        else:
            features['key_stability'] = 1.0
            
        # NEW: Harmonic rhythm (chord changes per minute)
        features['harmonic_rhythm'] = features['chord_change_rate'] * 60.0  # Assuming ~1 sec buffer
        
        # NEW: Tonal vs atonal ratio
        features['tonality_score'] = self.compute_tonality_score(chromagram_data)
        
        # Hip-hop specific features
        features['bass_emphasis'] = self.compute_bass_emphasis(fft_data, freqs)
        features['beat_regularity'] = self.compute_beat_regularity(drum_info)
        features['vocal_presence'] = self.compute_vocal_presence(fft_data, freqs)
        
        return features
    
    def compute_spectral_centroid(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate the spectral centroid (center of mass of spectrum)"""
        if np.sum(magnitude) == 0:
            return 0
        return np.sum(freqs[:len(magnitude)] * magnitude) / np.sum(magnitude)
    
    def compute_spectral_rolloff(self, freqs: np.ndarray, magnitude: np.ndarray, percentile: float = 0.85) -> float:
        """Calculate frequency below which percentile of energy is contained"""
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] == 0:
            return 0
        rolloff_idx = np.where(cumsum >= percentile * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            return freqs[min(rolloff_idx[0], len(freqs)-1)]
        return 0
    
    def compute_zero_crossing_rate(self, audio_chunk: np.ndarray) -> float:
        """Calculate rate of sign changes in the signal"""
        zero_crossings = np.where(np.diff(np.sign(audio_chunk)))[0]
        return len(zero_crossings) / len(audio_chunk)
    
    def compute_spectral_bandwidth(self, freqs: np.ndarray, magnitude: np.ndarray, centroid: float) -> float:
        """Calculate the weighted standard deviation around spectral centroid"""
        if np.sum(magnitude) == 0:
            return 0
        deviation = np.sqrt(np.sum(((freqs[:len(magnitude)] - centroid) ** 2) * magnitude) / np.sum(magnitude))
        return deviation
    
    def estimate_dynamic_range(self, audio_chunk: np.ndarray) -> float:
        """Estimate dynamic range as ratio of loud to quiet parts"""
        if len(audio_chunk) == 0:
            return 0
        # Use percentiles to avoid outliers
        loud = np.percentile(np.abs(audio_chunk), 95)
        quiet = np.percentile(np.abs(audio_chunk), 20)
        if quiet > 0:
            return min(1.0, (loud - quiet) / loud)
        return 0
    
    def estimate_tempo_from_transients(self, drum_info: Dict) -> float:
        """Simple tempo estimation from drum hits"""
        # This is a placeholder - real implementation would track beat intervals
        kick_detected = drum_info.get('kick', {}).get('kick_detected', False)
        if kick_detected:
            return 120  # Default estimate
        return 0
    
    def classify(self, features: Dict[str, float], audio_chunk: np.ndarray = None,
                 fft_data: np.ndarray = None, freqs: np.ndarray = None,
                 drum_info: Dict = None) -> Dict[str, float]:
        """Classify genre based on extracted features"""
        genre_scores = {}
        
        # Use dedicated hip-hop detector for hip-hop classification
        hip_hop_analysis = None
        if audio_chunk is not None and fft_data is not None and freqs is not None and drum_info is not None:
            hip_hop_analysis = self.hip_hop_detector.analyze(audio_chunk, fft_data, freqs, drum_info)
        
        for genre, patterns in self.genre_patterns.items():
            # Special handling for Hip-Hop using dedicated detector
            if genre == 'Hip-Hop' and hip_hop_analysis is not None:
                # Use the dedicated hip-hop detector's confidence
                genre_scores[genre] = hip_hop_analysis['confidence']
                # Store subgenre info for later use
                self._last_hip_hop_analysis = hip_hop_analysis
                continue
            
            score = 0.0
            weight_sum = 0.0
            
            # Spectral centroid matching
            if 'spectral_centroid' in patterns and features.get('spectral_centroid', 0) > 0:
                min_val, max_val = patterns['spectral_centroid']
                centroid = features['spectral_centroid']
                if min_val <= centroid <= max_val:
                    score += 1.0
                else:
                    # Partial score based on distance
                    if centroid < min_val:
                        score += max(0, 1 - (min_val - centroid) / min_val)
                    else:
                        score += max(0, 1 - (centroid - max_val) / max_val)
                weight_sum += 1.0
            
            # Zero crossing rate matching
            if 'zero_crossing_rate' in patterns and features.get('zero_crossing_rate', 0) > 0:
                min_val, max_val = patterns['zero_crossing_rate']
                zcr = features['zero_crossing_rate']
                if min_val <= zcr <= max_val:
                    score += 0.8
                else:
                    if zcr < min_val:
                        score += max(0, 0.8 - (min_val - zcr) / min_val)
                    else:
                        score += max(0, 0.8 - (zcr - max_val) / max_val)
                weight_sum += 0.8
            
            # Percussion strength matching
            if 'percussion_strength' in patterns:
                min_val, max_val = patterns['percussion_strength']
                perc = features.get('percussion_strength', 0)
                if min_val <= perc <= max_val:
                    score += 0.9
                else:
                    if perc < min_val:
                        score += max(0, 0.9 - (min_val - perc))
                    else:
                        score += max(0, 0.9 - (perc - max_val))
                weight_sum += 0.9
            
            # Dynamic range matching
            if 'dynamic_range' in patterns and features.get('dynamic_range', 0) > 0:
                min_val, max_val = patterns['dynamic_range']
                dr = features['dynamic_range']
                if min_val <= dr <= max_val:
                    score += 0.7
                else:
                    if dr < min_val:
                        score += max(0, 0.7 - (min_val - dr))
                    else:
                        score += max(0, 0.7 - (dr - max_val))
                weight_sum += 0.7
            
            # Check harmonic features (important for genre distinction)
            if 'harmonic_complexity' in patterns and 'harmonic_complexity' in features:
                min_val, max_val = patterns['harmonic_complexity']
                harm_comp = features.get('harmonic_complexity', 0.5)
                if min_val <= harm_comp <= max_val:
                    score += 0.8
                else:
                    # Penalize being outside range
                    if harm_comp < min_val:
                        score += max(0, 0.8 - 2 * (min_val - harm_comp))
                    else:
                        score += max(0, 0.8 - 2 * (harm_comp - max_val))
                weight_sum += 0.8
            
            if 'chord_change_rate' in patterns and 'chord_change_rate' in features:
                min_val, max_val = patterns['chord_change_rate']
                chord_rate = features.get('chord_change_rate', 0.3)
                if min_val <= chord_rate <= max_val:
                    score += 0.6
                else:
                    score += max(0, 0.6 - 2 * abs(chord_rate - (min_val + max_val) / 2))
                weight_sum += 0.6
            
            if 'key_stability' in patterns and 'key_stability' in features:
                min_val, max_val = patterns['key_stability']
                key_stab = features.get('key_stability', 0.8)
                if min_val <= key_stab <= max_val:
                    score += 0.5
                else:
                    score += max(0, 0.5 - 2 * abs(key_stab - (min_val + max_val) / 2))
                weight_sum += 0.5
            
            # Hip-hop specific features
            if genre == 'Hip-Hop':
                # Bass emphasis (very important for hip-hop)
                if 'bass_emphasis' in features:
                    bass_emp = features['bass_emphasis']
                    if bass_emp >= 0.7:  # Strong bass
                        score += 1.5  # Very high weight for bass
                    else:
                        score += max(0, 1.5 * (bass_emp / 0.7))
                    weight_sum += 1.5
                
                # Beat regularity
                if 'beat_regularity' in patterns and 'beat_regularity' in features:
                    min_val, max_val = patterns['beat_regularity']
                    beat_reg = features['beat_regularity']
                    if min_val <= beat_reg <= max_val:
                        score += 1.0
                    else:
                        score += max(0, 1.0 - 2 * (min_val - beat_reg))
                    weight_sum += 1.0
                
                # Vocal presence
                if 'vocal_presence' in patterns and 'vocal_presence' in features:
                    min_val, max_val = patterns['vocal_presence']
                    vocal = features['vocal_presence']
                    if min_val <= vocal <= max_val:
                        score += 0.8
                    else:
                        score += max(0, 0.8 - (min_val - vocal))
                    weight_sum += 0.8
            
            # Classical-specific penalties (classical should NOT have these features)
            if genre == 'Classical':
                # Penalize classical for high bass emphasis
                if 'bass_emphasis' in features:
                    bass_emp = features.get('bass_emphasis', 0)
                    if bass_emp > 0.5:  # Too much bass for classical
                        score -= 0.8 * bass_emp
                        weight_sum += 0.8
                
                # Penalize classical for high percussion
                if features.get('percussion_strength', 0) > 0.6:
                    score -= 0.6
                    weight_sum += 0.6
                
                # Penalize classical for vocal presence in pop/hip-hop range
                if 'vocal_presence' in features and features['vocal_presence'] > 0.7:
                    score -= 0.5
                    weight_sum += 0.5
            
            # Boost hip-hop score if multiple hip-hop features are present
            if genre == 'Hip-Hop':
                hip_hop_features = 0
                if features.get('bass_emphasis', 0) > 0.6:
                    hip_hop_features += 1
                if features.get('beat_regularity', 0) > 0.7:
                    hip_hop_features += 1
                if features.get('percussion_strength', 0) > 0.8:
                    hip_hop_features += 1
                if features.get('key_stability', 0) > 0.9:
                    hip_hop_features += 1
                
                # Synergy bonus - multiple hip-hop features together
                if hip_hop_features >= 3:
                    score += 0.5  # Bonus for having multiple hip-hop characteristics
                    weight_sum += 0.5
            
            # Normalize score
            if weight_sum > 0:
                genre_scores[genre] = score / weight_sum
            else:
                genre_scores[genre] = 0.0
                
            # Apply genre-specific penalties/bonuses to avoid defaulting to Folk/Electronic
            if genre == 'Folk':
                # Folk should have low percussion and simple harmony
                if features.get('percussion_strength', 0) > 0.7:
                    genre_scores[genre] *= 0.3  # Heavy penalty for strong percussion
                if features.get('bass_emphasis', 0) > 0.6:
                    genre_scores[genre] *= 0.5  # Penalty for bass emphasis
                    
            elif genre == 'Electronic':
                # Electronic should have steady beats and synthetic timbres
                if features.get('harmonic_complexity', 0) > 0.5:
                    genre_scores[genre] *= 0.6  # Penalty for complex harmony
                if features.get('percussion_strength', 0) < 0.5:
                    genre_scores[genre] *= 0.5  # Electronic needs beats
        
        # Add temporal smoothing
        self.genre_history.append(genre_scores)
        
        # Average over history for stability
        smoothed_scores = {}
        for genre in self.genres:
            scores = [h.get(genre, 0) for h in self.genre_history]
            smoothed_scores[genre] = np.mean(scores) if scores else 0
        
        # Apply confidence boost to top genres
        # This helps differentiate when all genres have similar low scores
        if smoothed_scores:
            max_score = max(smoothed_scores.values())
            if max_score > 0 and max_score < 0.3:  # Low confidence scenario
                # Boost top genre to make it more distinguishable
                for genre in smoothed_scores:
                    if smoothed_scores[genre] == max_score:
                        smoothed_scores[genre] *= 1.5
                        
            # Additional check: if Hip-Hop has significant features, boost it
            if 'Hip-Hop' in smoothed_scores:
                hip_hop_features_present = 0
                if features.get('bass_emphasis', 0) > 0.7:
                    hip_hop_features_present += 1
                if features.get('percussion_strength', 0) > 0.8:
                    hip_hop_features_present += 1
                if features.get('beat_regularity', 0) > 0.8:
                    hip_hop_features_present += 1
                    
                if hip_hop_features_present >= 2:
                    smoothed_scores['Hip-Hop'] *= 1.3  # Boost hip-hop when features match
        
        # Normalize to probabilities
        total = sum(smoothed_scores.values())
        if total > 0:
            for genre in smoothed_scores:
                smoothed_scores[genre] /= total
        else:
            # If no scores, set all to equal probability
            for genre in self.genres:
                smoothed_scores[genre] = 1.0 / len(self.genres)
        
        return smoothed_scores
    
    def get_top_genres(self, probabilities: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N most likely genres"""
        sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres[:top_n]
    
    # NEW: Harmonic feature computation methods
    def compute_pitch_class_concentration(self, chromagram: np.ndarray) -> float:
        """Compute how concentrated the pitch classes are (0=uniform, 1=single pitch)"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.5
            
        # Normalize chromagram
        norm_chroma = chromagram / np.sum(chromagram)
        
        # Compute Gini coefficient (measure of inequality)
        sorted_chroma = np.sort(norm_chroma)
        n = len(sorted_chroma)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_chroma)) / (n * np.sum(sorted_chroma)) - (n + 1) / n
        
        return gini
    
    def compute_pitch_class_entropy(self, chromagram: np.ndarray) -> float:
        """Compute entropy of pitch class distribution (0=concentrated, 1=uniform)"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.0
            
        # Normalize to probability distribution
        probs = chromagram / np.sum(chromagram)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize to 0-1 range (max entropy for 12 pitch classes is log2(12))
        return entropy / np.log2(12)
    
    def compute_chord_change_rate(self, chord_history: List[str]) -> float:
        """Compute rate of chord changes (0=static, 1=constantly changing)"""
        if len(chord_history) < 2:
            return 0.0
            
        changes = sum(1 for i in range(1, len(chord_history)) 
                     if chord_history[i] != chord_history[i-1])
        
        return changes / (len(chord_history) - 1)
    
    def compute_key_stability(self, key_history: List[str]) -> float:
        """Compute key stability over time (0=constantly modulating, 1=stable)"""
        if len(key_history) < 2:
            return 1.0
            
        # Count key changes
        changes = sum(1 for i in range(1, len(key_history)) 
                     if key_history[i] != key_history[i-1])
        
        # Invert to get stability
        return 1.0 - (changes / (len(key_history) - 1))
    
    def compute_tonality_score(self, chromagram: np.ndarray) -> float:
        """Compute how tonal vs atonal the music is"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.5
            
        # Major and minor key profiles
        major_profile = np.array([1.0, 0.0, 0.5, 0.0, 0.7, 0.5, 0.0, 0.8, 0.0, 0.5, 0.0, 0.3])
        minor_profile = np.array([1.0, 0.0, 0.5, 0.7, 0.0, 0.5, 0.0, 0.8, 0.3, 0.0, 0.5, 0.0])
        
        # Normalize chromagram
        norm_chroma = chromagram / (np.sum(chromagram) + 1e-10)
        
        # Compute correlation with tonal profiles
        correlations = []
        for shift in range(12):
            # Shift profiles to all keys
            shifted_major = np.roll(major_profile, shift)
            shifted_minor = np.roll(minor_profile, shift)
            
            # Compute correlation
            major_corr = np.corrcoef(norm_chroma, shifted_major)[0, 1]
            minor_corr = np.corrcoef(norm_chroma, shifted_minor)[0, 1]
            
            correlations.extend([major_corr, minor_corr])
        
        # Return maximum correlation as tonality score
        return max(0, max(correlations))
    
    def compute_bass_emphasis(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Compute how much bass (20-250Hz) dominates the spectrum"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        # Ensure arrays have the same length
        if len(fft_data) != len(freqs):
            # If sizes don't match, truncate or pad to match
            min_len = min(len(fft_data), len(freqs))
            fft_data = fft_data[:min_len]
            freqs = freqs[:min_len]
        
        # Find indices for bass frequencies (20-250Hz)
        bass_mask = (freqs >= 20) & (freqs <= 250)
        mid_mask = (freqs > 250) & (freqs <= 2000)
        
        if not np.any(bass_mask) or not np.any(mid_mask):
            return 0.0
        
        # Calculate energy in different bands
        bass_energy = np.sum(fft_data[bass_mask] ** 2)
        mid_energy = np.sum(fft_data[mid_mask] ** 2)
        total_energy = np.sum(fft_data ** 2)
        
        if total_energy == 0:
            return 0.0
        
        # Bass emphasis is ratio of bass to total, weighted by bass/mid ratio
        bass_ratio = bass_energy / total_energy
        bass_to_mid = bass_energy / (mid_energy + 1e-10)
        
        # Hip-hop typically has very strong bass relative to mids
        emphasis = bass_ratio * min(bass_to_mid / 2.0, 1.0)  # Normalize by expected ratio
        
        return min(emphasis, 1.0)
    
    def compute_beat_regularity(self, drum_info: Dict) -> float:
        """Compute how regular/consistent the beat pattern is"""
        # For now, use drum detection strength as proxy
        # In a full implementation, this would track beat intervals
        kick_strength = drum_info.get('kick', {}).get('magnitude', 0)
        snare_strength = drum_info.get('snare', {}).get('magnitude', 0)
        
        # Hip-hop has very strong, regular kick and snare
        if kick_strength > 0.7 and snare_strength > 0.5:
            return 0.9  # Very regular
        elif kick_strength > 0.5 or snare_strength > 0.4:
            return 0.7  # Somewhat regular
        else:
            return 0.3  # Not very regular
    
    def compute_vocal_presence(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Detect presence of vocals in typical vocal frequency range"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        # Ensure arrays have the same length
        if len(fft_data) != len(freqs):
            # If sizes don't match, truncate or pad to match
            min_len = min(len(fft_data), len(freqs))
            fft_data = fft_data[:min_len]
            freqs = freqs[:min_len]
        
        # Vocal presence frequencies (200Hz - 4kHz with emphasis on 300-3kHz)
        vocal_mask = (freqs >= 200) & (freqs <= 4000)
        core_vocal_mask = (freqs >= 300) & (freqs <= 3000)
        
        if not np.any(vocal_mask):
            return 0.0
        
        # Look for energy concentration in vocal range
        vocal_energy = np.sum(fft_data[vocal_mask] ** 2)
        core_vocal_energy = np.sum(fft_data[core_vocal_mask] ** 2)
        total_energy = np.sum(fft_data ** 2)
        
        if total_energy == 0:
            return 0.0
        
        # Vocal presence is combination of overall vocal energy and core concentration
        vocal_ratio = vocal_energy / total_energy
        core_ratio = core_vocal_energy / vocal_energy if vocal_energy > 0 else 0
        
        # Hip-hop vocals are typically prominent and centered in core range
        presence = vocal_ratio * (0.5 + 0.5 * core_ratio)
        
        return min(presence * 2.0, 1.0)  # Scale up as vocals are usually prominent
    
    def get_hip_hop_analysis(self) -> Dict[str, any]:
        """Get detailed hip-hop analysis results"""
        if hasattr(self, '_last_hip_hop_analysis'):
            return self._last_hip_hop_analysis
        return None


class GenreClassificationPanel:
    """OMEGA-2 Genre Classification Panel with visualization"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.classifier = GenreClassifier(sample_rate)
        
        # Display state
        self.genre_info = {
            'genres': {},
            'top_genre': 'Unknown',
            'confidence': 0.0,
            'features': {},
            'top_3': []
        }
        
        # Storage for cross-panel data
        self.chromagram_data = None
        self.chord_history = deque(maxlen=30)  # ~30 frames of history
        self.key_history = deque(maxlen=30)
        
        # Expose classifier and features for integrated analysis
        self.genre_probabilities = {}
        self.features = {}
        self.current_genre = 'Unknown'
        self.genre_confidence = 0.0
        
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
        
    def update(self, fft_data: np.ndarray, audio_chunk: np.ndarray, 
               frequencies: np.ndarray, drum_info: Dict, harmonic_info: Dict,
               chromagram_data: np.ndarray = None, current_chord: str = None,
               detected_key: str = None):
        """Update genre classification with new audio data
        
        Enhanced to accept harmonic features from chromagram analysis
        """
        if fft_data is not None and len(fft_data) > 0 and audio_chunk is not None:
            # Check for silence - if RMS is very low, don't classify
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < 0.001:  # Silence threshold
                # Reset to unknown when silent
                self.genre_info = {
                    'genres': {'Unknown': 1.0},
                    'top_genre': 'Unknown',
                    'confidence': 0.0,
                    'features': {},
                    'top_3': [('Unknown', 1.0)]
                }
                self.genre_probabilities = {'Unknown': 1.0}
                self.features = {}
                self.current_genre = 'Unknown'
                self.genre_confidence = 0.0
                return
            
            # Update cross-panel data
            if chromagram_data is not None:
                self.chromagram_data = chromagram_data
            if current_chord is not None:
                self.chord_history.append(current_chord)
            if detected_key is not None:
                self.key_history.append(detected_key)
                
            # Extract features with harmonic data
            features = self.classifier.extract_features(
                fft_data, audio_chunk, frequencies, drum_info, harmonic_info,
                self.chromagram_data,
                list(self.chord_history),
                list(self.key_history)
            )
            
            # Classify genre with enhanced hip-hop detection
            genre_probabilities = self.classifier.classify(
                features, audio_chunk, fft_data, frequencies, drum_info
            )
            
            # Get top genres
            top_genres = self.classifier.get_top_genres(genre_probabilities, top_n=3)
            
            # Update genre info
            self.genre_info = {
                'genres': genre_probabilities,
                'top_genre': top_genres[0][0] if top_genres else 'Unknown',
                'confidence': top_genres[0][1] if top_genres else 0.0,
                'features': features,
                'top_3': top_genres
            }
            
            # Update exposed attributes for integration
            self.genre_probabilities = genre_probabilities.copy()
            self.features = features.copy()
            self.current_genre = self.genre_info['top_genre']
            self.genre_confidence = self.genre_info['confidence']
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """OMEGA-2: Draw genre classification information"""
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((40, 30, 25))  # Brown tint for genre theme
        screen.blit(overlay, (x, y))
        
        # Border
        pygame.draw.rect(
            screen, (160, 120, 80), (x, y, width, height), 2
        )
        
        y_offset = y + int(20 * ui_scale)
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("OMEGA-2 Genre Classification", True, (255, 220, 180))
            screen.blit(title_text, (x + int(20 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Get genre data
        top_genre = self.genre_info.get('top_genre', 'Unknown')
        confidence = self.genre_info.get('confidence', 0.0)
        top_3 = self.genre_info.get('top_3', [])
        features = self.genre_info.get('features', {})
        
        # Display detected genre
        if self.font_large:
            genre_color = (
                (255, 150, 150) if confidence < 0.3 else
                (255, 255, 150) if confidence < 0.5 else
                (150, 255, 150)
            )
            genre_text = f"Genre: {top_genre}"
            
            # Add subgenre info for hip-hop
            if top_genre == 'Hip-Hop':
                hip_hop_info = self.classifier.get_hip_hop_analysis()
                if hip_hop_info and hip_hop_info.get('subgenre'):
                    subgenre = hip_hop_info['subgenre']
                    if subgenre != 'unknown':
                        genre_text = f"Genre: {top_genre} ({subgenre})"
            
            genre_surf = self.font_large.render(genre_text, True, genre_color)
            screen.blit(genre_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Confidence
        if self.font_small:
            conf_text = f"Confidence: {confidence:.0%}"
            conf_surf = self.font_small.render(conf_text, True, (200, 200, 220))
            screen.blit(conf_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        # Draw genre probability bars
        bar_x = x + int(20 * ui_scale)
        bar_width = width - int(40 * ui_scale)
        bar_height = int(15 * ui_scale)
        
        # Show top 3 genres
        if self.font_tiny:
            for i, (genre, prob) in enumerate(top_3[:3]):
                # Genre name
                genre_name_surf = self.font_tiny.render(genre, True, (180, 180, 200))
                screen.blit(genre_name_surf, (bar_x, y_offset))
                
                # Probability text
                prob_text = f"{prob:.0%}"
                prob_surf = self.font_tiny.render(prob_text, True, (150, 150, 170))
                prob_rect = prob_surf.get_rect(right=x + width - int(20 * ui_scale), y=y_offset)
                screen.blit(prob_surf, prob_rect)
                
                y_offset += int(15 * ui_scale)
                
                # Background bar
                pygame.draw.rect(screen, (40, 40, 50), 
                               (bar_x, y_offset, bar_width, bar_height))
                
                # Probability fill
                fill_width = int(bar_width * prob)
                if i == 0:  # Top genre
                    bar_color = (100, 200, 100) if prob > 0.5 else (200, 200, 100)
                else:
                    bar_color = (100, 100, 150)
                
                pygame.draw.rect(screen, bar_color,
                               (bar_x, y_offset, fill_width, bar_height))
                
                # Border
                pygame.draw.rect(screen, (100, 100, 120),
                               (bar_x, y_offset, bar_width, bar_height), 1)
                
                y_offset += bar_height + int(5 * ui_scale)
        
        y_offset += int(10 * ui_scale)
        
        # Show key audio features
        if self.font_tiny and features:
            features_text = "Audio Features:"
            features_surf = self.font_tiny.render(features_text, True, (180, 180, 200))
            screen.blit(features_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(15 * ui_scale)
            
            # Display some key features
            centroid = features.get('spectral_centroid', 0)
            if centroid > 0:
                cent_text = f"  Brightness: {centroid:.0f} Hz"
                cent_surf = self.font_tiny.render(cent_text, True, (150, 150, 170))
                screen.blit(cent_surf, (x + int(20 * ui_scale), y_offset))
                y_offset += int(12 * ui_scale)
            
            perc = features.get('percussion_strength', 0)
            perc_text = f"  Percussion: {perc:.0%}"
            perc_surf = self.font_tiny.render(perc_text, True, (150, 150, 170))
            screen.blit(perc_surf, (x + int(20 * ui_scale), y_offset))
            y_offset += int(12 * ui_scale)
            
            dr = features.get('dynamic_range', 0)
            dr_text = f"  Dynamics: {dr:.0%}"
            dr_surf = self.font_tiny.render(dr_text, True, (150, 150, 170))
            screen.blit(dr_surf, (x + int(20 * ui_scale), y_offset))
    
    def get_results(self) -> Dict[str, Any]:
        """Get current genre classification results"""
        return self.genre_info.copy()