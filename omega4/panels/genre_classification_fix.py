"""
Genre Classification Panel - Fixed version
Fixes:
1. Reduced history buffer for faster response
2. Better feature extraction
3. More distinctive genre patterns
4. Fixed harmonic complexity calculation
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
        
        # Feature history for temporal analysis - REDUCED for faster response
        self.feature_buffer = deque(maxlen=10)  # Reduced from 43
        self.genre_history = deque(maxlen=5)   # Reduced from 30 - much faster response
        
        # Initialize dedicated hip-hop detector
        self.hip_hop_detector = HipHopDetector(sample_rate)
        
        # Genre-specific feature patterns with tighter ranges
        self.genre_patterns = {
            'Rock': {
                'tempo_range': (100, 140),
                'spectral_centroid': (1500, 3000),
                'zero_crossing_rate': (0.08, 0.18),  # Tightened
                'spectral_rolloff': (3000, 5000),
                'dynamic_range': (0.4, 0.8),
                'percussion_strength': (0.6, 1.0),
                'harmonic_complexity': (0.3, 0.6),   # Tightened
                'chord_change_rate': (0.3, 0.6),
                'key_stability': (0.6, 0.9),
                'pitch_class_concentration': (0.6, 0.8),
                'priority': 0.9  # Genre priority multiplier
            },
            'Pop': {
                'tempo_range': (100, 130),
                'spectral_centroid': (2000, 3500),
                'zero_crossing_rate': (0.08, 0.18),
                'spectral_rolloff': (3500, 6000),
                'dynamic_range': (0.3, 0.6),
                'percussion_strength': (0.5, 0.8),
                'harmonic_complexity': (0.2, 0.4),
                'chord_change_rate': (0.4, 0.7),
                'key_stability': (0.8, 1.0),
                'pitch_class_concentration': (0.6, 0.8),
                'priority': 0.8
            },
            'Jazz': {
                'tempo_range': (60, 180),
                'spectral_centroid': (1000, 2500),
                'zero_crossing_rate': (0.03, 0.12),
                'spectral_rolloff': (2500, 4500),
                'dynamic_range': (0.5, 0.9),
                'percussion_strength': (0.3, 0.7),
                'harmonic_complexity': (0.6, 1.0),
                'chord_change_rate': (0.5, 0.9),
                'key_stability': (0.2, 0.6),
                'pitch_class_concentration': (0.3, 0.6),
                'priority': 0.85
            },
            'Classical': {
                'tempo_range': (40, 180),
                'spectral_centroid': (1200, 3000),
                'zero_crossing_rate': (0.02, 0.08),
                'spectral_rolloff': (3000, 6000),
                'dynamic_range': (0.6, 1.0),
                'percussion_strength': (0.0, 0.3),   # Tightened
                'harmonic_complexity': (0.4, 0.8),
                'chord_change_rate': (0.2, 0.5),
                'key_stability': (0.7, 0.95),
                'pitch_class_concentration': (0.4, 0.7),
                'priority': 0.9
            },
            'Electronic': {
                'tempo_range': (120, 140),
                'spectral_centroid': (2000, 4000),
                'zero_crossing_rate': (0.1, 0.25),
                'spectral_rolloff': (4000, 8000),
                'dynamic_range': (0.2, 0.5),
                'percussion_strength': (0.7, 1.0),
                'harmonic_complexity': (0.0, 0.3),
                'chord_change_rate': (0.0, 0.3),
                'key_stability': (0.9, 1.0),
                'pitch_class_concentration': (0.8, 1.0),
                'priority': 0.8
            },
            'Hip-Hop': {
                'tempo_range': (60, 110),
                'spectral_centroid': (800, 2500),
                'zero_crossing_rate': (0.04, 0.20),
                'spectral_rolloff': (2000, 5000),
                'dynamic_range': (0.4, 0.8),
                'percussion_strength': (0.85, 1.0),
                'harmonic_complexity': (0.1, 0.3),
                'chord_change_rate': (0.0, 0.3),
                'key_stability': (0.9, 1.0),
                'pitch_class_concentration': (0.8, 1.0),
                'bass_emphasis': (0.7, 1.0),
                'beat_regularity': (0.8, 1.0),
                'vocal_presence': (0.6, 1.0),
                'priority': 1.0
            },
            'Metal': {
                'tempo_range': (80, 200),
                'spectral_centroid': (1800, 3500),   # Adjusted for metal
                'zero_crossing_rate': (0.15, 0.30),   # Higher for distortion
                'spectral_rolloff': (3500, 6500),
                'dynamic_range': (0.2, 0.4),         # Compressed
                'percussion_strength': (0.7, 1.0),
                'harmonic_complexity': (0.3, 0.7),   # Adjusted
                'chord_change_rate': (0.4, 0.8),
                'key_stability': (0.5, 0.8),
                'pitch_class_concentration': (0.6, 0.8),
                'priority': 1.1  # Higher priority for distinctive features
            },
            'Country': {
                'tempo_range': (80, 120),
                'spectral_centroid': (1400, 2800),   # Tightened
                'zero_crossing_rate': (0.05, 0.10),   # Tightened
                'spectral_rolloff': (2800, 4200),    # Tightened
                'dynamic_range': (0.4, 0.6),         # Tightened
                'percussion_strength': (0.4, 0.6),    # Tightened
                'harmonic_complexity': (0.2, 0.35),   # Tightened significantly
                'chord_change_rate': (0.3, 0.5),      # Tightened
                'key_stability': (0.85, 0.95),        # Tightened
                'pitch_class_concentration': (0.65, 0.75),  # Tightened
                'priority': 0.7  # Lower priority to prevent defaulting
            },
            'R&B': {
                'tempo_range': (60, 100),
                'spectral_centroid': (1500, 3000),
                'zero_crossing_rate': (0.05, 0.12),
                'spectral_rolloff': (3000, 5000),
                'dynamic_range': (0.3, 0.6),
                'percussion_strength': (0.5, 0.8),
                'harmonic_complexity': (0.3, 0.7),
                'chord_change_rate': (0.3, 0.6),
                'key_stability': (0.6, 0.9),
                'pitch_class_concentration': (0.5, 0.7),
                'priority': 0.85
            },
            'Folk': {
                'tempo_range': (70, 120),
                'spectral_centroid': (1000, 2000),
                'zero_crossing_rate': (0.03, 0.08),   # Tightened
                'spectral_rolloff': (2000, 3500),
                'dynamic_range': (0.5, 0.8),
                'percussion_strength': (0.1, 0.4),    # Tightened
                'harmonic_complexity': (0.1, 0.25),   # Tightened
                'chord_change_rate': (0.2, 0.4),      # Tightened
                'key_stability': (0.9, 1.0),
                'pitch_class_concentration': (0.7, 0.85),  # Tightened
                'priority': 0.6  # Lower priority
            }
        }
        
    def extract_features(self, fft_data: np.ndarray, audio_chunk: np.ndarray,
                        freqs: np.ndarray, drum_info: Dict, harmonic_info: Dict,
                        chromagram_data: np.ndarray = None,
                        chord_history: List[str] = None,
                        key_history: List[str] = None) -> Dict[str, float]:
        """Extract comprehensive audio features for genre classification"""
        features = {}
        
        # Ensure we have valid data
        if len(fft_data) == 0 or len(audio_chunk) == 0:
            return features
            
        # Use provided frequency array - ensure matching lengths
        magnitude = np.abs(fft_data)
        if len(freqs) != len(magnitude):
            # Create matching frequency array if needed
            freqs = np.fft.rfftfreq(len(audio_chunk), 1/self.sample_rate)[:len(magnitude)]
        
        features['spectral_centroid'] = self.compute_spectral_centroid(freqs, magnitude)
        features['spectral_rolloff'] = self.compute_spectral_rolloff(freqs, magnitude)
        features['zero_crossing_rate'] = self.compute_zero_crossing_rate(audio_chunk)
        features['spectral_bandwidth'] = self.compute_spectral_bandwidth(freqs, magnitude, features['spectral_centroid'])
        features['dynamic_range'] = self.estimate_dynamic_range(audio_chunk)
        
        # Percussion strength from drum detection
        kick_mag = drum_info.get('kick', {}).get('magnitude', 0)
        snare_mag = drum_info.get('snare', {}).get('magnitude', 0)
        features['percussion_strength'] = max(kick_mag, snare_mag)
        
        # Enhanced harmonic complexity calculation
        harmonic_series = harmonic_info.get('harmonic_series', [])
        if harmonic_series and len(harmonic_series) > 0:
            # Better harmonic complexity based on number and strength
            num_harmonics = len(harmonic_series)
            harmonic_strengths = [h.get('strength', 0) for h in harmonic_series if isinstance(h, dict)]
            
            if harmonic_strengths:
                avg_strength = np.mean(harmonic_strengths)
                # Complexity based on both number and strength of harmonics
                complexity = min(1.0, (num_harmonics / 10.0) * avg_strength)
                features['harmonic_complexity'] = complexity
            else:
                features['harmonic_complexity'] = min(1.0, num_harmonics / 10.0)
        else:
            # Calculate from spectral flatness if no harmonic info
            features['harmonic_complexity'] = self.compute_spectral_complexity(magnitude, freqs)
        
        # Tempo estimation
        features['estimated_tempo'] = self.estimate_tempo_from_transients(drum_info)
        
        # Chromagram features
        if chromagram_data is not None and len(chromagram_data) == 12:
            features['pitch_class_concentration'] = self.compute_pitch_class_concentration(chromagram_data)
            features['pitch_class_entropy'] = self.compute_pitch_class_entropy(chromagram_data)
        else:
            features['pitch_class_concentration'] = 0.5
            features['pitch_class_entropy'] = 0.5
            
        # Chord change rate
        if chord_history is not None and len(chord_history) > 1:
            features['chord_change_rate'] = self.compute_chord_change_rate(chord_history)
        else:
            features['chord_change_rate'] = 0.0
            
        # Key stability
        if key_history is not None and len(key_history) > 1:
            features['key_stability'] = self.compute_key_stability(key_history)
        else:
            features['key_stability'] = 1.0
            
        # Genre-specific features
        features['bass_emphasis'] = self.compute_bass_emphasis(magnitude, freqs)
        features['beat_regularity'] = self.compute_beat_regularity(drum_info)
        features['vocal_presence'] = self.compute_vocal_presence(magnitude, freqs)
        
        # High frequency content (for metal detection)
        features['high_freq_energy'] = self.compute_high_freq_energy(magnitude, freqs)
        
        return features
    
    def compute_spectral_complexity(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """Compute spectral complexity from magnitude spectrum"""
        if len(magnitude) == 0:
            return 0.3
            
        # Focus on harmonic range (100Hz - 4kHz)
        harmonic_mask = (freqs >= 100) & (freqs <= 4000)
        if not np.any(harmonic_mask):
            return 0.3
            
        harmonic_mag = magnitude[harmonic_mask]
        
        # Calculate spectral flatness in harmonic range
        safe_mag = harmonic_mag[harmonic_mag > 1e-10]
        if len(safe_mag) > 0:
            geometric_mean = np.exp(np.mean(np.log(safe_mag)))
            arithmetic_mean = np.mean(safe_mag)
            if arithmetic_mean > 0:
                flatness = geometric_mean / arithmetic_mean
                # Invert and scale - more peaks = more complex
                complexity = np.clip(1.0 - flatness, 0.0, 1.0)
                return complexity
        
        return 0.3
    
    def compute_high_freq_energy(self, magnitude: np.ndarray, freqs: np.ndarray) -> float:
        """Compute high frequency energy ratio (important for metal/rock)"""
        if len(magnitude) == 0 or len(freqs) == 0:
            return 0.0
            
        # High frequency band (4kHz - 10kHz)
        high_mask = (freqs >= 4000) & (freqs <= 10000)
        total_mask = freqs <= 10000
        
        if not np.any(high_mask) or not np.any(total_mask):
            return 0.0
            
        high_energy = np.sum(magnitude[high_mask] ** 2)
        total_energy = np.sum(magnitude[total_mask] ** 2)
        
        if total_energy > 0:
            return high_energy / total_energy
        return 0.0
    
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
            # Special handling for Hip-Hop
            if genre == 'Hip-Hop' and hip_hop_analysis is not None:
                genre_scores[genre] = hip_hop_analysis['confidence']
                self._last_hip_hop_analysis = hip_hop_analysis
                continue
            
            score = 0.0
            weight_sum = 0.0
            matches = 0
            total_features = 0
            
            # Check each feature
            for feature_name, (min_val, max_val) in patterns.items():
                if feature_name == 'priority':
                    continue
                    
                if feature_name in features:
                    feature_val = features[feature_name]
                    total_features += 1
                    
                    # Calculate match score
                    if min_val <= feature_val <= max_val:
                        score += 1.0
                        matches += 1
                    else:
                        # Partial score based on distance
                        if feature_val < min_val:
                            distance_score = max(0, 1 - (min_val - feature_val) / max(min_val, 0.1))
                        else:
                            distance_score = max(0, 1 - (feature_val - max_val) / max(max_val, 0.1))
                        score += distance_score * 0.5  # Partial credit
                    weight_sum += 1.0
            
            # Calculate base score
            if weight_sum > 0:
                base_score = score / weight_sum
            else:
                base_score = 0.0
                
            # Apply genre-specific modifiers
            
            # Metal detection boost
            if genre == 'Metal':
                if features.get('zero_crossing_rate', 0) > 0.15 and features.get('high_freq_energy', 0) > 0.3:
                    base_score *= 1.5  # Boost for high distortion
                if features.get('percussion_strength', 0) > 0.8:
                    base_score *= 1.2  # Boost for strong drums
                    
            # Country penalty for complex harmony
            elif genre == 'Country':
                if features.get('harmonic_complexity', 0) > 0.4:
                    base_score *= 0.5  # Penalty for complex harmony
                if features.get('spectral_centroid', 0) > 3000:
                    base_score *= 0.7  # Penalty for bright spectrum
                    
            # Folk penalty for strong percussion
            elif genre == 'Folk':
                if features.get('percussion_strength', 0) > 0.5:
                    base_score *= 0.4  # Heavy penalty for drums
                if features.get('bass_emphasis', 0) > 0.5:
                    base_score *= 0.6  # Penalty for bass
                    
            # Apply priority multiplier
            priority = patterns.get('priority', 1.0)
            genre_scores[genre] = base_score * priority
        
        # Add temporal smoothing with REDUCED history
        self.genre_history.append(genre_scores)
        
        # Average over shorter history for faster response
        smoothed_scores = {}
        for genre in self.genres:
            scores = [h.get(genre, 0) for h in self.genre_history]
            smoothed_scores[genre] = np.mean(scores) if scores else 0
        
        # Enhanced score differentiation
        if smoothed_scores:
            # Find top score
            max_score = max(smoothed_scores.values())
            
            # If all scores are low, boost the leader more aggressively
            if max_score > 0 and max_score < 0.4:
                # Find top 2 genres
                sorted_genres = sorted(smoothed_scores.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_genres) >= 2:
                    top_genre, top_score = sorted_genres[0]
                    second_genre, second_score = sorted_genres[1]
                    
                    # Boost top genre significantly if it's clearly ahead
                    if top_score > second_score * 1.2:
                        smoothed_scores[top_genre] *= 2.0
                    else:
                        smoothed_scores[top_genre] *= 1.5
        
        # Normalize with enhanced softmax
        total = sum(smoothed_scores.values())
        if total > 0:
            # Apply temperature-scaled softmax for better separation
            temp = 3.0  # Increased temperature for more separation
            exp_scores = {}
            
            # First normalize
            for genre in smoothed_scores:
                smoothed_scores[genre] /= total
            
            # Apply exponential scaling
            for genre in smoothed_scores:
                exp_scores[genre] = np.exp(smoothed_scores[genre] * temp)
            
            # Renormalize
            exp_total = sum(exp_scores.values())
            for genre in smoothed_scores:
                smoothed_scores[genre] = exp_scores[genre] / exp_total
                
            # Final boost for clear winner
            max_prob = max(smoothed_scores.values())
            if max_prob > 0.3:  # If we have a reasonably confident leader
                for genre in smoothed_scores:
                    if smoothed_scores[genre] == max_prob:
                        # Scale up the confidence
                        smoothed_scores[genre] = min(0.95, max_prob * 1.5)
                        break
                        
                # Renormalize one more time
                total = sum(smoothed_scores.values())
                for genre in smoothed_scores:
                    smoothed_scores[genre] /= total
        else:
            smoothed_scores = {'Unknown': 1.0}
        
        return smoothed_scores
    
    # Copy all the compute_* methods from the original file
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
        kick_detected = drum_info.get('kick', {}).get('kick_detected', False)
        if kick_detected:
            return 120  # Default estimate
        return 0
    
    def compute_pitch_class_concentration(self, chromagram: np.ndarray) -> float:
        """Compute how concentrated the pitch classes are"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.5
            
        norm_chroma = chromagram / np.sum(chromagram)
        sorted_chroma = np.sort(norm_chroma)
        n = len(sorted_chroma)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_chroma)) / (n * np.sum(sorted_chroma)) - (n + 1) / n
        
        return gini
    
    def compute_pitch_class_entropy(self, chromagram: np.ndarray) -> float:
        """Compute entropy of pitch class distribution"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.0
            
        probs = chromagram / np.sum(chromagram)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy / np.log2(12)
    
    def compute_chord_change_rate(self, chord_history: List[str]) -> float:
        """Compute rate of chord changes"""
        if len(chord_history) < 2:
            return 0.0
            
        changes = sum(1 for i in range(1, len(chord_history)) 
                     if chord_history[i] != chord_history[i-1])
        
        return changes / (len(chord_history) - 1)
    
    def compute_key_stability(self, key_history: List[str]) -> float:
        """Compute key stability over time"""
        if len(key_history) < 2:
            return 1.0
            
        changes = sum(1 for i in range(1, len(key_history)) 
                     if key_history[i] != key_history[i-1])
        
        return 1.0 - (changes / (len(key_history) - 1))
    
    def compute_tonality_score(self, chromagram: np.ndarray) -> float:
        """Compute how tonal vs atonal the music is"""
        if chromagram is None or np.sum(chromagram) == 0:
            return 0.5
            
        major_profile = np.array([1.0, 0.0, 0.5, 0.0, 0.7, 0.5, 0.0, 0.8, 0.0, 0.5, 0.0, 0.3])
        minor_profile = np.array([1.0, 0.0, 0.5, 0.7, 0.0, 0.5, 0.0, 0.8, 0.3, 0.0, 0.5, 0.0])
        
        norm_chroma = chromagram / (np.sum(chromagram) + 1e-10)
        
        correlations = []
        for shift in range(12):
            shifted_major = np.roll(major_profile, shift)
            shifted_minor = np.roll(minor_profile, shift)
            
            major_corr = np.corrcoef(norm_chroma, shifted_major)[0, 1]
            minor_corr = np.corrcoef(norm_chroma, shifted_minor)[0, 1]
            
            correlations.extend([major_corr, minor_corr])
        
        return max(0, max(correlations))
    
    def compute_bass_emphasis(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Compute how much bass (20-250Hz) dominates the spectrum"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        magnitude = fft_data if fft_data.ndim == 1 else np.abs(fft_data)
        
        if len(magnitude) != len(freqs):
            min_len = min(len(magnitude), len(freqs))
            magnitude = magnitude[:min_len]
            freqs = freqs[:min_len]
        
        bass_mask = (freqs >= 20) & (freqs <= 250)
        mid_mask = (freqs > 250) & (freqs <= 2000)
        
        if not np.any(bass_mask) or not np.any(mid_mask):
            return 0.0
        
        bass_energy = np.sum(magnitude[bass_mask] ** 2)
        mid_energy = np.sum(magnitude[mid_mask] ** 2)
        total_energy = np.sum(magnitude ** 2)
        
        if total_energy == 0:
            return 0.0
        
        bass_ratio = bass_energy / total_energy
        bass_to_mid = bass_energy / (mid_energy + 1e-10)
        
        emphasis = bass_ratio * min(bass_to_mid / 2.0, 1.0)
        
        return min(emphasis, 1.0)
    
    def compute_beat_regularity(self, drum_info: Dict) -> float:
        """Compute how regular/consistent the beat pattern is"""
        kick_strength = drum_info.get('kick', {}).get('magnitude', 0)
        snare_strength = drum_info.get('snare', {}).get('magnitude', 0)
        
        if kick_strength > 0.7 and snare_strength > 0.5:
            return 0.9
        elif kick_strength > 0.5 or snare_strength > 0.4:
            return 0.7
        else:
            return 0.3
    
    def compute_vocal_presence(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Detect presence of vocals in typical vocal frequency range"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        magnitude = fft_data if fft_data.ndim == 1 else np.abs(fft_data)
        
        if len(magnitude) != len(freqs):
            min_len = min(len(magnitude), len(freqs))
            magnitude = magnitude[:min_len]
            freqs = freqs[:min_len]
        
        vocal_mask = (freqs >= 200) & (freqs <= 4000)
        core_vocal_mask = (freqs >= 300) & (freqs <= 3000)
        
        if not np.any(vocal_mask):
            return 0.0
        
        vocal_energy = np.sum(magnitude[vocal_mask] ** 2)
        core_vocal_energy = np.sum(magnitude[core_vocal_mask] ** 2) if np.any(core_vocal_mask) else 0
        total_energy = np.sum(magnitude ** 2)
        
        if total_energy == 0:
            return 0.0
        
        vocal_ratio = vocal_energy / total_energy
        core_ratio = core_vocal_energy / vocal_energy if vocal_energy > 0 else 0
        
        presence = vocal_ratio * (0.5 + 0.5 * core_ratio)
        
        return min(presence * 2.0, 1.0)
    
    def get_top_genres(self, probabilities: Dict[str, float], top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top N most likely genres"""
        sorted_genres = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_genres[:top_n]
    
    def get_hip_hop_analysis(self) -> Dict[str, any]:
        """Get detailed hip-hop analysis results"""
        if hasattr(self, '_last_hip_hop_analysis'):
            return self._last_hip_hop_analysis
        return None


# Keep the GenreClassificationPanel class the same, just import the fixed classifier
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
        self.chord_history = deque(maxlen=30)
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
        """Update genre classification with new audio data"""
        if fft_data is not None and len(fft_data) > 0 and audio_chunk is not None:
            # Check for silence
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < 0.001:
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
                
            # Extract features
            features = self.classifier.extract_features(
                fft_data, audio_chunk, frequencies, drum_info, harmonic_info,
                self.chromagram_data,
                list(self.chord_history),
                list(self.key_history)
            )
            
            # Classify genre
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
            
            # Update exposed attributes
            self.genre_probabilities = genre_probabilities.copy()
            self.features = features.copy()
            self.current_genre = self.genre_info['top_genre']
            self.genre_confidence = self.genre_info['confidence']
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """OMEGA-2: Draw genre classification information"""
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((40, 30, 25))
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