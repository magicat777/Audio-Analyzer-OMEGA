"""
Unified Music Analysis Engine for OMEGA-4 Audio Analyzer
Coordinates chromagram and genre classification for enhanced music understanding
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging

# Import the existing modules
from .chromagram import ChromagramPanel
from .genre_classification import GenreClassificationPanel
from .hip_hop_detector import HipHopDetector

logger = logging.getLogger(__name__)


class MusicAnalysisConfig:
    """Configuration for integrated music analysis"""
    
    def __init__(self):
        # Integration settings
        self.enable_harmonic_genre_features = True
        self.enable_genre_informed_chromagram = True
        self.enable_cross_validation = True
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.3
        self.high_confidence_threshold = 0.7
        self.consistency_bonus = 0.15  # Boost when modules agree
        
        # Temporal smoothing
        self.temporal_smoothing = 30  # frames
        self.history_weight = 0.7  # Weight for historical data
        
        # Feature weights for fusion
        self.harmonic_weight = 0.4
        self.timbral_weight = 0.3
        self.rhythmic_weight = 0.3
        
        # Genre-specific harmonic patterns
        self.genre_harmonic_patterns = {
            'jazz': {
                'extended_chords': True,
                'complex_progressions': True,
                'frequent_modulation': True,
                'chromatic_movement': True
            },
            'classical': {
                'diatonic_harmony': True,
                'cadence_patterns': True,
                'key_stability': True,
                'voice_leading': True
            },
            'pop': {
                'simple_triads': True,
                'common_progressions': True,  # I-V-vi-IV
                'repetitive_structure': True,
                'clear_tonality': True
            },
            'electronic': {
                'minimal_harmony': True,
                'static_harmony': True,
                'emphasis_on_timbre': True,
                'rhythmic_focus': True
            },
            'rock': {
                'power_chords': True,
                'blues_influence': True,
                'pentatonic_scales': True,
                'guitar_centric': True
            }
        }


class MusicAnalysisEngine:
    """Unified engine for comprehensive music analysis"""
    
    def __init__(self, sample_rate: int = 48000, config: Optional[MusicAnalysisConfig] = None):
        self.sample_rate = sample_rate
        self.config = config or MusicAnalysisConfig()
        
        # Initialize component analyzers
        self.chromagram = ChromagramPanel(sample_rate)
        self.genre_classifier = GenreClassificationPanel(sample_rate)
        self.hip_hop_detector = HipHopDetector(sample_rate)  # Direct access to hip-hop detector
        
        # Analysis history for temporal consistency
        self.analysis_history = deque(maxlen=self.config.temporal_smoothing)
        
        # Cross-module feature cache
        self.feature_cache = {}
        
        # Combined results with proper defaults
        self.current_analysis = {
            'harmony': {
                'key': 'Unknown',
                'key_confidence': 0.0,
                'chords': [],
                'chromagram': np.zeros(12),
                'chord_progression': [],
                'harmonic_complexity': 0.0
            },
            'genre': {
                'top_genre': 'Unknown',
                'confidence': 0.0,
                'all_genres': {},
                'features': {}
            },
            'cross_analysis': {
                'overall_confidence': 0.0,
                'harmonic_genre_consistency': 0.0,
                'genre_typical_progression': '',
                'suggested_genres': []
            },
            'confidence': 0.0
        }
        
    def analyze_music(self, fft_magnitude: np.ndarray, audio_data: np.ndarray, 
                     frequencies: np.ndarray, drum_info: Dict[str, Any], 
                     harmonic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integrated music analysis
        
        Args:
            fft_magnitude: FFT magnitude spectrum
            audio_data: Raw audio samples
            frequencies: Frequency bins
            drum_info: Drum detection results
            harmonic_info: Harmonic analysis results
            
        Returns:
            Comprehensive music analysis results
        """
        # Check for silence first
        if audio_data is not None and len(audio_data) > 0:
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.001:  # Silence threshold
                # Return default/unknown state during silence
                silence_results = {
                    'harmony': {
                        'key': 'Unknown',
                        'key_confidence': 0.0,
                        'chords': [],
                        'chromagram': np.zeros(12),
                        'harmonic_complexity': 0.0,
                        'circle_position': 0
                    },
                    'genre': {
                        'top_genre': 'Unknown',
                        'confidence': 0.0,
                        'all_genres': {'Unknown': 1.0},
                        'features': {}
                    },
                    'hip_hop': {
                        'confidence': 0.0,
                        'features': {},
                        'subgenre': 'unknown',
                        'is_hip_hop': False
                    },
                    'cross_analysis': {
                        'overall_confidence': 0.0,
                        'harmonic_genre_consistency': 0.0,
                        'suggested_key_for_genre': 'Unknown',
                        'genre_typical_progression': '',
                        'cross_features': {}
                    },
                    'temporal': {
                        'stability': 0.0,
                        'trend': 'stable'
                    }
                }
                self.current_analysis = silence_results
                return silence_results
        
        # Update individual analyzers
        self.chromagram.update(fft_magnitude, audio_data, frequencies)
        self.genre_classifier.update(
            fft_magnitude, audio_data, frequencies, 
            drum_info, harmonic_info
        )
        
        # Perform hip-hop analysis
        hip_hop_analysis = self.hip_hop_detector.analyze(
            audio_data, fft_magnitude, frequencies, drum_info
        )
        
        # Extract features from both modules
        harmonic_features = self._extract_harmonic_features()
        genre_features = self._extract_genre_features()
        
        # Perform cross-module analysis
        cross_features = self._compute_cross_features(harmonic_features, genre_features)
        
        # Feature fusion
        fused_confidence = self._fuse_confidences(harmonic_features, genre_features, cross_features)
        
        # Genre-harmonic consistency check
        consistency = self._check_genre_harmonic_consistency(
            harmonic_features, genre_features
        )
        
        # Build comprehensive results
        results = {
            'harmony': {
                'key': self.chromagram.detected_key,
                'key_confidence': self.chromagram.key_confidence,
                'chords': self.chromagram.chord_sequence[-5:] if hasattr(self.chromagram, 'chord_sequence') else [],
                'chromagram': self.chromagram.chromagram.copy() if self.chromagram.chromagram is not None else np.zeros(12),
                'harmonic_complexity': self._calculate_harmonic_complexity(harmonic_features),
                'circle_position': self.chromagram.circle_position
            },
            'genre': {
                'top_genre': self.genre_classifier.current_genre,
                'confidence': self.genre_classifier.genre_confidence,
                'all_genres': self.genre_classifier.genre_probabilities.copy(),
                'features': genre_features
            },
            'hip_hop': hip_hop_analysis,  # Add dedicated hip-hop analysis
            'cross_analysis': {
                'overall_confidence': fused_confidence,
                'harmonic_genre_consistency': consistency,
                'suggested_key_for_genre': self._suggest_key_for_genre(
                    self.genre_classifier.current_genre
                ),
                'genre_typical_progression': self._get_genre_progression(
                    self.genre_classifier.current_genre
                ),
                'cross_features': cross_features
            },
            'temporal': {
                'stability': self._calculate_temporal_stability(),
                'trend': self._analyze_temporal_trend()
            }
        }
        
        # Update history
        self.analysis_history.append(results)
        self.current_analysis = results
        
        return results
    
    def _extract_harmonic_features(self) -> Dict[str, Any]:
        """Extract features from chromagram analysis"""
        features = {
            'key_strength': self.chromagram.key_confidence,
            'chromatic_energy': np.sum(self.chromagram.chromagram) if self.chromagram.chromagram is not None else 0,
            'pitch_class_entropy': self._calculate_pitch_class_entropy(),
            'strongest_pitch_classes': self._get_strongest_pitch_classes(),
            'harmonic_change_rate': self._calculate_harmonic_change_rate()
        }
        
        # Add chord complexity if available
        if hasattr(self.chromagram, 'current_chord') and self.chromagram.current_chord:
            features['chord_complexity'] = len(self.chromagram.current_chord.split('/'))
        else:
            features['chord_complexity'] = 0
            
        return features
    
    def _extract_genre_features(self) -> Dict[str, Any]:
        """Extract features from genre classification"""
        return {
            'genre_confidence': self.genre_classifier.genre_confidence,
            'genre_entropy': self._calculate_genre_entropy(),
            'spectral_centroid': self.genre_classifier.features.get('spectral_centroid', 0),
            'percussion_strength': self.genre_classifier.features.get('percussion_strength', 0),
            'zero_crossing_rate': self.genre_classifier.features.get('zero_crossing_rate', 0)
        }
    
    def _compute_cross_features(self, harmonic: Dict[str, Any], 
                               genre: Dict[str, Any]) -> Dict[str, Any]:
        """Compute features that combine harmonic and genre information"""
        features = {
            'harmonic_to_percussion_ratio': (
                harmonic['chromatic_energy'] / 
                (genre['percussion_strength'] + 1e-6)
            ),
            'complexity_genre_match': self._match_complexity_to_genre(
                harmonic['chord_complexity'], 
                self.genre_classifier.current_genre
            ),
            'key_genre_affinity': self._calculate_key_genre_affinity(
                self.chromagram.detected_key,
                self.genre_classifier.current_genre
            )
        }
        
        return features
    
    def _fuse_confidences(self, harmonic: Dict[str, Any], 
                         genre: Dict[str, Any], 
                         cross: Dict[str, Any]) -> float:
        """Fuse confidence scores from multiple sources"""
        # Base confidences
        key_conf = harmonic['key_strength']
        genre_conf = genre['genre_confidence']
        
        # Weighted combination
        base_confidence = (
            self.config.harmonic_weight * key_conf +
            self.config.timbral_weight * genre_conf
        )
        
        # Apply cross-validation bonus
        if self.config.enable_cross_validation:
            if cross['complexity_genre_match'] > 0.7:
                base_confidence += self.config.consistency_bonus
                
        # Temporal consistency bonus
        if len(self.analysis_history) > 10:
            stability = self._calculate_temporal_stability()
            base_confidence += stability * 0.1
            
        return min(1.0, base_confidence)
    
    def _check_genre_harmonic_consistency(self, harmonic: Dict[str, Any], 
                                         genre: Dict[str, Any]) -> float:
        """Check if harmonic features match expected genre patterns"""
        current_genre = self.genre_classifier.current_genre
        
        if current_genre not in self.config.genre_harmonic_patterns:
            return 0.5  # Neutral for unknown genres
            
        expected_patterns = self.config.genre_harmonic_patterns[current_genre]
        consistency_score = 0.0
        checks = 0
        
        # Check chord complexity
        if expected_patterns.get('extended_chords'):
            if harmonic['chord_complexity'] > 2:
                consistency_score += 1.0
            checks += 1
        elif expected_patterns.get('simple_triads'):
            if harmonic['chord_complexity'] <= 2:
                consistency_score += 1.0
            checks += 1
            
        # Check harmonic change rate
        if expected_patterns.get('frequent_modulation'):
            if harmonic['harmonic_change_rate'] > 0.5:
                consistency_score += 1.0
            checks += 1
        elif expected_patterns.get('key_stability'):
            if harmonic['harmonic_change_rate'] < 0.2:
                consistency_score += 1.0
            checks += 1
            
        # Check pitch class entropy
        if expected_patterns.get('chromatic_movement'):
            if harmonic['pitch_class_entropy'] > 2.0:
                consistency_score += 1.0
            checks += 1
        elif expected_patterns.get('diatonic_harmony'):
            if harmonic['pitch_class_entropy'] < 1.5:
                consistency_score += 1.0
            checks += 1
            
        return consistency_score / max(1, checks)
    
    def _calculate_pitch_class_entropy(self) -> float:
        """Calculate entropy of pitch class distribution"""
        if self.chromagram.chromagram is None:
            return 0.0
            
        # Normalize chromagram
        total = np.sum(self.chromagram.chromagram)
        if total == 0:
            return 0.0
            
        probs = self.chromagram.chromagram / total
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def _get_strongest_pitch_classes(self, n: int = 3) -> List[int]:
        """Get indices of strongest pitch classes"""
        if self.chromagram.chromagram is None:
            return []
            
        return np.argsort(self.chromagram.chromagram)[-n:].tolist()
    
    def _calculate_harmonic_change_rate(self) -> float:
        """Calculate rate of harmonic change over time"""
        if len(self.analysis_history) < 2:
            return 0.0
            
        changes = 0
        for i in range(1, min(10, len(self.analysis_history))):
            prev_key = self.analysis_history[-i-1]['harmony']['key']
            curr_key = self.analysis_history[-i]['harmony']['key']
            if prev_key != curr_key:
                changes += 1
                
        return changes / min(10, len(self.analysis_history) - 1)
    
    def _calculate_genre_entropy(self) -> float:
        """Calculate entropy of genre probability distribution"""
        probs = list(self.genre_classifier.genre_probabilities.values())
        if not probs or sum(probs) == 0:
            return 0.0
            
        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        return entropy
    
    def _match_complexity_to_genre(self, complexity: int, genre: str) -> float:
        """Match chord complexity to expected genre patterns"""
        complexity_expectations = {
            'jazz': (3, 5),      # Extended chords
            'classical': (2, 4), # Triads to 7ths
            'pop': (1, 3),       # Simple triads
            'rock': (1, 3),      # Power chords to triads
            'electronic': (0, 2) # Minimal harmony
        }
        
        if genre not in complexity_expectations:
            return 0.5
            
        min_expected, max_expected = complexity_expectations[genre]
        
        if min_expected <= complexity <= max_expected:
            return 1.0
        elif complexity < min_expected:
            return max(0, 1 - (min_expected - complexity) * 0.3)
        else:
            return max(0, 1 - (complexity - max_expected) * 0.3)
    
    def _calculate_key_genre_affinity(self, key: str, genre: str) -> float:
        """Calculate affinity between detected key and genre"""
        # Simplified affinity based on common keys per genre
        genre_key_preferences = {
            'jazz': {'F', 'Bb', 'Eb', 'Ab'},  # Flat keys common in jazz
            'classical': {'C', 'G', 'D', 'A', 'F'},  # Common classical keys
            'pop': {'C', 'G', 'D', 'A', 'E'},  # Guitar-friendly keys
            'rock': {'E', 'A', 'D', 'G'},  # Guitar power chord keys
            'electronic': {'Am', 'Cm', 'Dm'}  # Minor keys common in EDM
        }
        
        if genre not in genre_key_preferences or not key:
            return 0.5
            
        # Extract key without mode
        key_base = key.replace('m', '').strip()
        
        if key_base in genre_key_preferences[genre]:
            return 1.0
        else:
            return 0.3
    
    def _suggest_key_for_genre(self, genre: str) -> str:
        """Suggest appropriate key for detected genre"""
        suggestions = {
            'jazz': 'F or Bb (jazz-friendly keys)',
            'classical': 'C or G (classical home keys)',
            'pop': 'C, G, or D (radio-friendly keys)',
            'rock': 'E or A (guitar-friendly keys)',
            'electronic': 'Am or Cm (electronic atmosphere)'
        }
        
        return suggestions.get(genre, 'C (universal)')
    
    def _get_genre_progression(self, genre: str) -> str:
        """Get typical chord progression for genre"""
        progressions = {
            'jazz': 'ii-V-I or I-vi-ii-V',
            'classical': 'I-IV-V-I or I-V-vi-IV',
            'pop': 'I-V-vi-IV or vi-IV-I-V',
            'rock': 'I-IV-V or I-bVII-IV-I',
            'electronic': 'i-i-i-i or i-VI-VII-i'
        }
        
        return progressions.get(genre, 'I-IV-V')
    
    def _calculate_harmonic_complexity(self, features: Dict[str, Any]) -> float:
        """Calculate overall harmonic complexity score"""
        complexity = 0.0
        
        # Factors contributing to complexity
        complexity += features['pitch_class_entropy'] / 3.5  # Normalized entropy
        complexity += features['chord_complexity'] / 5.0     # Normalized chord complexity
        complexity += features['harmonic_change_rate']       # Already normalized
        
        return min(1.0, complexity / 3.0)
    
    def _calculate_temporal_stability(self) -> float:
        """Calculate stability of analysis over time"""
        if len(self.analysis_history) < 5:
            return 0.0
            
        # Check genre stability
        recent_genres = [h['genre']['top_genre'] for h in list(self.analysis_history)[-10:]]
        genre_changes = sum(1 for i in range(1, len(recent_genres)) 
                           if recent_genres[i] != recent_genres[i-1])
        
        # Check key stability
        recent_keys = [h['harmony']['key'] for h in list(self.analysis_history)[-10:]]
        key_changes = sum(1 for i in range(1, len(recent_keys)) 
                         if recent_keys[i] != recent_keys[i-1])
        
        # Calculate stability (inverse of change rate)
        stability = 1.0 - (genre_changes + key_changes) / (2 * (len(recent_genres) - 1))
        
        return max(0, stability)
    
    def _analyze_temporal_trend(self) -> Dict[str, str]:
        """Analyze trends in music over time"""
        if len(self.analysis_history) < 10:
            return {'genre': 'stable', 'harmony': 'stable'}
            
        # Analyze recent history
        recent = list(self.analysis_history)[-20:]
        
        # Genre trend
        genre_counts = {}
        for h in recent:
            g = h['genre']['top_genre']
            genre_counts[g] = genre_counts.get(g, 0) + 1
            
        dominant_genre = max(genre_counts, key=genre_counts.get)
        genre_trend = 'changing' if genre_counts[dominant_genre] < len(recent) * 0.6 else 'stable'
        
        # Harmonic trend
        complexity_values = [h['cross_analysis']['cross_features'].get('harmonic_to_percussion_ratio', 0) 
                           for h in recent]
        complexity_trend = 'increasing' if complexity_values[-1] > complexity_values[0] * 1.2 else \
                          'decreasing' if complexity_values[-1] < complexity_values[0] * 0.8 else 'stable'
        
        return {
            'genre': genre_trend,
            'harmony': complexity_trend
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get current analysis results"""
        return self.current_analysis.copy()
    
    def reset(self):
        """Reset analysis history"""
        self.analysis_history.clear()
        self.feature_cache.clear()
        self.chromagram.chromagram = np.zeros(12)
        self.genre_classifier.genre_probabilities = {
            genre: 0.0 for genre in self.genre_classifier.genre_probabilities
        }