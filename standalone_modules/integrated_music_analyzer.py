"""
Integrated Music Analysis Engine - Combines Chromagram and Genre Classification
Phase 3: Unified music analysis with cross-module feature sharing
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class MusicAnalysisConfig:
    """Configuration for integrated music analysis"""
    sample_rate: int = 48000
    
    # Cross-module weighting
    harmonic_weight: float = 0.6  # Weight of harmonic features in genre classification
    timbral_weight: float = 0.4   # Weight of timbral features in genre classification
    
    # Genre-specific analysis
    enable_genre_informed_chromagram: bool = True
    enable_harmonic_genre_features: bool = True
    
    # Integration parameters
    confidence_threshold: float = 0.5
    cross_validation_weight: float = 0.2
    temporal_smoothing: int = 30  # frames

class IntegratedMusicAnalyzer:
    """Unified music analysis combining chromagram and genre classification"""
    
    def __init__(self, config: Optional[MusicAnalysisConfig] = None):
        self.config = config or MusicAnalysisConfig()
        
        # Import and initialize both analyzers
        from chromagram import ChromagramAnalyzer
        from genre_classification import GenreClassifier
        
        self.chromagram_analyzer = ChromagramAnalyzer(self.config.sample_rate)
        self.genre_classifier = GenreClassifier(self.config.sample_rate)
        
        # Shared feature cache
        self.feature_cache = {}
        
        # Cross-module analysis history
        self.analysis_history = deque(maxlen=self.config.temporal_smoothing)
        
        # Genre-specific harmonic patterns
        self.genre_harmonic_profiles = self._initialize_harmonic_profiles()
        
    def _initialize_harmonic_profiles(self) -> Dict[str, Dict]:
        """Initialize genre-specific harmonic analysis profiles"""
        return {
            'Jazz': {
                'preferred_keys': ['C', 'F', 'Bb', 'Eb', 'G', 'D'],
                'chord_complexity': 0.8,  # Expects complex chords
                'modulation_frequency': 0.6,  # Frequent key changes
                'extended_chords': True,
                'harmonic_rhythm': 0.7  # Fast chord changes
            },
            'Classical': {
                'preferred_keys': ['C', 'G', 'D', 'A', 'F', 'Bb'],
                'chord_complexity': 0.9,
                'modulation_frequency': 0.4,  # Structured modulations
                'extended_chords': False,
                'harmonic_rhythm': 0.3  # Slower, more deliberate
            },
            'Pop': {
                'preferred_keys': ['C', 'G', 'D', 'A', 'F'],
                'chord_complexity': 0.3,  # Simple triads
                'modulation_frequency': 0.1,  # Rare key changes
                'extended_chords': False,
                'harmonic_rhythm': 0.4
            },
            'Rock': {
                'preferred_keys': ['E', 'A', 'D', 'G', 'C'],
                'chord_complexity': 0.4,
                'modulation_frequency': 0.2,
                'extended_chords': False,
                'harmonic_rhythm': 0.5
            },
            'Electronic': {
                'preferred_keys': ['C', 'G', 'Am', 'Em'],
                'chord_complexity': 0.2,  # Often minimal harmony
                'modulation_frequency': 0.3,
                'extended_chords': False,
                'harmonic_rhythm': 0.6
            }
        }
    
    def extract_integrated_features(self, fft_data: np.ndarray, audio_chunk: np.ndarray, 
                                  freqs: np.ndarray, drum_info: Dict, 
                                  harmonic_info: Dict) -> Dict[str, Any]:
        """Extract comprehensive features from both harmonic and timbral analysis"""
        
        # Get chromagram features
        chromagram = self.chromagram_analyzer.compute_chromagram(fft_data, freqs)
        key, key_confidence = self.chromagram_analyzer.detect_key(chromagram)
        key_stability = self.chromagram_analyzer.get_key_stability()
        
        # Get timbral features
        timbral_features = self.genre_classifier.extract_features(
            fft_data, audio_chunk, drum_info, harmonic_info
        )
        
        # Compute cross-module features
        integrated_features = {
            # Harmonic features
            'chromagram': chromagram,
            'detected_key': key,
            'key_confidence': key_confidence,
            'key_stability': key_stability,
            'harmonic_complexity': self._compute_harmonic_complexity(chromagram),
            'chord_richness': self._compute_chord_richness(chromagram),
            
            # Timbral features (from genre classifier)
            **timbral_features,
            
            # Cross-module features
            'harmonic_to_percussive_ratio': self._compute_harmonic_percussive_ratio(
                chromagram, timbral_features.get('percussion_strength', 0)
            ),
            'tonal_clarity': self._compute_tonal_clarity(chromagram, timbral_features),
            'harmonic_rhythm': self._estimate_harmonic_rhythm(),
            'genre_harmonic_consistency': 0.0  # Will be computed after genre detection
        }
        
        # Cache features for cross-module validation
        self.feature_cache = integrated_features
        
        return integrated_features
    
    def _compute_harmonic_complexity(self, chromagram: np.ndarray) -> float:
        """Calculate complexity of harmonic content"""
        # More evenly distributed chroma = more complex harmony
        if np.sum(chromagram) == 0:
            return 0.0
        
        # Use entropy as complexity measure
        normalized_chroma = chromagram / np.sum(chromagram)
        # Avoid log(0)
        normalized_chroma = normalized_chroma + 1e-10
        entropy = -np.sum(normalized_chroma * np.log2(normalized_chroma))
        
        # Normalize to 0-1 range (max entropy for 12 bins is log2(12))
        max_entropy = np.log2(12)
        return entropy / max_entropy
    
    def _compute_chord_richness(self, chromagram: np.ndarray) -> float:
        """Estimate chord richness (triads vs extended chords)"""
        # Count significant chroma bins (above threshold)
        threshold = np.max(chromagram) * 0.3 if np.max(chromagram) > 0 else 0
        active_bins = np.sum(chromagram > threshold)
        
        # Normalize: 3 bins = triad (0.25), 4+ bins = extended (up to 1.0)
        return min(1.0, max(0.0, (active_bins - 2) / 6))
    
    def _compute_harmonic_percussive_ratio(self, chromagram: np.ndarray, 
                                         percussion_strength: float) -> float:
        """Ratio of harmonic to percussive content"""
        harmonic_strength = np.sum(chromagram)
        if harmonic_strength + percussion_strength == 0:
            return 0.5  # Neutral
        
        return harmonic_strength / (harmonic_strength + percussion_strength)
    
    def _compute_tonal_clarity(self, chromagram: np.ndarray, 
                             timbral_features: Dict) -> float:
        """Measure how clearly tonal the music is"""
        # Combine chroma concentration with spectral clarity
        chroma_concentration = np.max(chromagram) if len(chromagram) > 0 else 0
        
        # Higher spectral centroid with concentrated chroma = clear tonality
        centroid = timbral_features.get('spectral_centroid', 0)
        if centroid > 0:
            # Normalize centroid to 0-1 (assuming 4000Hz as high)
            centroid_norm = min(1.0, centroid / 4000)
            return (chroma_concentration + centroid_norm) / 2
        
        return chroma_concentration
    
    def _estimate_harmonic_rhythm(self) -> float:
        """Estimate rate of harmonic change"""
        # Simplified implementation - would need chord change detection
        # For now, return based on key stability
        stability = self.chromagram_analyzer.get_key_stability()
        return 1.0 - stability  # Low stability = high harmonic rhythm
    
    def classify_genre_with_harmonics(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced genre classification using both timbral and harmonic features"""
        
        # Get base genre classification from timbral features
        base_probabilities = self.genre_classifier.classify(features)
        
        if not self.config.enable_harmonic_genre_features:
            return base_probabilities
        
        # Enhance with harmonic features
        enhanced_probabilities = {}
        
        for genre, base_prob in base_probabilities.items():
            harmonic_boost = self._calculate_harmonic_genre_match(genre, features)
            
            # Weighted combination
            enhanced_prob = (
                base_prob * self.config.timbral_weight + 
                harmonic_boost * self.config.harmonic_weight
            )
            enhanced_probabilities[genre] = enhanced_prob
        
        # Renormalize
        total = sum(enhanced_probabilities.values())
        if total > 0:
            for genre in enhanced_probabilities:
                enhanced_probabilities[genre] /= total
        
        return enhanced_probabilities
    
    def _calculate_harmonic_genre_match(self, genre: str, features: Dict[str, Any]) -> float:
        """Calculate how well harmonic features match expected genre patterns"""
        if genre not in self.genre_harmonic_profiles:
            return 0.5  # Neutral if unknown genre
        
        profile = self.genre_harmonic_profiles[genre]
        score = 0.0
        
        # Chord complexity matching (more lenient)
        expected_complexity = profile['chord_complexity']
        actual_complexity = features.get('harmonic_complexity', 0)
        complexity_diff = abs(expected_complexity - actual_complexity)
        complexity_match = 1.0 - (complexity_diff * 0.7)  # Less penalty for mismatch
        score += complexity_match * 0.3
        
        # Key stability matching (more lenient)
        expected_modulation = profile['modulation_frequency']
        actual_stability = features.get('key_stability', 0)
        # Convert stability to modulation frequency (inverse relationship)
        actual_modulation = 1.0 - actual_stability
        modulation_diff = abs(expected_modulation - actual_modulation)
        modulation_match = 1.0 - (modulation_diff * 0.6)  # Less penalty
        score += modulation_match * 0.3
        
        # Harmonic rhythm matching
        expected_rhythm = profile['harmonic_rhythm']
        actual_rhythm = features.get('harmonic_rhythm', 0)
        rhythm_match = 1.0 - abs(expected_rhythm - actual_rhythm)
        score += rhythm_match * 0.2
        
        # Chord richness for genres that use extended chords
        if profile['extended_chords']:
            chord_richness = features.get('chord_richness', 0)
            score += chord_richness * 0.2
        else:
            chord_richness = features.get('chord_richness', 0)
            score += (1.0 - chord_richness) * 0.2  # Prefer simpler chords
        
        return max(0.0, min(1.0, score))
    
    def analyze_music(self, fft_data: np.ndarray, audio_chunk: np.ndarray,
                     freqs: np.ndarray, drum_info: Dict, 
                     harmonic_info: Dict) -> Dict[str, Any]:
        """Main analysis method combining both modules"""
        
        # Extract integrated features
        features = self.extract_integrated_features(
            fft_data, audio_chunk, freqs, drum_info, harmonic_info
        )
        
        # Enhanced genre classification
        genre_probabilities = self.classify_genre_with_harmonics(features)
        top_genre = max(genre_probabilities, key=genre_probabilities.get)
        genre_confidence = genre_probabilities[top_genre]
        
        # Update genre-harmonic consistency
        features['genre_harmonic_consistency'] = self._calculate_genre_harmonic_consistency(
            top_genre, features
        )
        
        # Get chromagram analysis results
        chromagram_results = {
            'chromagram': features['chromagram'],
            'key': features['detected_key'],
            'key_confidence': features['key_confidence'],
            'key_stability': features['key_stability']
        }
        
        # Combine results
        integrated_results = {
            'genre': {
                'top_genre': top_genre,
                'confidence': genre_confidence,
                'probabilities': genre_probabilities,
                'top_3': sorted(genre_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
            },
            'harmony': chromagram_results,
            'features': features,
            'cross_analysis': {
                'harmonic_genre_consistency': features['genre_harmonic_consistency'],
                'tonal_clarity': features['tonal_clarity'],
                'harmonic_complexity': features['harmonic_complexity'],
                'overall_confidence': (genre_confidence * 0.7 + features.get('key_confidence', 0.5) * 0.3)
            }
        }
        
        # Add to history for temporal analysis
        self.analysis_history.append(integrated_results)
        
        return integrated_results
    
    def _calculate_genre_harmonic_consistency(self, genre: str, features: Dict[str, Any]) -> float:
        """Calculate how well the detected genre matches harmonic characteristics"""
        return self._calculate_harmonic_genre_match(genre, features)
    
    def get_temporal_analysis(self) -> Dict[str, Any]:
        """Get analysis of how music characteristics change over time"""
        if len(self.analysis_history) < 10:
            return {'insufficient_data': True}
        
        # Analyze genre stability
        recent_genres = [analysis['genre']['top_genre'] for analysis in self.analysis_history[-20:]]
        genre_changes = len(set(recent_genres))
        
        # Analyze key stability
        recent_keys = [analysis['harmony']['key'] for analysis in self.analysis_history[-20:]]
        key_changes = len(set(recent_keys))
        
        # Analyze confidence trends
        recent_confidences = [analysis['cross_analysis']['overall_confidence'] 
                            for analysis in self.analysis_history[-10:]]
        confidence_trend = np.mean(recent_confidences) if recent_confidences else 0
        
        return {
            'genre_stability': 1.0 - (genre_changes / len(recent_genres)),
            'key_stability': 1.0 - (key_changes / len(recent_keys)),
            'confidence_trend': confidence_trend,
            'analysis_consistency': np.mean([
                analysis['cross_analysis']['harmonic_genre_consistency'] 
                for analysis in self.analysis_history[-10:]
            ]) if len(self.analysis_history) >= 10 else 0
        }