"""
Enhanced Genre Classification Panel with Integrated Hip Hop Detection
Demonstrates integration of the hip hop detector into existing genre classification
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import logging

# Import the hip hop detector
from .hip_hop_detector import HipHopClassifier

logger = logging.getLogger(__name__)


class GenreClassificationPanel:
    """Enhanced genre classification with specialized hip hop detection"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Initialize specialized hip hop detector
        self.hip_hop_classifier = HipHopClassifier(sample_rate)
        
        # Genre probabilities with hip hop variants
        self.genre_probabilities = {
            'Rock': 0.0,
            'Pop': 0.0,
            'Jazz': 0.0,
            'Classical': 0.0,
            'Electronic': 0.0,
            'Hip-Hop': 0.0,
            'Hip-Hop (Trap)': 0.0,
            'Hip-Hop (Boom Bap)': 0.0,
            'Hip-Hop (Modern)': 0.0,
            'Metal': 0.0,
            'Country': 0.0,
            'R&B': 0.0,
            'Folk': 0.0
        }
        
        # Current classification
        self.current_genre = 'Unknown'
        self.genre_confidence = 0.0
        self.hip_hop_subgenre = 'unknown'
        
        # Feature storage
        self.features = {}
        
        # Hip hop detection mode
        self.hip_hop_priority_mode = True  # Give priority to hip hop detection
        self.hip_hop_confidence_boost = 0.15  # Boost hip hop confidence when detected
        
        # History for stability
        self.genre_history = deque(maxlen=30)
        self.hip_hop_detection_history = deque(maxlen=30)
        
    def update(self, fft_magnitude: np.ndarray, audio_data: np.ndarray, 
               frequencies: np.ndarray, drum_info: Dict[str, Any], 
               harmonic_info: Dict[str, Any]):
        """Update genre classification with hip hop focus"""
        
        # First, run specialized hip hop detection
        is_hip_hop, hip_hop_confidence, subgenre = self.hip_hop_classifier.classify(
            fft_magnitude, audio_data, frequencies,
            additional_features={
                'tempo': drum_info.get('tempo', 0),
                'has_drums': drum_info.get('drum_presence', False),
                'percussion_strength': drum_info.get('strength', 0)
            }
        )
        
        # Store hip hop detection result
        self.hip_hop_detection_history.append((is_hip_hop, hip_hop_confidence, subgenre))
        
        # Extract general genre features
        self.features = self._extract_genre_features(
            fft_magnitude, audio_data, frequencies, drum_info, harmonic_info
        )
        
        # Update all genre probabilities
        self._update_genre_probabilities()
        
        # Apply hip hop detection results
        if self.hip_hop_priority_mode and is_hip_hop:
            # Override with hip hop detection
            self._apply_hip_hop_detection(hip_hop_confidence, subgenre)
        
        # Determine top genre
        self._determine_current_genre()
        
        # Store in history
        self.genre_history.append(self.current_genre)
    
    def _extract_genre_features(self, fft_magnitude: np.ndarray, audio_data: np.ndarray,
                               frequencies: np.ndarray, drum_info: Dict[str, Any],
                               harmonic_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for general genre classification"""
        
        features = {}
        
        # Spectral features
        total_energy = np.sum(fft_magnitude)
        if total_energy > 0:
            # Spectral centroid
            features['spectral_centroid'] = np.sum(frequencies * fft_magnitude) / total_energy
            
            # Spectral rolloff
            cumsum = np.cumsum(fft_magnitude)
            rolloff_idx = np.where(cumsum >= 0.85 * total_energy)[0]
            features['spectral_rolloff'] = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            
            # Spectral bandwidth
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((frequencies - features['spectral_centroid']) ** 2) * fft_magnitude) / total_energy
            )
        else:
            features['spectral_centroid'] = 0
            features['spectral_rolloff'] = 0
            features['spectral_bandwidth'] = 0
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / 2
        features['zero_crossing_rate'] = zero_crossings / len(audio_data)
        
        # Percussion features from drum_info
        features['percussion_strength'] = drum_info.get('strength', 0)
        features['tempo'] = drum_info.get('tempo', 0)
        features['has_drums'] = drum_info.get('drum_presence', False)
        
        # Harmonic features
        features['harmonic_complexity'] = harmonic_info.get('complexity', 0)
        features['key_strength'] = harmonic_info.get('key_confidence', 0)
        
        # Get hip hop specific features
        hip_hop_features = self.hip_hop_classifier.feature_extractor.feature_history[-1] if \
                          self.hip_hop_classifier.feature_extractor.feature_history else {}
        features.update(hip_hop_features)
        
        return features
    
    def _update_genre_probabilities(self):
        """Update probabilities for all genres based on features"""
        
        # Reset probabilities
        for genre in self.genre_probabilities:
            self.genre_probabilities[genre] = 0.0
        
        # Rock: High spectral centroid, strong percussion, moderate complexity
        if self.features.get('spectral_centroid', 0) > 2000 and \
           self.features.get('percussion_strength', 0) > 0.5:
            self.genre_probabilities['Rock'] = 0.7
        
        # Pop: Balanced features, clear structure
        if 1500 < self.features.get('spectral_centroid', 0) < 3000 and \
           self.features.get('key_strength', 0) > 0.6:
            self.genre_probabilities['Pop'] = 0.6
        
        # Jazz: High harmonic complexity, irregular rhythm
        if self.features.get('harmonic_complexity', 0) > 0.7 and \
           self.features.get('rhythm_regularity', 0) < 0.5:
            self.genre_probabilities['Jazz'] = 0.7
        
        # Classical: Very high harmonic complexity, no drums
        if self.features.get('harmonic_complexity', 0) > 0.8 and \
           not self.features.get('has_drums', True):
            self.genre_probabilities['Classical'] = 0.8
        
        # Electronic: Low spectral centroid, regular rhythm, high bass
        if self.features.get('spectral_centroid', 0) < 1500 and \
           self.features.get('rhythm_regularity', 0) > 0.8 and \
           self.features.get('bass_dominance', 0) > 0.4:
            self.genre_probabilities['Electronic'] = 0.7
        
        # R&B: Smooth vocals, moderate tempo, harmonic richness
        if 70 < self.features.get('tempo', 0) < 100 and \
           self.features.get('harmonic_complexity', 0) > 0.5:
            self.genre_probabilities['R&B'] = 0.5
        
        # Normalize probabilities
        total = sum(self.genre_probabilities.values())
        if total > 0:
            for genre in self.genre_probabilities:
                self.genre_probabilities[genre] /= total
    
    def _apply_hip_hop_detection(self, hip_hop_confidence: float, subgenre: str):
        """Apply specialized hip hop detection results"""
        
        # Base hip hop probability from specialized detector
        base_hip_hop_prob = hip_hop_confidence
        
        # Apply confidence boost if enabled
        if self.hip_hop_confidence_boost > 0:
            base_hip_hop_prob = min(1.0, base_hip_hop_prob + self.hip_hop_confidence_boost)
        
        # Check temporal consistency
        recent_detections = list(self.hip_hop_detection_history)[-10:]
        if len(recent_detections) >= 5:
            # Count positive detections
            positive_count = sum(1 for d in recent_detections if d[0])
            consistency_factor = positive_count / len(recent_detections)
            
            # Apply consistency boost
            base_hip_hop_prob *= (0.7 + 0.3 * consistency_factor)
        
        # Set hip hop probabilities
        if subgenre == 'trap':
            self.genre_probabilities['Hip-Hop (Trap)'] = base_hip_hop_prob * 0.8
            self.genre_probabilities['Hip-Hop'] = base_hip_hop_prob * 0.2
        elif subgenre == 'boom_bap':
            self.genre_probabilities['Hip-Hop (Boom Bap)'] = base_hip_hop_prob * 0.8
            self.genre_probabilities['Hip-Hop'] = base_hip_hop_prob * 0.2
        elif subgenre == 'modern':
            self.genre_probabilities['Hip-Hop (Modern)'] = base_hip_hop_prob * 0.8
            self.genre_probabilities['Hip-Hop'] = base_hip_hop_prob * 0.2
        else:
            # Unknown subgenre or general hip hop
            self.genre_probabilities['Hip-Hop'] = base_hip_hop_prob
        
        # Store subgenre
        self.hip_hop_subgenre = subgenre
        
        # Reduce other genre probabilities proportionally
        hip_hop_total = sum(v for k, v in self.genre_probabilities.items() if 'Hip-Hop' in k)
        if hip_hop_total > 0.5:
            # Hip hop is dominant, reduce others
            reduction_factor = (1 - hip_hop_total) / (1 - hip_hop_total + 0.001)
            for genre in self.genre_probabilities:
                if 'Hip-Hop' not in genre:
                    self.genre_probabilities[genre] *= reduction_factor
    
    def _determine_current_genre(self):
        """Determine the current genre based on probabilities"""
        
        # Find top genre
        if self.genre_probabilities:
            top_genre = max(self.genre_probabilities.items(), key=lambda x: x[1])
            self.current_genre = top_genre[0]
            self.genre_confidence = top_genre[1]
        else:
            self.current_genre = 'Unknown'
            self.genre_confidence = 0.0
        
        # Apply temporal smoothing
        if len(self.genre_history) >= 10:
            # Check recent history
            recent_genres = list(self.genre_history)[-10:]
            genre_counts = {}
            
            for g in recent_genres:
                if g not in genre_counts:
                    genre_counts[g] = 0
                genre_counts[g] += 1
            
            # If current genre isn't stable, use most common recent
            if genre_counts.get(self.current_genre, 0) < 3:
                most_common = max(genre_counts.items(), key=lambda x: x[1])
                if most_common[1] >= 5:  # At least 50% of recent history
                    self.current_genre = most_common[0]
                    # Adjust confidence based on stability
                    self.genre_confidence *= (most_common[1] / 10)
    
    def get_hip_hop_analysis(self) -> Dict[str, Any]:
        """Get detailed hip hop analysis for debugging"""
        
        analysis = {
            'is_hip_hop': False,
            'confidence': 0.0,
            'subgenre': 'unknown',
            'feature_analysis': {},
            'detection_history': []
        }
        
        if self.hip_hop_detection_history:
            # Latest detection
            latest = self.hip_hop_detection_history[-1]
            analysis['is_hip_hop'] = latest[0]
            analysis['confidence'] = latest[1]
            analysis['subgenre'] = latest[2]
            
            # Feature analysis
            analysis['feature_analysis'] = self.hip_hop_classifier.get_feature_analysis()
            
            # Recent history
            analysis['detection_history'] = list(self.hip_hop_detection_history)[-10:]
        
        return analysis
    
    def toggle_hip_hop_priority(self):
        """Toggle hip hop priority mode"""
        self.hip_hop_priority_mode = not self.hip_hop_priority_mode
        logger.info(f"Hip hop priority mode: {self.hip_hop_priority_mode}")
    
    def adjust_hip_hop_sensitivity(self, delta: float):
        """Adjust hip hop detection sensitivity"""
        self.hip_hop_confidence_boost = np.clip(
            self.hip_hop_confidence_boost + delta, 
            -0.3, 0.3
        )
        logger.info(f"Hip hop confidence boost: {self.hip_hop_confidence_boost:.2f}")