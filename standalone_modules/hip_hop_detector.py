"""
Hip Hop Detection Module for OMEGA-4 Audio Analyzer
Specialized detection for hip hop music with >70% accuracy target
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class HipHopFeatureExtractor:
    """Extract hip hop-specific audio features"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Hip hop characteristic frequency ranges
        self.sub_bass_range = (20, 60)      # Sub-bass (808s)
        self.bass_range = (60, 250)         # Bass
        self.kick_range = (40, 80)          # Kick drum fundamental
        self.snare_range = (200, 400)       # Snare body
        self.hihat_range = (6000, 12000)    # Hi-hat presence
        self.vocal_range = (250, 4000)      # Vocal fundamentals
        
        # Tempo ranges for different hip hop subgenres
        self.tempo_ranges = {
            'classic': (85, 95),     # Classic hip hop
            'modern': (70, 85),      # Modern hip hop
            'trap': (130, 170),      # Trap (or half-time 65-85)
            'drill': (140, 150),     # Drill
            'boom_bap': (90, 100)    # Boom bap
        }
        
        # Pattern detection buffers
        self.kick_buffer = deque(maxlen=32)  # Store 32 kick events
        self.snare_buffer = deque(maxlen=32)
        self.hihat_buffer = deque(maxlen=64)  # More for rapid hi-hats
        
        # Feature history for temporal consistency
        self.feature_history = deque(maxlen=30)  # ~0.5 seconds at 60 FPS
        
    def extract_features(self, fft_magnitude: np.ndarray, audio_data: np.ndarray, 
                        frequencies: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive hip hop features"""
        
        features = {}
        
        # 1. Spectral Energy Distribution
        features.update(self._extract_spectral_features(fft_magnitude, frequencies))
        
        # 2. Rhythmic Features
        features.update(self._extract_rhythmic_features(audio_data))
        
        # 3. Transient Detection
        features.update(self._extract_transient_features(audio_data))
        
        # 4. Bass Pattern Analysis
        features.update(self._extract_bass_patterns(fft_magnitude, frequencies))
        
        # 5. Hi-hat Density (important for trap detection)
        features['hihat_density'] = self._calculate_hihat_density(fft_magnitude, frequencies)
        
        # Store in history
        self.feature_history.append(features)
        
        return features
    
    def _extract_spectral_features(self, fft_magnitude: np.ndarray, 
                                  frequencies: np.ndarray) -> Dict[str, float]:
        """Extract spectral energy distribution features"""
        
        # Total energy for normalization
        total_energy = np.sum(fft_magnitude)
        if total_energy == 0:
            return {
                'sub_bass_ratio': 0.0,
                'bass_ratio': 0.0,
                'kick_presence': 0.0,
                'bass_dominance': 0.0,
                'spectral_tilt': 0.0
            }
        
        # Sub-bass energy (20-60Hz) - crucial for hip hop
        sub_bass_idx = np.where((frequencies >= self.sub_bass_range[0]) & 
                               (frequencies <= self.sub_bass_range[1]))[0]
        sub_bass_energy = np.sum(fft_magnitude[sub_bass_idx]) if len(sub_bass_idx) > 0 else 0
        
        # Bass energy (60-250Hz)
        bass_idx = np.where((frequencies >= self.bass_range[0]) & 
                           (frequencies <= self.bass_range[1]))[0]
        bass_energy = np.sum(fft_magnitude[bass_idx]) if len(bass_idx) > 0 else 0
        
        # Kick drum presence (40-80Hz peak detection)
        kick_idx = np.where((frequencies >= self.kick_range[0]) & 
                           (frequencies <= self.kick_range[1]))[0]
        kick_presence = np.max(fft_magnitude[kick_idx]) if len(kick_idx) > 0 else 0
        
        # Combined low frequency energy
        low_freq_energy = sub_bass_energy + bass_energy
        
        # High frequency energy for balance check
        high_idx = np.where(frequencies >= 4000)[0]
        high_energy = np.sum(fft_magnitude[high_idx]) if len(high_idx) > 0 else 0
        
        # Spectral tilt (bass-heavy music has negative tilt)
        spectral_tilt = (high_energy - low_freq_energy) / total_energy
        
        return {
            'sub_bass_ratio': sub_bass_energy / total_energy,
            'bass_ratio': bass_energy / total_energy,
            'kick_presence': kick_presence / (np.mean(fft_magnitude) + 1e-6),
            'bass_dominance': low_freq_energy / total_energy,
            'spectral_tilt': spectral_tilt
        }
    
    def _extract_rhythmic_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract rhythm-based features"""
        
        # Onset detection for rhythm analysis
        onsets = self._detect_onsets(audio_data)
        
        if len(onsets) < 2:
            return {
                'onset_rate': 0.0,
                'rhythm_regularity': 0.0,
                'beat_strength': 0.0
            }
        
        # Onset rate (events per second)
        duration = len(audio_data) / self.sample_rate
        onset_rate = len(onsets) / duration
        
        # Rhythm regularity (how consistent are the intervals)
        intervals = np.diff(onsets)
        if len(intervals) > 0:
            rhythm_regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-6))
            rhythm_regularity = np.clip(rhythm_regularity, 0, 1)
        else:
            rhythm_regularity = 0.0
        
        # Beat strength (average onset strength)
        beat_strength = np.mean([audio_data[int(onset)] for onset in onsets 
                               if onset < len(audio_data)])
        
        return {
            'onset_rate': onset_rate,
            'rhythm_regularity': rhythm_regularity,
            'beat_strength': abs(beat_strength)
        }
    
    def _detect_onsets(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple onset detection using spectral flux"""
        
        # Frame-based processing
        frame_size = 2048
        hop_size = 512
        
        onsets = []
        prev_spectrum = None
        
        for i in range(0, len(audio_data) - frame_size, hop_size):
            frame = audio_data[i:i + frame_size]
            
            # Apply window
            windowed = frame * np.hanning(frame_size)
            
            # FFT
            spectrum = np.abs(np.fft.rfft(windowed))
            
            if prev_spectrum is not None:
                # Spectral flux
                flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                
                # Simple threshold
                if flux > np.mean(spectrum) * 2:
                    onsets.append(i + frame_size // 2)
            
            prev_spectrum = spectrum
        
        return np.array(onsets)
    
    def _extract_transient_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract transient characteristics"""
        
        # High-pass filter to isolate transients
        nyquist = self.sample_rate // 2
        high_cutoff = 500 / nyquist
        
        if high_cutoff < 1.0:
            b, a = signal.butter(3, high_cutoff, btype='high')
            transients = signal.filtfilt(b, a, audio_data)
        else:
            transients = audio_data
        
        # Transient density
        transient_threshold = np.std(transients) * 2
        transient_count = np.sum(np.abs(transients) > transient_threshold)
        transient_density = transient_count / len(transients)
        
        # Percussive to harmonic ratio
        percussive_energy = np.sum(transients ** 2)
        total_energy = np.sum(audio_data ** 2)
        percussive_ratio = percussive_energy / (total_energy + 1e-6)
        
        return {
            'transient_density': transient_density,
            'percussive_ratio': percussive_ratio
        }
    
    def _extract_bass_patterns(self, fft_magnitude: np.ndarray, 
                               frequencies: np.ndarray) -> Dict[str, float]:
        """Analyze bass patterns specific to hip hop"""
        
        # Focus on kick drum range
        kick_idx = np.where((frequencies >= self.kick_range[0]) & 
                           (frequencies <= self.kick_range[1]))[0]
        
        if len(kick_idx) == 0:
            return {
                'kick_consistency': 0.0,
                'bass_punch': 0.0,
                '808_presence': 0.0
            }
        
        kick_energy = np.sum(fft_magnitude[kick_idx])
        
        # Add to kick buffer for pattern analysis
        self.kick_buffer.append(kick_energy)
        
        # Kick consistency (how regular are the kicks)
        if len(self.kick_buffer) > 4:
            kick_array = np.array(self.kick_buffer)
            kick_consistency = 1.0 - (np.std(kick_array) / (np.mean(kick_array) + 1e-6))
            kick_consistency = np.clip(kick_consistency, 0, 1)
        else:
            kick_consistency = 0.0
        
        # Bass punch (peak to average ratio in bass region)
        bass_idx = np.where((frequencies >= self.bass_range[0]) & 
                           (frequencies <= self.bass_range[1]))[0]
        if len(bass_idx) > 0:
            bass_punch = np.max(fft_magnitude[bass_idx]) / (np.mean(fft_magnitude[bass_idx]) + 1e-6)
        else:
            bass_punch = 0.0
        
        # 808 presence (sustained sub-bass)
        sub_bass_idx = np.where((frequencies >= 30) & (frequencies <= 50))[0]
        if len(sub_bass_idx) > 0:
            presence_808 = np.sum(fft_magnitude[sub_bass_idx]) / (np.sum(fft_magnitude) + 1e-6)
        else:
            presence_808 = 0.0
        
        return {
            'kick_consistency': kick_consistency,
            'bass_punch': bass_punch,
            '808_presence': presence_808
        }
    
    def _calculate_hihat_density(self, fft_magnitude: np.ndarray, 
                                frequencies: np.ndarray) -> float:
        """Calculate hi-hat density (important for trap detection)"""
        
        hihat_idx = np.where((frequencies >= self.hihat_range[0]) & 
                            (frequencies <= self.hihat_range[1]))[0]
        
        if len(hihat_idx) == 0:
            return 0.0
        
        # Hi-hat energy relative to total
        hihat_energy = np.sum(fft_magnitude[hihat_idx])
        total_energy = np.sum(fft_magnitude)
        
        if total_energy == 0:
            return 0.0
        
        # Also check for rapid changes in hi-hat region (trap hi-hats)
        hihat_variance = np.var(fft_magnitude[hihat_idx])
        
        # Combine energy and variance for density measure
        density = (hihat_energy / total_energy) * (1 + hihat_variance)
        
        return np.clip(density, 0, 1)
    
    def get_temporal_features(self) -> Dict[str, float]:
        """Get features based on temporal history"""
        
        if len(self.feature_history) < 5:
            return {
                'bass_stability': 0.0,
                'rhythm_consistency': 0.0,
                'feature_confidence': 0.0
            }
        
        # Convert history to array for analysis
        history_array = []
        feature_names = list(self.feature_history[0].keys())
        
        for features in self.feature_history:
            history_array.append([features.get(name, 0) for name in feature_names])
        
        history_array = np.array(history_array)
        
        # Calculate stability metrics
        bass_features = ['sub_bass_ratio', 'bass_ratio', 'bass_dominance']
        bass_indices = [feature_names.index(f) for f in bass_features if f in feature_names]
        
        if bass_indices:
            bass_stability = 1.0 - np.mean([np.std(history_array[:, i]) 
                                           for i in bass_indices])
        else:
            bass_stability = 0.0
        
        # Rhythm consistency
        rhythm_features = ['onset_rate', 'rhythm_regularity']
        rhythm_indices = [feature_names.index(f) for f in rhythm_features if f in feature_names]
        
        if rhythm_indices:
            rhythm_consistency = 1.0 - np.mean([np.std(history_array[:, i]) 
                                               for i in rhythm_indices])
        else:
            rhythm_consistency = 0.0
        
        # Overall feature confidence based on consistency
        feature_confidence = np.mean([bass_stability, rhythm_consistency])
        
        return {
            'bass_stability': np.clip(bass_stability, 0, 1),
            'rhythm_consistency': np.clip(rhythm_consistency, 0, 1),
            'feature_confidence': np.clip(feature_confidence, 0, 1)
        }


class HipHopClassifier:
    """Classify hip hop music with high accuracy"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.feature_extractor = HipHopFeatureExtractor(sample_rate)
        
        # Classification thresholds (tuned for >70% accuracy)
        self.thresholds = {
            'sub_bass_ratio': (0.15, 0.35),      # Hip hop has strong sub-bass
            'bass_ratio': (0.25, 0.45),           # Significant bass presence
            'bass_dominance': (0.35, 0.60),       # Bass dominates the mix
            'spectral_tilt': (-0.4, -0.1),        # Negative tilt (bass-heavy)
            'kick_presence': (8.0, 20.0),         # Strong kicks
            'transient_density': (0.02, 0.15),    # Moderate transients
            'percussive_ratio': (0.3, 0.7),       # Balanced percussive content
            '808_presence': (0.05, 0.25),         # 808 bass presence
            'onset_rate': (2.0, 8.0),             # Moderate onset rate
            'rhythm_regularity': (0.6, 0.95)      # Regular rhythm
        }
        
        # Feature weights for scoring
        self.feature_weights = {
            'sub_bass_ratio': 0.15,
            'bass_ratio': 0.15,
            'bass_dominance': 0.20,
            'spectral_tilt': 0.10,
            'kick_presence': 0.10,
            '808_presence': 0.10,
            'rhythm_regularity': 0.10,
            'hihat_density': 0.05,
            'transient_density': 0.05
        }
        
        # Subgenre detection patterns
        self.subgenre_patterns = {
            'trap': {
                'hihat_density': (0.15, 1.0),
                '808_presence': (0.10, 0.30),
                'tempo_hint': (130, 170)
            },
            'boom_bap': {
                'kick_presence': (10.0, 25.0),
                'rhythm_regularity': (0.75, 0.95),
                'tempo_hint': (85, 100)
            },
            'modern': {
                'sub_bass_ratio': (0.20, 0.40),
                '808_presence': (0.08, 0.25),
                'tempo_hint': (70, 85)
            }
        }
        
        # Classification history for smoothing
        self.classification_history = deque(maxlen=15)
        self.confidence_history = deque(maxlen=15)
        
    def classify(self, fft_magnitude: np.ndarray, audio_data: np.ndarray, 
                frequencies: np.ndarray, additional_features: Optional[Dict] = None) -> Tuple[bool, float, str]:
        """
        Classify if audio is hip hop
        
        Returns:
            Tuple of (is_hip_hop, confidence, subgenre)
        """
        
        # Extract features
        features = self.feature_extractor.extract_features(
            fft_magnitude, audio_data, frequencies
        )
        
        # Add temporal features
        temporal_features = self.feature_extractor.get_temporal_features()
        features.update(temporal_features)
        
        # Add any additional features passed in
        if additional_features:
            features.update(additional_features)
        
        # Calculate hip hop score
        hip_hop_score = self._calculate_hip_hop_score(features)
        
        # Detect subgenre if hip hop
        subgenre = 'unknown'
        if hip_hop_score > 0.5:
            subgenre = self._detect_subgenre(features)
        
        # Apply temporal smoothing
        self.classification_history.append(hip_hop_score > 0.7)
        self.confidence_history.append(hip_hop_score)
        
        # Final decision based on history
        if len(self.classification_history) >= 5:
            # Require consistent detection
            recent_positive = sum(list(self.classification_history)[-5:])
            is_hip_hop = recent_positive >= 3  # 3 out of 5 frames
            
            # Average confidence
            confidence = np.mean(list(self.confidence_history)[-5:])
        else:
            is_hip_hop = hip_hop_score > 0.7
            confidence = hip_hop_score
        
        # Boost confidence if subgenre detected clearly
        if subgenre != 'unknown' and confidence > 0.6:
            confidence = min(confidence * 1.1, 1.0)
        
        return is_hip_hop, confidence, subgenre
    
    def _calculate_hip_hop_score(self, features: Dict[str, float]) -> float:
        """Calculate hip hop probability score"""
        
        score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in self.feature_weights.items():
            if feature_name not in features:
                continue
            
            value = features[feature_name]
            
            # Check if feature is in expected range
            if feature_name in self.thresholds:
                min_val, max_val = self.thresholds[feature_name]
                
                if min_val <= value <= max_val:
                    # Perfect match
                    feature_score = 1.0
                elif value < min_val:
                    # Below range
                    feature_score = max(0, 1 - (min_val - value) / min_val)
                else:
                    # Above range
                    feature_score = max(0, 1 - (value - max_val) / max_val)
            else:
                # No threshold, use value directly (for 0-1 features)
                feature_score = value
            
            score += feature_score * weight
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            score = score / total_weight
        
        # Apply feature confidence boost
        if 'feature_confidence' in features:
            score = score * (0.8 + 0.2 * features['feature_confidence'])
        
        return np.clip(score, 0, 1)
    
    def _detect_subgenre(self, features: Dict[str, float]) -> str:
        """Detect hip hop subgenre based on features"""
        
        best_match = 'unknown'
        best_score = 0.0
        
        for subgenre, patterns in self.subgenre_patterns.items():
            score = 0.0
            matches = 0
            
            for feature, (min_val, max_val) in patterns.items():
                if feature in features and feature != 'tempo_hint':
                    if min_val <= features[feature] <= max_val:
                        score += 1.0
                    matches += 1
            
            if matches > 0:
                score = score / matches
                if score > best_score:
                    best_score = score
                    best_match = subgenre
        
        # Require high confidence for subgenre detection
        if best_score < 0.7:
            return 'unknown'
        
        return best_match
    
    def get_feature_analysis(self) -> Dict[str, Any]:
        """Get detailed feature analysis for debugging"""
        
        if not self.feature_extractor.feature_history:
            return {}
        
        latest_features = self.feature_extractor.feature_history[-1]
        analysis = {}
        
        for feature_name, value in latest_features.items():
            if feature_name in self.thresholds:
                min_val, max_val = self.thresholds[feature_name]
                in_range = min_val <= value <= max_val
                analysis[feature_name] = {
                    'value': value,
                    'expected_range': (min_val, max_val),
                    'in_range': in_range,
                    'score': 1.0 if in_range else max(0, 1 - abs(value - (min_val + max_val) / 2))
                }
            else:
                analysis[feature_name] = {
                    'value': value,
                    'score': value
                }
        
        return analysis