"""
Hip-Hop Detection Module for OMEGA-4 Audio Analyzer
Advanced multi-layer analysis for accurate hip-hop genre detection
Addresses team concerns about better hip-hop recognition
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from scipy import signal as scipy_signal
import logging

logger = logging.getLogger(__name__)


class HipHopDetector:
    """Dedicated hip-hop detection with multi-layer analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # History buffers for temporal consistency
        self.sub_bass_history = deque(maxlen=30)  # 30 frames ~ 0.7 seconds
        self.kick_pattern_history = deque(maxlen=30)
        self.hihat_density_history = deque(maxlen=30)
        self.spectral_tilt_history = deque(maxlen=30)
        self.confidence_history = deque(maxlen=30)
        
        # Sub-genre specific parameters
        self.subgenre_profiles = {
            'trap': {
                'tempo_range': (130, 170),
                'hihat_density': (0.7, 1.0),  # Very high hi-hat density
                'sub_bass_presence': (0.8, 1.0),  # Heavy 808s
                'kick_spacing': 'syncopated',
                'snare_pattern': 'sparse',
                'vocal_style': 'melodic_autotuned'
            },
            'boom_bap': {
                'tempo_range': (80, 100),
                'hihat_density': (0.3, 0.6),  # Moderate hi-hat use
                'sub_bass_presence': (0.4, 0.7),  # Less sub-heavy
                'kick_spacing': 'regular',
                'snare_pattern': 'backbeat',
                'vocal_style': 'rhythmic_traditional'
            },
            'modern': {
                'tempo_range': (70, 90),
                'hihat_density': (0.5, 0.8),
                'sub_bass_presence': (0.6, 0.9),
                'kick_spacing': 'varied',
                'snare_pattern': 'trap_influenced',
                'vocal_style': 'mixed'
            }
        }
        
        # Feature weights for confidence scoring
        self.feature_weights = {
            'sub_bass': 0.25,      # 808s and sub-bass are crucial
            'kick_pattern': 0.20,   # Kick drum patterns
            'hihat_density': 0.15,  # Hi-hat patterns (especially for trap)
            'spectral_tilt': 0.15,  # Bass-heavy frequency distribution
            'vocal_presence': 0.15, # Rap vocals in specific range
            'tempo_match': 0.10     # Tempo in hip-hop ranges
        }
        
        # Pre-computed filter banks for efficiency
        self._init_filter_banks()
        
    def _init_filter_banks(self):
        """Initialize filter banks for sub-bass and frequency analysis"""
        # Sub-bass filter (20-60 Hz)
        nyquist = self.sample_rate / 2
        self.sub_bass_sos = scipy_signal.butter(
            4, [20, 60], btype='bandpass', fs=self.sample_rate, output='sos'
        )
        
        # Extended bass filter (60-250 Hz)
        self.bass_sos = scipy_signal.butter(
            4, [60, 250], btype='bandpass', fs=self.sample_rate, output='sos'
        )
        
        # Kick drum filter (40-100 Hz)
        self.kick_sos = scipy_signal.butter(
            4, [40, 100], btype='bandpass', fs=self.sample_rate, output='sos'
        )
        
        # Hi-hat filter (3000-10000 Hz)
        self.hihat_sos = scipy_signal.butter(
            4, [3000, min(10000, nyquist * 0.9)], btype='bandpass', 
            fs=self.sample_rate, output='sos'
        )
        
    def analyze(self, audio_chunk: np.ndarray, fft_data: np.ndarray, 
                freqs: np.ndarray, drum_info: Dict) -> Dict[str, float]:
        """Perform comprehensive hip-hop analysis"""
        
        # Check for silence first
        if audio_chunk is not None and len(audio_chunk) > 0:
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < 0.001:  # Silence threshold
                # Return no hip-hop detected during silence
                return {
                    'confidence': 0.0,
                    'features': {},
                    'subgenre': 'unknown',
                    'is_hip_hop': False,
                    'sub_bass_detected': False,
                    'strong_beat': False
                }
        
        # Multi-layer feature extraction
        features = {}
        
        # 1. Sub-bass and 808 detection (20-60Hz)
        features['sub_bass_presence'] = self._detect_sub_bass(audio_chunk, fft_data, freqs)
        
        # 2. Kick pattern analysis
        features['kick_pattern_score'] = self._analyze_kick_pattern(audio_chunk, drum_info)
        
        # 3. Hi-hat density analysis
        features['hihat_density'] = self._analyze_hihat_density(audio_chunk)
        
        # 4. Spectral tilt (bass-heavy distribution)
        features['spectral_tilt'] = self._analyze_spectral_tilt(fft_data, freqs)
        
        # 5. Vocal presence in rap frequency range
        features['vocal_presence'] = self._detect_rap_vocals(fft_data, freqs)
        
        # 6. Tempo estimation
        features['tempo_match'] = self._check_tempo_match(drum_info)
        
        # 7. Calculate overall hip-hop confidence
        confidence = self._calculate_confidence(features)
        
        # 8. Detect sub-genre if confidence is high
        subgenre = 'unknown'
        if confidence > 0.6:
            subgenre = self._detect_subgenre(features)
        
        # Update history for temporal consistency
        self._update_history(features, confidence)
        
        # Apply temporal smoothing
        smoothed_confidence = self._apply_temporal_smoothing()
        
        return {
            'confidence': smoothed_confidence,
            'features': features,
            'subgenre': subgenre,
            'is_hip_hop': smoothed_confidence > 0.5,
            'sub_bass_detected': features['sub_bass_presence'] > 0.6,
            'strong_beat': features['kick_pattern_score'] > 0.7
        }
    
    def _detect_sub_bass(self, audio_chunk: np.ndarray, fft_data: np.ndarray, 
                         freqs: np.ndarray) -> float:
        """Detect presence and strength of sub-bass (20-60Hz)"""
        if len(audio_chunk) == 0:
            return 0.0
        
        # Method 1: Time-domain filtering
        try:
            sub_bass_signal = scipy_signal.sosfilt(self.sub_bass_sos, audio_chunk)
            sub_bass_energy = np.sqrt(np.mean(sub_bass_signal ** 2))
            total_energy = np.sqrt(np.mean(audio_chunk ** 2))
            
            if total_energy > 0:
                time_domain_ratio = sub_bass_energy / total_energy
            else:
                time_domain_ratio = 0.0
        except Exception as e:
            logger.warning(f"Sub-bass filter error: {e}")
            time_domain_ratio = 0.0
        
        # Method 2: Frequency domain analysis
        if len(fft_data) > 0 and len(freqs) > 0:
            # Ensure arrays match in size
            min_len = min(len(fft_data), len(freqs))
            fft_data = fft_data[:min_len]
            freqs = freqs[:min_len]
            
            # Find sub-bass region
            sub_bass_mask = (freqs >= 20) & (freqs <= 60)
            bass_mask = (freqs >= 60) & (freqs <= 250)
            total_mask = freqs < self.sample_rate / 2
            
            if np.any(sub_bass_mask):
                sub_bass_power = np.sum(np.abs(fft_data[sub_bass_mask]) ** 2)
                bass_power = np.sum(np.abs(fft_data[bass_mask]) ** 2)
                total_power = np.sum(np.abs(fft_data[total_mask]) ** 2)
                
                if total_power > 0:
                    freq_domain_ratio = sub_bass_power / total_power
                    # Check if sub-bass dominates the bass region
                    sub_to_bass_ratio = sub_bass_power / (bass_power + 1e-10)
                else:
                    freq_domain_ratio = 0.0
                    sub_to_bass_ratio = 0.0
            else:
                freq_domain_ratio = 0.0
                sub_to_bass_ratio = 0.0
        else:
            freq_domain_ratio = 0.0
            sub_to_bass_ratio = 0.0
        
        # Combine methods with emphasis on frequency domain
        presence = (0.3 * time_domain_ratio + 0.5 * freq_domain_ratio + 
                   0.2 * min(sub_to_bass_ratio, 1.0))
        
        # Hip-hop typically has very prominent sub-bass
        # Scale up the presence score
        return min(presence * 3.0, 1.0)
    
    def _analyze_kick_pattern(self, audio_chunk: np.ndarray, drum_info: Dict) -> float:
        """Analyze kick drum patterns characteristic of hip-hop"""
        # Get kick detection from drum info
        kick_detected = drum_info.get('kick', {}).get('kick_detected', False)
        kick_magnitude = drum_info.get('kick', {}).get('magnitude', 0.0)
        
        if not kick_detected:
            return 0.0
        
        # Filter for kick frequencies
        try:
            kick_signal = scipy_signal.sosfilt(self.kick_sos, audio_chunk)
            
            # Detect transients in kick signal
            envelope = np.abs(scipy_signal.hilbert(kick_signal))
            
            # Find peaks (potential kick hits)
            mean_envelope = np.mean(envelope)
            std_envelope = np.std(envelope)
            threshold = mean_envelope + 1.5 * std_envelope
            
            peaks = scipy_signal.find_peaks(envelope, height=threshold, distance=int(0.1 * self.sample_rate))[0]
            
            if len(peaks) > 1:
                # Analyze spacing between kicks
                spacings = np.diff(peaks)
                
                # Hip-hop often has regular kick patterns
                spacing_regularity = 1.0 - (np.std(spacings) / (np.mean(spacings) + 1e-10))
                spacing_regularity = max(0, min(spacing_regularity, 1))
                
                # Check for syncopation (off-beat kicks)
                expected_spacing = np.median(spacings)
                syncopation = np.sum(np.abs(spacings - expected_spacing) > expected_spacing * 0.2) / len(spacings)
                
                # Combine factors
                pattern_score = (0.6 * spacing_regularity + 0.4 * (1 - syncopation)) * kick_magnitude
            else:
                pattern_score = kick_magnitude * 0.5
                
        except Exception as e:
            logger.warning(f"Kick pattern analysis error: {e}")
            pattern_score = kick_magnitude * 0.5
        
        return min(pattern_score, 1.0)
    
    def _analyze_hihat_density(self, audio_chunk: np.ndarray) -> float:
        """Analyze hi-hat density (crucial for trap detection)"""
        if len(audio_chunk) == 0:
            return 0.0
        
        try:
            # Filter for hi-hat frequencies
            hihat_signal = scipy_signal.sosfilt(self.hihat_sos, audio_chunk)
            
            # Calculate zero-crossing rate in hi-hat band
            zero_crossings = np.where(np.diff(np.sign(hihat_signal)))[0]
            zcr = len(zero_crossings) / len(hihat_signal)
            
            # Calculate energy in hi-hat band
            hihat_energy = np.sqrt(np.mean(hihat_signal ** 2))
            total_energy = np.sqrt(np.mean(audio_chunk ** 2))
            
            if total_energy > 0:
                energy_ratio = hihat_energy / total_energy
            else:
                energy_ratio = 0.0
            
            # Combine ZCR and energy for density measure
            # High ZCR + significant energy = dense hi-hats
            density = min(zcr * 10, 1.0) * min(energy_ratio * 5, 1.0)
            
        except Exception as e:
            logger.warning(f"Hi-hat analysis error: {e}")
            density = 0.0
        
        return density
    
    def _analyze_spectral_tilt(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Analyze spectral tilt (bass-heavy = positive tilt for hip-hop)"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        # Ensure arrays match
        min_len = min(len(fft_data), len(freqs))
        fft_data = fft_data[:min_len]
        freqs = freqs[:min_len]
        
        # Calculate energy in frequency bands
        low_mask = (freqs >= 20) & (freqs <= 500)
        mid_mask = (freqs >= 500) & (freqs <= 2000)
        high_mask = (freqs >= 2000) & (freqs <= 8000)
        
        if not (np.any(low_mask) and np.any(mid_mask) and np.any(high_mask)):
            return 0.0
        
        low_energy = np.sum(np.abs(fft_data[low_mask]) ** 2)
        mid_energy = np.sum(np.abs(fft_data[mid_mask]) ** 2)
        high_energy = np.sum(np.abs(fft_data[high_mask]) ** 2)
        
        total_energy = low_energy + mid_energy + high_energy
        if total_energy == 0:
            return 0.0
        
        # Calculate tilt (positive = bass-heavy)
        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Hip-hop typically has much more low than high
        tilt = (low_ratio - high_ratio + 1.0) / 2.0  # Normalize to 0-1
        
        # Emphasize strong bass tilt
        if low_ratio > 0.5:  # More than half energy in lows
            tilt = min(tilt * 1.2, 1.0)
        
        return tilt
    
    def _detect_rap_vocals(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """Detect presence of rap vocals in characteristic frequency range"""
        if len(fft_data) == 0 or len(freqs) == 0:
            return 0.0
        
        # Ensure arrays match
        min_len = min(len(fft_data), len(freqs))
        fft_data = fft_data[:min_len]
        freqs = freqs[:min_len]
        
        # Rap vocals typically centered around 200-3000 Hz
        # with emphasis on 300-2000 Hz (speech fundamentals)
        vocal_mask = (freqs >= 200) & (freqs <= 3000)
        core_vocal_mask = (freqs >= 300) & (freqs <= 2000)
        
        if not np.any(vocal_mask):
            return 0.0
        
        # Look for energy concentration in vocal range
        vocal_energy = np.sum(np.abs(fft_data[vocal_mask]) ** 2)
        core_vocal_energy = np.sum(np.abs(fft_data[core_vocal_mask]) ** 2)
        total_energy = np.sum(np.abs(fft_data) ** 2)
        
        if total_energy == 0 or vocal_energy == 0:
            return 0.0
        
        # Check for formant-like peaks in vocal range
        vocal_magnitude = np.abs(fft_data[vocal_mask])
        
        # Find peaks in vocal range
        if len(vocal_magnitude) > 10:
            peaks, properties = scipy_signal.find_peaks(
                vocal_magnitude, 
                prominence=np.max(vocal_magnitude) * 0.3,
                distance=20
            )
            
            # More peaks = more complex vocal content
            peak_density = len(peaks) / len(vocal_magnitude)
            
            # Energy concentration
            vocal_ratio = vocal_energy / total_energy
            core_concentration = core_vocal_energy / vocal_energy
            
            # Combine factors
            presence = (0.4 * vocal_ratio + 0.3 * core_concentration + 0.3 * peak_density)
            
            # Scale up for typical rap vocal prominence
            presence = min(presence * 2.5, 1.0)
        else:
            presence = 0.0
        
        return presence
    
    def _check_tempo_match(self, drum_info: Dict) -> float:
        """Check if tempo matches hip-hop ranges"""
        # This is simplified - in real implementation would use beat tracking
        kick_detected = drum_info.get('kick', {}).get('kick_detected', False)
        
        if kick_detected:
            # Assume tempo is in hip-hop range if strong kicks detected
            # Real implementation would calculate actual BPM
            return 0.8
        
        return 0.3
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate overall hip-hop confidence score"""
        confidence = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in features:
                confidence += features[feature] * weight
        
        # Apply non-linear scaling to emphasize strong matches
        if confidence > 0.7:
            confidence = min(confidence * 1.2, 1.0)
        elif confidence < 0.3:
            confidence = confidence * 0.8
        
        return confidence
    
    def _detect_subgenre(self, features: Dict[str, float]) -> str:
        """Detect hip-hop subgenre based on features"""
        scores = {}
        
        # Trap detection
        trap_score = 0.0
        if features.get('hihat_density', 0) > 0.7:
            trap_score += 0.4
        if features.get('sub_bass_presence', 0) > 0.8:
            trap_score += 0.3
        if features.get('tempo_match', 0) > 0.5:  # Assuming trap tempo
            trap_score += 0.3
        scores['trap'] = trap_score
        
        # Boom bap detection
        boom_bap_score = 0.0
        if 0.3 <= features.get('hihat_density', 0) <= 0.6:
            boom_bap_score += 0.3
        if 0.4 <= features.get('sub_bass_presence', 0) <= 0.7:
            boom_bap_score += 0.3
        if features.get('kick_pattern_score', 0) > 0.8:  # Regular pattern
            boom_bap_score += 0.4
        scores['boom_bap'] = boom_bap_score
        
        # Modern hip-hop (default)
        modern_score = 0.5  # Base score
        if features.get('sub_bass_presence', 0) > 0.6:
            modern_score += 0.2
        if features.get('vocal_presence', 0) > 0.6:
            modern_score += 0.3
        scores['modern'] = modern_score
        
        # Return highest scoring subgenre
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _update_history(self, features: Dict[str, float], confidence: float):
        """Update history buffers for temporal consistency"""
        self.sub_bass_history.append(features.get('sub_bass_presence', 0))
        self.kick_pattern_history.append(features.get('kick_pattern_score', 0))
        self.hihat_density_history.append(features.get('hihat_density', 0))
        self.spectral_tilt_history.append(features.get('spectral_tilt', 0))
        self.confidence_history.append(confidence)
    
    def _apply_temporal_smoothing(self) -> float:
        """Apply temporal smoothing for stable detection"""
        if len(self.confidence_history) == 0:
            return 0.0
        
        # Use weighted average with recent frames having more weight
        weights = np.linspace(0.5, 1.0, len(self.confidence_history))
        weights = weights / np.sum(weights)
        
        smoothed = np.sum(np.array(self.confidence_history) * weights)
        
        # If we've had consistent high confidence, boost it
        recent_high = sum(1 for c in list(self.confidence_history)[-10:] if c > 0.6)
        if recent_high >= 7:
            smoothed = min(smoothed * 1.1, 1.0)
        
        return smoothed
    
    def get_debug_info(self) -> Dict[str, any]:
        """Get detailed debug information for analysis"""
        return {
            'sub_bass_history': list(self.sub_bass_history),
            'kick_pattern_history': list(self.kick_pattern_history),
            'hihat_density_history': list(self.hihat_density_history),
            'spectral_tilt_history': list(self.spectral_tilt_history),
            'confidence_history': list(self.confidence_history),
            'avg_sub_bass': np.mean(self.sub_bass_history) if self.sub_bass_history else 0,
            'avg_hihat_density': np.mean(self.hihat_density_history) if self.hihat_density_history else 0,
            'current_confidence': self.confidence_history[-1] if self.confidence_history else 0
        }