"""
Voice Detection Module for OMEGA-4 Audio Analyzer
Phase 5: Wrapper for industry voice detection integration
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional

# Add path for industry voice detection module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "voice_detection"))

try:
    from industry_voice_detection import IndustryVoiceDetector
    VOICE_DETECTION_AVAILABLE = True
except ImportError:
    VOICE_DETECTION_AVAILABLE = False
    print("Warning: Industry voice detection module not available")


class VoiceDetectionWrapper:
    """Wrapper for industry voice detection with fallback support"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.detector = None
        
        if VOICE_DETECTION_AVAILABLE:
            try:
                self.detector = IndustryVoiceDetector(sample_rate)
                print("✓ Industry voice detection initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize voice detection: {e}")
                self.detector = None
        else:
            print("⚠ Voice detection not available - using fallback")
    
    def detect_voice_realtime(self, audio_data: np.ndarray) -> Dict:
        """Detect voice activity in real-time audio data"""
        if self.detector:
            try:
                result = self.detector.detect_voice_realtime(audio_data)
                # Ensure consistent format
                if isinstance(result, dict):
                    # Add any missing fields
                    default = self._get_default_response()
                    for key in default:
                        if key not in result:
                            result[key] = default[key]
                    return result
                else:
                    # Handle non-dict results
                    return self._get_default_response()
            except Exception as e:
                print(f"Voice detection error: {e}")
                return self._simple_voice_detection(audio_data)
        else:
            # Use simple voice detection as fallback
            return self._simple_voice_detection(audio_data)
    
    def analyze_formants(self, audio_data: np.ndarray) -> List[float]:
        """Analyze formant frequencies in audio data"""
        if self.detector and hasattr(self.detector, 'analyze_formants'):
            try:
                return self.detector.analyze_formants(audio_data)
            except:
                return []
        else:
            # Simple formant estimation fallback
            return self._estimate_formants_simple(audio_data)
    
    def _get_default_response(self) -> Dict:
        """Return default voice detection response"""
        return {
            'voice_detected': False,
            'confidence': 0.0,
            'voice_probability': 0.0,
            'pitch': 0.0,
            'formants': [],
            'speaking_rate': 0.0,
            'gender': 'unknown'
        }
    
    def _estimate_formants_simple(self, audio_data: np.ndarray) -> List[float]:
        """Simple formant estimation using spectral peaks"""
        if len(audio_data) < 1024:
            return []
            
        # Apply window
        windowed = audio_data[:1024] * np.hanning(1024)
        
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(1024, 1/self.sample_rate)
        
        # Find peaks in voice frequency range
        voice_mask = (freqs > 80) & (freqs < 5000)
        voice_spectrum = spectrum[voice_mask]
        voice_freqs = freqs[voice_mask]
        
        # Simple peak detection - more sensitive
        formants = []
        if len(voice_spectrum) > 10:
            # Find local maxima
            mean_level = np.mean(voice_spectrum)
            for i in range(1, len(voice_spectrum) - 1):
                if (voice_spectrum[i] > voice_spectrum[i-1] and 
                    voice_spectrum[i] > voice_spectrum[i+1] and
                    voice_spectrum[i] > mean_level * 1.5):  # Lower threshold
                    formants.append(voice_freqs[i])
                    
                if len(formants) >= 4:  # Typically F1-F4
                    break
                    
        return formants[:4]  # Return up to 4 formants
    
    def is_available(self) -> bool:
        """Check if voice detection is available"""
        return self.detector is not None
    
    def _simple_voice_detection(self, audio_data: np.ndarray) -> Dict:
        """Simple voice detection based on spectral characteristics"""
        response = self._get_default_response()
        
        if len(audio_data) < 2048:
            return response
        
        # Apply window
        windowed = audio_data[:2048] * np.hanning(2048)
        
        # Compute spectrum
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(2048, 1/self.sample_rate)
        
        # Voice frequency range (80Hz - 5000Hz)
        voice_mask = (freqs >= 80) & (freqs <= 5000)
        voice_spectrum = spectrum[voice_mask]
        
        # Check for voice characteristics
        if len(voice_spectrum) > 0:
            # 1. Energy in voice range
            voice_energy = np.sum(voice_spectrum)
            total_energy = np.sum(spectrum)
            
            if total_energy > 0:
                voice_ratio = voice_energy / total_energy
            else:
                voice_ratio = 0
            
            # 2. Spectral centroid in voice range
            voice_freqs = freqs[voice_mask]
            if np.sum(voice_spectrum) > 0:
                centroid = np.sum(voice_freqs * voice_spectrum) / np.sum(voice_spectrum)
            else:
                centroid = 0
            
            # 3. Check for harmonic structure (formants)
            formants = self._estimate_formants_simple(audio_data)
            
            # 4. Voice detection heuristics
            has_voice = False
            confidence = 0.0
            
            # Check if energy is concentrated in voice range - more sensitive
            if voice_ratio > 0.35 and 120 < centroid < 4500:
                has_voice = True
                confidence = min(voice_ratio * 200, 90)
            
            # Check for formant structure
            if len(formants) >= 2:
                has_voice = True
                confidence = max(confidence, 75)
            
            # Additional check for pop vocals (wider range)
            if voice_ratio > 0.25 and 80 < centroid < 5500 and len(formants) >= 1:
                has_voice = True  
                confidence = max(confidence, 60)
                
            # Check for any significant energy in vocal range
            if voice_ratio > 0.2 and np.max(voice_spectrum) > np.mean(spectrum) * 3:
                has_voice = True
                confidence = max(confidence, 40)
            
            # Additional check for singing voice (stronger harmonics)
            if has_voice:
                # Check for pitch stability and harmonic richness
                harmonic_mask = (freqs >= 200) & (freqs <= 2000)
                harmonic_spectrum = spectrum[harmonic_mask]
                if len(harmonic_spectrum) > 0:
                    peak_indices = np.where(harmonic_spectrum > np.mean(harmonic_spectrum) * 2)[0]
                    if len(peak_indices) > 3:  # Multiple harmonics
                        confidence = min(confidence * 1.2, 95)
            
            response['voice_detected'] = has_voice
            response['confidence'] = confidence
            response['voice_probability'] = confidence / 100.0
            response['formants'] = formants
            
            # Estimate pitch from fundamental frequency
            if has_voice and len(voice_spectrum) > 0:
                # Find the first significant peak
                threshold = np.max(voice_spectrum) * 0.3
                peaks = np.where(voice_spectrum > threshold)[0]
                if len(peaks) > 0:
                    pitch_idx = peaks[0]
                    response['pitch'] = voice_freqs[pitch_idx]
        
        return response