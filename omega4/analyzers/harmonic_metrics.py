"""
Harmonic Metrics Calculator for OMEGA-4 Audio Analyzer
Provides THD, THD+N, formant analysis, and spectral metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal as scipy_signal
from scipy.signal import lfilter, butter
from scipy.linalg import solve_toeplitz


class HarmonicMetrics:
    """Calculate various harmonic and spectral metrics"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
        # Formant ranges for different voice types
        self.formant_ranges = {
            'male': {
                'F1': (200, 900),    # First formant range
                'F2': (700, 2300),   # Second formant range
                'F3': (1500, 3500),  # Third formant range
                'F4': (2500, 4500)   # Fourth formant range
            },
            'female': {
                'F1': (250, 1000),
                'F2': (800, 2800),
                'F3': (1800, 4000),
                'F4': (3000, 5000)
            },
            'child': {
                'F1': (300, 1100),
                'F2': (900, 3200),
                'F3': (2000, 4500),
                'F4': (3500, 5500)
            }
        }
        
        # Vowel formant patterns (average Hz values)
        self.vowel_formants = {
            'a': {'F1': 730, 'F2': 1090, 'F3': 2440},  # "father"
            'e': {'F1': 530, 'F2': 1840, 'F3': 2480},  # "bet"
            'i': {'F1': 270, 'F2': 2290, 'F3': 3010},  # "beat"
            'o': {'F1': 570, 'F2': 840, 'F3': 2410},   # "bought"
            'u': {'F1': 300, 'F2': 870, 'F3': 2240},   # "boot"
        }
    
    def calculate_thd(self, fft_data: np.ndarray, freqs: np.ndarray, 
                     fundamental_freq: float, num_harmonics: int = 5) -> float:
        """
        Calculate Total Harmonic Distortion (THD)
        THD = sqrt(sum(V_n^2 for n=2 to N)) / V_1 * 100%
        """
        if fundamental_freq <= 0 or fundamental_freq > self.nyquist:
            return 0.0
        
        # Find fundamental bin
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_power = fft_data[fund_idx] ** 2
        
        if fund_power < 1e-10:
            return 0.0
        
        # Sum harmonic powers
        harmonic_power_sum = 0.0
        
        for n in range(2, num_harmonics + 1):
            harmonic_freq = fundamental_freq * n
            if harmonic_freq > self.nyquist:
                break
            
            # Find harmonic bin (with tolerance)
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if abs(freqs[harmonic_idx] - harmonic_freq) < fundamental_freq * 0.1:
                harmonic_power_sum += fft_data[harmonic_idx] ** 2
        
        # Calculate THD percentage
        thd = np.sqrt(harmonic_power_sum / fund_power) * 100
        return min(thd, 100.0)  # Cap at 100%
    
    def calculate_thd_plus_noise(self, fft_data: np.ndarray, freqs: np.ndarray,
                                fundamental_freq: float, num_harmonics: int = 5) -> float:
        """
        Calculate Total Harmonic Distortion + Noise (THD+N)
        THD+N = sqrt(sum(all non-fundamental power)) / fundamental power * 100%
        """
        if fundamental_freq <= 0 or fundamental_freq > self.nyquist:
            return 0.0
        
        # Find fundamental and harmonic bins
        fund_idx = np.argmin(np.abs(freqs - fundamental_freq))
        fund_power = fft_data[fund_idx] ** 2
        
        if fund_power < 1e-10:
            return 0.0
        
        # Total power in spectrum
        total_power = np.sum(fft_data ** 2)
        
        # Remove fundamental and its immediate neighbors
        bandwidth = fundamental_freq * 0.05  # 5% bandwidth
        fund_start = np.argmin(np.abs(freqs - (fundamental_freq - bandwidth)))
        fund_end = np.argmin(np.abs(freqs - (fundamental_freq + bandwidth)))
        
        # Calculate non-fundamental power
        non_fund_power = total_power - np.sum(fft_data[fund_start:fund_end+1] ** 2)
        
        # Calculate THD+N percentage
        thd_n = np.sqrt(non_fund_power / fund_power) * 100
        return min(thd_n, 100.0)
    
    def detect_formants_lpc(self, audio_data: np.ndarray, order: int = 14) -> List[float]:
        """
        Detect formants using Linear Predictive Coding (LPC)
        Returns formant frequencies F1-F4
        """
        if len(audio_data) < order + 1:
            return []
        
        # Ensure we have enough data and it's not silent
        if np.max(np.abs(audio_data)) < 1e-10:
            return []
        
        # Pre-emphasis filter to enhance high frequencies
        pre_emphasis = 0.97
        emphasized = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Apply Hamming window
        windowed = emphasized * np.hamming(len(emphasized))
        
        # Compute autocorrelation
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Ensure we have enough autocorrelation values
        if len(autocorr) < order + 1:
            return []
        
        # Check for zero autocorrelation at lag 0 (would cause division by zero)
        if abs(autocorr[0]) < 1e-10:
            return []
        
        # Levinson-Durbin recursion for LPC coefficients
        try:
            lpc_coeffs = self._levinson_durbin(autocorr[:order+1], order)
        except Exception:
            return []
        
        # Find roots of LPC polynomial
        roots = np.roots(np.append([1], -lpc_coeffs))
        
        # Convert roots to frequencies
        formants = []
        for root in roots:
            if np.imag(root) > 0:  # Only positive frequency roots
                angle = np.angle(root)
                freq = angle * self.sample_rate / (2 * np.pi)
                if 50 < freq < self.nyquist:  # Valid frequency range
                    formants.append(freq)
        
        # Sort and return first 4 formants
        formants.sort()
        return formants[:4]
    
    def _levinson_durbin(self, r: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin recursion for solving Toeplitz system"""
        # Initialize
        a = np.zeros(order + 1)
        a[0] = 1.0
        k = np.zeros(order)
        
        # Recursion
        for m in range(order):
            # Calculate reflection coefficient
            # Need to properly index the autocorrelation
            if m == 0:
                k_m = -r[1] / r[0]
            else:
                # Compute sum: a[0]*r[m] + a[1]*r[m-1] + ... + a[m]*r[0]
                sum_val = 0.0
                for j in range(m + 1):
                    sum_val += a[j] * r[m - j]
                k_m = -sum_val / r[0]
            
            k[m] = k_m
            
            # Update coefficients
            a_new = a.copy()
            for i in range(1, m + 2):
                a_new[i] = a[i] + k_m * a[m + 1 - i]
            a = a_new
        
        return -a[1:]  # Return LPC coefficients (without leading 1)
    
    def calculate_spectral_centroid(self, fft_data: np.ndarray, freqs: np.ndarray) -> float:
        """
        Calculate spectral centroid (center of mass of spectrum)
        Indicates brightness of sound
        """
        magnitudes = np.abs(fft_data)
        
        # Weighted average of frequencies
        if np.sum(magnitudes) > 0:
            centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes)
            return centroid
        return 0.0
    
    def calculate_spectral_spread(self, fft_data: np.ndarray, freqs: np.ndarray,
                                 centroid: Optional[float] = None) -> float:
        """
        Calculate spectral spread (standard deviation around centroid)
        Indicates how spread out the spectrum is
        """
        if centroid is None:
            centroid = self.calculate_spectral_centroid(fft_data, freqs)
        
        magnitudes = np.abs(fft_data)
        
        if np.sum(magnitudes) > 0:
            variance = np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes)
            return np.sqrt(variance)
        return 0.0
    
    def calculate_spectral_flux(self, current_fft: np.ndarray, previous_fft: np.ndarray) -> float:
        """
        Calculate spectral flux (change between consecutive spectra)
        Useful for onset detection
        """
        # Ensure same size
        min_len = min(len(current_fft), len(previous_fft))
        current = np.abs(current_fft[:min_len])
        previous = np.abs(previous_fft[:min_len])
        
        # Half-wave rectification (only positive changes)
        diff = current - previous
        positive_diff = np.maximum(diff, 0)
        
        return np.sum(positive_diff)
    
    def calculate_spectral_rolloff(self, fft_data: np.ndarray, freqs: np.ndarray,
                                  percentage: float = 0.85) -> float:
        """
        Calculate spectral rolloff frequency
        Frequency below which 'percentage' of total spectral energy is contained
        """
        magnitudes = np.abs(fft_data)
        cumsum = np.cumsum(magnitudes)
        
        if cumsum[-1] > 0:
            threshold = percentage * cumsum[-1]
            rolloff_idx = np.where(cumsum >= threshold)[0][0]
            return freqs[rolloff_idx]
        return 0.0
    
    def calculate_harmonic_to_noise_ratio(self, fft_data: np.ndarray, freqs: np.ndarray,
                                        fundamental_freq: float, num_harmonics: int = 5) -> float:
        """
        Calculate Harmonic-to-Noise Ratio (HNR) in dB
        Useful for voice quality assessment
        """
        if fundamental_freq <= 0:
            return 0.0
        
        # Calculate harmonic energy
        harmonic_energy = 0.0
        harmonic_bins = []
        
        for n in range(1, num_harmonics + 1):
            harmonic_freq = fundamental_freq * n
            if harmonic_freq > self.nyquist:
                break
            
            # Find harmonic bin with tolerance
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if abs(freqs[harmonic_idx] - harmonic_freq) < fundamental_freq * 0.1:
                harmonic_energy += fft_data[harmonic_idx] ** 2
                
                # Mark bins around harmonic as harmonic bins
                bandwidth = int(fundamental_freq * 0.05 / (freqs[1] - freqs[0]))
                for i in range(max(0, harmonic_idx - bandwidth), 
                             min(len(fft_data), harmonic_idx + bandwidth + 1)):
                    harmonic_bins.append(i)
        
        # Calculate noise energy (everything except harmonics)
        noise_energy = 0.0
        for i in range(len(fft_data)):
            if i not in harmonic_bins:
                noise_energy += fft_data[i] ** 2
        
        if noise_energy > 0:
            hnr_db = 10 * np.log10(harmonic_energy / noise_energy)
            return max(0, min(hnr_db, 60))  # Clamp to reasonable range
        return 60.0  # No noise detected
    
    def detect_vowel_from_formants(self, formants: List[float]) -> Tuple[str, float]:
        """
        Detect vowel from formant frequencies
        Returns vowel and confidence
        """
        if len(formants) < 2:
            return 'unknown', 0.0
        
        f1, f2 = formants[0], formants[1]
        
        best_vowel = 'unknown'
        best_distance = float('inf')
        
        # Compare with known vowel formants
        for vowel, formant_values in self.vowel_formants.items():
            distance = np.sqrt((f1 - formant_values['F1'])**2 + 
                             (f2 - formant_values['F2'])**2)
            
            if distance < best_distance:
                best_distance = distance
                best_vowel = vowel
        
        # Calculate confidence (inverse of normalized distance)
        max_distance = 1000  # Maximum expected distance
        confidence = max(0, 1 - (best_distance / max_distance))
        
        return best_vowel, confidence
    
    def calculate_inharmonicity(self, fft_data: np.ndarray, freqs: np.ndarray,
                               fundamental_freq: float, num_harmonics: int = 10) -> float:
        """
        Calculate inharmonicity coefficient
        Measures deviation from perfect harmonic series (important for pianos, bells)
        """
        if fundamental_freq <= 0:
            return 0.0
        
        deviations = []
        
        for n in range(2, num_harmonics + 1):
            expected_freq = fundamental_freq * n
            if expected_freq > self.nyquist:
                break
            
            # Find actual peak near expected harmonic
            search_range = fundamental_freq * 0.1
            start_idx = np.argmin(np.abs(freqs - (expected_freq - search_range)))
            end_idx = np.argmin(np.abs(freqs - (expected_freq + search_range)))
            
            if end_idx > start_idx:
                peak_idx = start_idx + np.argmax(fft_data[start_idx:end_idx])
                actual_freq = freqs[peak_idx]
                
                # Calculate relative deviation
                deviation = abs(actual_freq - expected_freq) / expected_freq
                deviations.append(deviation)
        
        if deviations:
            return np.mean(deviations)
        return 0.0