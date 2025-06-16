"""
Professional Meters Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract professional metering as self-contained module
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from scipy import signal as scipy_signal


class ProfessionalMetering:
    """Professional audio metering standards (LUFS, K-weighting, True Peak)"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

        # LUFS measurement history
        self.lufs_momentary_history = deque(maxlen=int(0.4 * 60))  # 400ms at 60 FPS
        self.lufs_short_term_history = deque(maxlen=int(3.0 * 60))  # 3s at 60 FPS
        self.lufs_integrated_history = deque(maxlen=int(60 * 60))  # 60s max integration

        # True peak detection
        self.peak_history = deque(maxlen=int(1.0 * 60))  # 1s peak hold

        # K-weighting filter coefficients
        self.k_weighting_filter = self.create_k_weighting_filter()

        # Gate threshold for integrated LUFS
        self.gate_threshold = -70.0  # LUFS

        # Current values
        self.current_lufs = {
            "momentary": -100.0,
            "short_term": -100.0,
            "integrated": -100.0,
            "range": 0.0,
            "true_peak": -100.0
        }
        self.current_true_peak = -100.0

    def create_k_weighting_filter(self):
        """Create K-weighting filter (simplified for real-time use)"""
        nyquist = self.sample_rate / 2

        # High-pass at 38 Hz
        hp_freq = 38 / nyquist
        hp_b, hp_a = scipy_signal.butter(2, hp_freq, btype="high")

        # High-shelf at 1500 Hz (+4 dB)
        shelf_freq = 1500 / nyquist
        shelf_gain_db = 4.0
        shelf_gain_linear = 10 ** (shelf_gain_db / 20)

        # Simplified shelf filter
        shelf_b, shelf_a = scipy_signal.iirfilter(
            2, shelf_freq, btype="high", ftype="butter", output="ba"
        )

        return {
            "hp_b": hp_b,
            "hp_a": hp_a,
            "shelf_b": shelf_b,
            "shelf_a": shelf_a,
            "shelf_gain": shelf_gain_linear,
        }

    def apply_k_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply K-weighting filter to audio data"""
        # Check if we have very low signal
        input_rms = np.sqrt(np.mean(audio_data**2))
        if input_rms < 1e-6:
            return np.zeros_like(audio_data)
            
        # Apply high-pass filter
        filtered = scipy_signal.filtfilt(
            self.k_weighting_filter["hp_b"], 
            self.k_weighting_filter["hp_a"], 
            audio_data
        )

        # Apply simplified high-shelf boost
        shelf_filtered = scipy_signal.filtfilt(
            self.k_weighting_filter["shelf_b"], 
            self.k_weighting_filter["shelf_a"], 
            filtered
        )

        # Combine with original for shelf effect
        result = filtered + (shelf_filtered - filtered) * 0.3

        return result

    def calculate_lufs(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate LUFS measurements"""
        if len(audio_data) == 0:
            return self.current_lufs

        # Apply K-weighting
        weighted = self.apply_k_weighting(audio_data)

        # Calculate mean square power
        mean_square = np.mean(weighted**2)

        # Convert to LUFS
        if mean_square > 1e-10:
            lufs_instantaneous = -0.691 + 10 * np.log10(mean_square)
        else:
            lufs_instantaneous = -100.0

        # Update histories
        self.lufs_momentary_history.append(lufs_instantaneous)
        self.lufs_short_term_history.append(lufs_instantaneous)
        self.lufs_integrated_history.append(lufs_instantaneous)

        # Calculate momentary (400ms)
        if len(self.lufs_momentary_history) > 0:
            self.current_lufs["momentary"] = np.mean(self.lufs_momentary_history)
        else:
            self.current_lufs["momentary"] = lufs_instantaneous

        # Calculate short-term (3s)
        if len(self.lufs_short_term_history) > 0:
            self.current_lufs["short_term"] = np.mean(self.lufs_short_term_history)
        else:
            self.current_lufs["short_term"] = lufs_instantaneous

        # Calculate integrated with gating
        if len(self.lufs_integrated_history) > 0:
            gated_values = [v for v in self.lufs_integrated_history if v > self.gate_threshold]
            if gated_values:
                self.current_lufs["integrated"] = np.mean(gated_values)
                # Calculate loudness range (simplified)
                self.current_lufs["range"] = np.percentile(gated_values, 95) - np.percentile(gated_values, 10)
            else:
                self.current_lufs["integrated"] = -100.0
                self.current_lufs["range"] = 0.0

        # True peak detection
        true_peak_db = 20 * np.log10(np.max(np.abs(audio_data)) + 1e-10)
        self.peak_history.append(true_peak_db)
        self.current_lufs["true_peak"] = max(self.peak_history)

        return self.current_lufs


class ProfessionalMetersPanel:
    """Professional meters panel with LUFS, True Peak, and transient analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.metering = ProfessionalMetering(sample_rate)
        
        # Transient analysis
        self.transient_info = {
            'attack_time': 0.0,
            'punch_factor': 0.0,
            'transients_detected': 0
        }
        
        # Fonts will be set by main app
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_meters = None
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_large = fonts.get('large')
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        self.font_meters = fonts.get('meters', fonts.get('small'))
        
    def update(self, audio_data: np.ndarray):
        """Update meters with new audio data"""
        # Update LUFS measurements
        self.lufs_info = self.metering.calculate_lufs(audio_data)
        
        # Simple transient detection
        if len(audio_data) > 1:
            # Energy envelope
            envelope = np.abs(audio_data)
            
            # Simple attack detection
            diff = np.diff(envelope)
            attack_indices = np.where(diff > 0.1)[0]
            
            if len(attack_indices) > 0:
                # Estimate attack time (simplified)
                attack_samples = attack_indices[0] if len(attack_indices) > 0 else 0
                self.transient_info['attack_time'] = (attack_samples / self.sample_rate) * 1000
                
                # Punch factor (ratio of peak to RMS)
                rms = np.sqrt(np.mean(audio_data**2))
                peak = np.max(np.abs(audio_data))
                if rms > 0:
                    self.transient_info['punch_factor'] = peak / rms
                else:
                    self.transient_info['punch_factor'] = 0.0
                    
                self.transient_info['transients_detected'] = len(attack_indices)
            
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw the professional meters panel"""
        if not hasattr(self, 'lufs_info'):
            return
            
        # Semi-transparent background
        panel_bg = pygame.Surface((width, height))
        panel_bg.set_alpha(240)
        panel_bg.fill((25, 30, 40))
        screen.blit(panel_bg, (x, y))
        
        # Border
        pygame.draw.rect(screen, (80, 90, 110), (x, y, width, height), 2)
        
        # Title
        if self.font_medium:
            title_y = y + int(10 * ui_scale)
            title = self.font_medium.render("Professional Meters", True, (180, 190, 200))
            screen.blit(title, (x + int(10 * ui_scale), title_y))
            
            # Meters
            current_y = title_y + int(40 * ui_scale)
            spacing = int(30 * ui_scale)
            
            # LUFS Meters
            meters = [
                ("M:", f"{self.lufs_info['momentary']:+5.1f} LUFS", 
                 self.get_lufs_color(self.lufs_info["momentary"])),
                ("S:", f"{self.lufs_info['short_term']:+5.1f} LUFS", 
                 self.get_lufs_color(self.lufs_info["short_term"])),
                ("I:", f"{self.lufs_info['integrated']:+5.1f} LU", 
                 self.get_lufs_color(self.lufs_info["integrated"])),
                ("LRA:", f"{self.lufs_info['range']:4.1f} LU", 
                 (150, 200, 150)),
                ("TP:", f"{self.lufs_info.get('true_peak', 0):+5.1f} dBTP", 
                 self.get_peak_color(self.lufs_info.get("true_peak", 0))),
            ]
            
            if self.font_small and self.font_meters:
                for label, value, color in meters:
                    # Label
                    label_surf = self.font_small.render(label, True, (140, 150, 160))
                    screen.blit(label_surf, (x + int(15 * ui_scale), current_y))
                    
                    # Value
                    value_surf = self.font_meters.render(value, True, color)
                    screen.blit(value_surf, (x + int(80 * ui_scale), current_y))
                    
                    current_y += spacing
                
                # Transient Analysis
                current_y += int(20 * ui_scale)
                
                attack_text = f"Attack: {self.transient_info.get('attack_time', 0):.1f}ms"
                punch_text = f"Punch: {self.transient_info.get('punch_factor', 0):.2f}"
                
                attack_surf = self.font_small.render(attack_text, True, (200, 220, 180))
                punch_surf = self.font_small.render(punch_text, True, (220, 200, 180))
                
                screen.blit(attack_surf, (x + 10, current_y))
                current_y += 20
                screen.blit(punch_surf, (x + 10, current_y))
    
    def get_lufs_color(self, lufs_value: float) -> Tuple[int, int, int]:
        """Get color for LUFS value based on broadcast standards"""
        if lufs_value > -9:  # Too loud
            return (255, 100, 100)
        elif lufs_value > -14:  # Loud
            return (255, 200, 100)
        elif lufs_value > -23:  # Normal range
            return (100, 255, 100)
        elif lufs_value > -35:  # Quiet
            return (180, 180, 255)
        else:  # Very quiet
            return (120, 120, 120)
    
    def get_peak_color(self, peak_db: float) -> Tuple[int, int, int]:
        """Get color for True Peak value"""
        if peak_db > -0.1:  # Clipping risk
            return (255, 50, 50)
        elif peak_db > -3:  # Hot
            return (255, 150, 50)
        elif peak_db > -6:  # Good level
            return (100, 255, 100)
        elif peak_db > -20:  # Normal
            return (180, 200, 220)
        else:  # Low
            return (120, 120, 120)
    
    def get_results(self) -> Dict[str, any]:
        """Get current meter values"""
        return {
            'lufs': self.lufs_info if hasattr(self, 'lufs_info') else None,
            'transient': self.transient_info
        }