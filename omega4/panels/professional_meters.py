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

        # Weighting filter coefficients
        self.k_weighting_filter = self.create_k_weighting_filter()
        self.a_weighting_filter = self.create_a_weighting_filter()
        self.c_weighting_filter = self.create_c_weighting_filter()
        
        # Current weighting mode
        self.weighting_mode = 'K'  # K, A, C, or Z (none)

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
    
    def create_a_weighting_filter(self):
        """Create A-weighting filter (40 phon curve) per IEC 61672-1"""
        # A-weighting transfer function frequencies
        f1 = 20.598997
        f2 = 107.65265
        f3 = 737.86223
        f4 = 12194.217
        
        # Normalized frequencies for digital filter
        nyquist = self.sample_rate / 2
        
        # Create analog prototype
        # H(s) = K * s^4 / ((s + 2πf1)^2 * (s + 2πf2) * (s + 2πf3) * (s + 2πf4)^2)
        # Simplified: Use cascaded filters
        
        # Two 2nd order highpass at 20.6 Hz
        hp1_b, hp1_a = scipy_signal.butter(2, f1 / nyquist, btype='high')
        
        # 1st order highpass at 107.7 Hz  
        hp2_b, hp2_a = scipy_signal.butter(1, f2 / nyquist, btype='high')
        
        # 1st order lowpass at 737.9 Hz
        lp1_b, lp1_a = scipy_signal.butter(1, f3 / nyquist, btype='low')
        
        # Two 1st order lowpass at 12194 Hz
        lp2_b, lp2_a = scipy_signal.butter(2, min(f4 / nyquist, 0.99), btype='low')
        
        return {
            'hp1': (hp1_b, hp1_a),
            'hp2': (hp2_b, hp2_a), 
            'lp1': (lp1_b, lp1_a),
            'lp2': (lp2_b, lp2_a),
            'gain': 1.0  # Normalize at 1kHz
        }
    
    def create_c_weighting_filter(self):
        """Create C-weighting filter (flat response) per IEC 61672-1"""
        # C-weighting is essentially flat from 31.5 Hz to 8 kHz
        f1 = 20.598997
        f4 = 12194.217
        
        nyquist = self.sample_rate / 2
        
        # 2nd order highpass at 20.6 Hz
        hp_b, hp_a = scipy_signal.butter(2, f1 / nyquist, btype='high')
        
        # 2nd order lowpass at 12194 Hz
        lp_b, lp_a = scipy_signal.butter(2, min(f4 / nyquist, 0.99), btype='low')
        
        return {
            'hp': (hp_b, hp_a),
            'lp': (lp_b, lp_a),
            'gain': 1.0
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
    
    def apply_a_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply A-weighting filter to audio data"""
        # Check if we have very low signal
        input_rms = np.sqrt(np.mean(audio_data**2))
        if input_rms < 1e-6:
            return np.zeros_like(audio_data)
        
        # Apply filters in cascade
        filtered = audio_data.copy()
        
        # Apply highpass filters
        filtered = scipy_signal.filtfilt(
            self.a_weighting_filter['hp1'][0],
            self.a_weighting_filter['hp1'][1],
            filtered
        )
        filtered = scipy_signal.filtfilt(
            self.a_weighting_filter['hp2'][0],
            self.a_weighting_filter['hp2'][1],
            filtered
        )
        
        # Apply lowpass filters
        filtered = scipy_signal.filtfilt(
            self.a_weighting_filter['lp1'][0],
            self.a_weighting_filter['lp1'][1],
            filtered
        )
        filtered = scipy_signal.filtfilt(
            self.a_weighting_filter['lp2'][0],
            self.a_weighting_filter['lp2'][1],
            filtered
        )
        
        # Apply gain normalization (calibrated for 1kHz = 0dB)
        filtered *= 2.5  # Empirical normalization factor
        
        return filtered
    
    def apply_c_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply C-weighting filter to audio data"""
        # Check if we have very low signal
        input_rms = np.sqrt(np.mean(audio_data**2))
        if input_rms < 1e-6:
            return np.zeros_like(audio_data)
        
        # Apply filters
        filtered = audio_data.copy()
        
        # Apply highpass
        filtered = scipy_signal.filtfilt(
            self.c_weighting_filter['hp'][0],
            self.c_weighting_filter['hp'][1],
            filtered
        )
        
        # Apply lowpass
        filtered = scipy_signal.filtfilt(
            self.c_weighting_filter['lp'][0],
            self.c_weighting_filter['lp'][1],
            filtered
        )
        
        return filtered
    
    def apply_weighting(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply selected weighting filter"""
        if self.weighting_mode == 'K':
            return self.apply_k_weighting(audio_data)
        elif self.weighting_mode == 'A':
            return self.apply_a_weighting(audio_data)
        elif self.weighting_mode == 'C':
            return self.apply_c_weighting(audio_data)
        else:  # Z-weighting (no weighting)
            return audio_data

    def calculate_lufs(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate LUFS measurements"""
        if len(audio_data) == 0:
            return self.current_lufs

        # Apply selected weighting
        weighted = self.apply_weighting(audio_data)

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

        # True peak detection with 4x oversampling
        true_peak_db = self.calculate_true_peak(audio_data)
        self.peak_history.append(true_peak_db)
        self.current_lufs["true_peak"] = max(self.peak_history)

        return self.current_lufs
    
    def calculate_true_peak(self, audio_data: np.ndarray, oversampling: int = 4) -> float:
        """Calculate true peak with oversampling per ITU-R BS.1770-4"""
        if len(audio_data) == 0:
            return -100.0
        
        # Oversample the signal
        oversampled = scipy_signal.resample(audio_data, len(audio_data) * oversampling)
        
        # Find the maximum absolute value
        peak = np.max(np.abs(oversampled))
        
        # Prevent log of zero
        if peak < 1e-10:
            return -100.0
        
        # Convert to dBTP (dB True Peak)
        return 20 * np.log10(peak)


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
        
        # Level histogram (last 10 seconds)
        self.level_history = deque(maxlen=600)  # 10 seconds at 60 FPS
        self.histogram_bins = np.linspace(-60, 0, 61)  # 1 dB resolution
        
        # Peak hold settings
        self.peak_hold_time = 1.0  # seconds
        self.peak_hold_samples = int(self.peak_hold_time * 60)  # frames
        self.peak_hold_value = -100.0
        self.peak_hold_counter = 0
        
        # True peak history for graph
        self.true_peak_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        
        # Gating mode
        self.use_gated_measurement = True
        
        # Loudness range visualization
        self.loudness_range_history = deque(maxlen=300)  # 5 seconds
        
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
        
        # Track level history for histogram
        if 'momentary' in self.lufs_info:
            self.level_history.append(self.lufs_info['momentary'])
        
        # Update loudness range history
        if 'range' in self.lufs_info:
            self.loudness_range_history.append(self.lufs_info['range'])
        
        # Update peak hold
        current_peak = self.lufs_info.get('true_peak', -100.0)
        
        # Track true peak history for graph
        self.true_peak_history.append(current_peak)
        
        if current_peak > self.peak_hold_value:
            self.peak_hold_value = current_peak
            self.peak_hold_counter = self.peak_hold_samples
        else:
            self.peak_hold_counter -= 1
            if self.peak_hold_counter <= 0:
                self.peak_hold_value = current_peak
        
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
        
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Draw background
        draw_panel_background(screen, x, y, width, height,
                            bg_color=(25, 30, 40), border_color=(80, 90, 110))
        
        # Draw centered header
        if self.font_medium:
            title_y = draw_panel_header(screen, "Professional Meters", self.font_medium,
                                      x, y, width, bg_color=(25, 30, 40),
                                      border_color=(80, 90, 110),
                                      text_color=(180, 190, 200))
        else:
            title_y = y + 35
            
        # Weighting mode and gating indicator below header
        if self.font_small:
            info_y = title_y + 5
            weighting_mode = self.metering.weighting_mode
            mode_text = f"{weighting_mode}-weighted"
            mode_surf = self.font_small.render(mode_text, True, (150, 160, 170))
            screen.blit(mode_surf, (x + 10, info_y))
            
            gating_text = "GATED" if self.use_gated_measurement else "UNGATED"
            gating_color = (100, 200, 100) if self.use_gated_measurement else (200, 200, 100)
            gating_surf = self.font_small.render(gating_text, True, gating_color)
            screen.blit(gating_surf, (x + width - 80, info_y))
            
            # Meters
            current_y = info_y + int(25 * ui_scale)
            spacing = int(25 * ui_scale)
            
            # LUFS Meters with peak hold
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
                    
                    # Peak hold indicator for True Peak
                    if label == "TP:" and self.peak_hold_value > -100:
                        hold_text = f"[{self.peak_hold_value:+5.1f}]"
                        hold_surf = self.font_small.render(hold_text, True, (200, 100, 100))
                        screen.blit(hold_surf, (x + int(200 * ui_scale), current_y))
                    
                    current_y += spacing
                
                # Position graphs right below the "GATED" text on the right side
                graphs_start_y = info_y + int(25 * ui_scale)  # Start at same level as meters
                
                # Graph dimensions and positioning
                graph_width = int(width * 0.45)  # Right side width
                graph_x = x + width - graph_width - 10  # Right side positioning
                graph_padding = 8  # 8px padding between graphs
                graph_bottom_padding = 10  # 10px padding below each graph's border
                
                # True Peak Graph (top) - made taller
                true_peak_height = 80  # Increased from 40 to 80
                if len(self.true_peak_history) > 10:
                    self._draw_true_peak_graph(screen, graph_x, graphs_start_y, graph_width, true_peak_height, ui_scale)
                current_graph_y = graphs_start_y + true_peak_height + graph_bottom_padding + graph_padding
                
                # Level Histogram (middle)
                histogram_height = 60
                if len(self.level_history) > 10:
                    self._draw_level_histogram(screen, graph_x, current_graph_y, graph_width, histogram_height, ui_scale)
                current_graph_y += histogram_height + graph_bottom_padding + graph_padding
                
                # Loudness Range Graph (bottom)
                loudness_height = 60
                if len(self.loudness_range_history) > 10:
                    self._draw_loudness_range(screen, graph_x, current_graph_y, graph_width, loudness_height, ui_scale)
                current_graph_y += loudness_height + graph_bottom_padding
                
                # Set current_y to continue after the last meter, not after graphs
                # This ensures transient analysis appears below the meter text, not below graphs
                
                # Transient Analysis
                current_y += int(10 * ui_scale)
                
                attack_text = f"Attack: {self.transient_info.get('attack_time', 0):.1f}ms"
                punch_text = f"Punch: {self.transient_info.get('punch_factor', 0):.2f}"
                
                attack_surf = self.font_small.render(attack_text, True, (200, 220, 180))
                punch_surf = self.font_small.render(punch_text, True, (220, 200, 180))
                
                screen.blit(attack_surf, (x + 10, current_y))
                current_y += 20
                screen.blit(punch_surf, (x + 10, current_y))
                current_y += 30
                
                # Mini LUFS Half-Dials - positioned 10px above panel bottom edge
                self._draw_lufs_mini_dials(screen, x, y, width, height, ui_scale)
    
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
    
    def set_weighting(self, mode: str):
        """Set weighting mode: K, A, C, or Z"""
        if mode in ['K', 'A', 'C', 'Z']:
            self.metering.weighting_mode = mode
    
    def set_peak_hold_time(self, seconds: float):
        """Set peak hold time in seconds"""
        self.peak_hold_time = max(0.0, seconds)
        self.peak_hold_samples = int(self.peak_hold_time * 60)
    
    def toggle_gating(self):
        """Toggle between gated and ungated measurements"""
        self.use_gated_measurement = not self.use_gated_measurement
    
    def reset_peak_hold(self):
        """Reset peak hold value"""
        self.peak_hold_value = -100.0
        self.peak_hold_counter = 0
    
    def get_level_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get level histogram data"""
        if not self.level_history:
            return self.histogram_bins[:-1], np.zeros(len(self.histogram_bins) - 1)
        
        # Calculate histogram
        hist, _ = np.histogram(list(self.level_history), bins=self.histogram_bins)
        # Normalize
        if hist.sum() > 0:
            hist = hist.astype(float) / hist.sum()
        
        return self.histogram_bins[:-1], hist
    
    def _draw_level_histogram(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw level histogram"""
        bins, hist = self.get_level_histogram()
        if hist.sum() == 0:
            return
        
        # Background
        pygame.draw.rect(screen, (20, 25, 35), (x, y, width, height))
        
        # Draw bars
        bar_width = width / len(hist)
        max_val = hist.max() if hist.max() > 0 else 1
        
        for i, val in enumerate(hist):
            if val > 0:
                bar_height = int((val / max_val) * height * 0.9)
                bar_x = x + int(i * bar_width)
                bar_y = y + height - bar_height
                
                # Color based on level
                level_db = bins[i]
                if level_db > -14:
                    color = (255, 100, 100)  # Red - too loud
                elif level_db > -23:
                    color = (100, 255, 100)  # Green - normal
                else:
                    color = (100, 100, 255)  # Blue - quiet
                
                pygame.draw.rect(screen, color, (bar_x, bar_y, int(bar_width - 1), bar_height))
        
        # Grid lines
        for i in range(0, 5):
            grid_y = y + int(i * height / 4)
            pygame.draw.line(screen, (40, 45, 55), (x, grid_y), (x + width, grid_y), 1)
        
        # Border
        pygame.draw.rect(screen, (60, 70, 90), (x, y, width, height), 1)
        
        # Label
        if self.font_small:
            label = self.font_small.render("Level Distribution", True, (140, 150, 160))
            screen.blit(label, (x + 2, y - 15))
    
    def _draw_true_peak_graph(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw true peak history graph"""
        if len(self.true_peak_history) < 2:
            return
        
        # Background with border
        pygame.draw.rect(screen, (25, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (70, 60, 80), (x, y, width, height), 1)
        
        # Convert deque to list for graphing
        peak_values = list(self.true_peak_history)
        
        # Set reasonable range for true peak display (-30 to +3 dBTP)
        min_peak = -30.0
        max_peak = 3.0
        peak_range = max_peak - min_peak
        
        # Draw reference lines (ensure they're contained within graph bounds)
        reference_levels = [0, -3, -6, -12, -20]
        for level in reference_levels:
            if min_peak <= level <= max_peak:
                line_y = y + height - int(((level - min_peak) / peak_range) * height)
                # Ensure line is within graph bounds
                line_y = max(y, min(y + height - 1, line_y))
                
                if level == 0:
                    # Red line for 0 dBTP (clipping threshold)
                    color = (255, 100, 100)
                    thickness = 2
                elif level == -3:
                    # Orange line for -3 dBTP (warning)
                    color = (255, 150, 100)
                    thickness = 1
                else:
                    # Gray reference lines
                    color = (60, 60, 80)
                    thickness = 1
                
                # Draw line only within the graph area
                pygame.draw.line(screen, color, (x, line_y), (x + width, line_y), thickness)
        
        # Draw peak history line
        points = []
        for i, peak in enumerate(peak_values):
            # Clamp peak value to display range
            clamped_peak = max(min_peak, min(peak, max_peak))
            
            px = x + int((i / len(peak_values)) * width)
            py = y + height - int(((clamped_peak - min_peak) / peak_range) * height)
            # Ensure point is within graph bounds
            py = max(y, min(y + height - 1, py))
            points.append((px, py))
        
        if len(points) >= 2:
            # Color the line based on current peak level
            current_peak = peak_values[-1]
            if current_peak > -0.1:
                line_color = (255, 80, 80)  # Red - clipping risk
            elif current_peak > -3:
                line_color = (255, 180, 80)  # Orange - hot
            elif current_peak > -6:
                line_color = (80, 255, 80)  # Green - good
            else:
                line_color = (180, 180, 220)  # Blue - normal
            
            pygame.draw.lines(screen, line_color, False, points, 2)
            
            # Highlight current value point
            if points:
                pygame.draw.circle(screen, line_color, points[-1], 3)
        
        # Draw peak hold line if active
        if self.peak_hold_value > min_peak and self.peak_hold_counter > 0:
            hold_y = y + height - int(((self.peak_hold_value - min_peak) / peak_range) * height)
            # Ensure hold line is within graph bounds
            hold_y = max(y, min(y + height - 1, hold_y))
            pygame.draw.line(screen, (255, 200, 200), (x, hold_y), (x + width, hold_y), 1)
        
        # Label and current value
        if self.font_small:
            label = self.font_small.render("True Peak", True, (180, 160, 200))
            screen.blit(label, (x + 2, y - 15))
            
            # Current value display
            if peak_values:
                current_peak = peak_values[-1]
                value_text = f"{current_peak:+4.1f}"
                value_color = self.get_peak_color(current_peak)
                value_surf = self.font_small.render(value_text, True, value_color)
                screen.blit(value_surf, (x + width - 40, y - 15))
    
    def _draw_loudness_range(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw loudness range history graph"""
        if len(self.loudness_range_history) < 2:
            return
        
        # Background
        pygame.draw.rect(screen, (20, 25, 35), (x, y, width, height))
        
        # Convert deque to list for graphing
        lr_values = list(self.loudness_range_history)
        max_lr = max(lr_values) if max(lr_values) > 0 else 20
        
        # Draw line graph
        points = []
        for i, lr in enumerate(lr_values):
            px = x + int((i / len(lr_values)) * width)
            py = y + height - int((lr / max_lr) * height * 0.9)
            # Ensure point is within graph bounds
            py = max(y, min(y + height - 1, py))
            points.append((px, py))
        
        if len(points) >= 2:
            pygame.draw.lines(screen, (150, 200, 150), False, points, 2)
        
        # Target lines
        # Typical broadcast target: 7-20 LU
        target_low = 7
        target_high = 20
        
        if max_lr > 0:
            low_y = y + height - int((target_low / max_lr) * height * 0.9)
            high_y = y + height - int((target_high / max_lr) * height * 0.9)
            
            # Ensure target lines are within graph bounds
            low_y = max(y, min(y + height - 1, low_y))
            high_y = max(y, min(y + height - 1, high_y))
            
            pygame.draw.line(screen, (100, 150, 100), (x, low_y), (x + width, low_y), 1)
            pygame.draw.line(screen, (150, 100, 100), (x, high_y), (x + width, high_y), 1)
        
        # Grid lines
        for i in range(1, 4):
            grid_y = y + int(i * height / 4)
            pygame.draw.line(screen, (40, 45, 55), (x, grid_y), (x + width, grid_y), 1)
        
        # Border
        pygame.draw.rect(screen, (60, 70, 90), (x, y, width, height), 1)
        
        # Label and current value
        if self.font_small:
            label = self.font_small.render("Loudness Range", True, (140, 150, 160))
            screen.blit(label, (x + 2, y - 15))
            
            if lr_values:
                current_lr = lr_values[-1]
                value_text = f"{current_lr:.1f} LU"
                value_surf = self.font_small.render(value_text, True, (200, 220, 200))
                screen.blit(value_surf, (x + width - 50, y - 15))
    
    def _draw_lufs_mini_dials(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float):
        """Draw three mini LUFS half-dials for M/S/I values"""
        if not hasattr(self, 'lufs_info'):
            return
        
        # Dial specifications - doubled in size (100% increase)
        dial_radius = 70  # Increased from 35 to 70 (100% larger)
        
        # Improved spacing and positioning
        border_padding = 8  # Padding between each border (reduced to fit)
        border_extension = 10  # Optimized to fit within panel width
        total_border_width = (dial_radius + border_extension) * 2  # Each border width
        
        # Calculate centered positioning
        total_needed_width = 3 * total_border_width + 2 * border_padding
        left_margin = (width - total_needed_width) // 2
        dial_start_x = x + left_margin + border_extension  # Start position for first dial center
        
        # Position with bottom border just below LUFS value text (reduce whitespace)
        dial_y = y + height - dial_radius - 25  # Adjust to align bottom border with LUFS text level
        
        # LUFS range for dial (-40 to -5 LUFS is typical broadcast range)
        min_lufs = -40.0
        max_lufs = -5.0
        lufs_range = max_lufs - min_lufs
        
        # Three dials: Momentary, Short-term, Integrated
        dials = [
            ("M", self.lufs_info.get('momentary', -100), "Momentary"),
            ("S", self.lufs_info.get('short_term', -100), "Short-term"), 
            ("I", self.lufs_info.get('integrated', -100), "Integrated")
        ]
        
        for i, (label, value, full_name) in enumerate(dials):
            # Calculate dial position with proper spacing and padding between borders
            border_size = dial_radius + border_extension  # Border extends beyond dial
            
            # Fixed positioning calculation: start from left margin + border size, then add spacing for each dial
            dial_x = x + left_margin + border_size + i * (total_border_width + border_padding)
            
            # Draw rectangular border around each meter (darker shade) with reduced height
            border_height = border_size * 2 - 20  # Reduce total height by 20px
            border_rect = pygame.Rect(dial_x - border_size, dial_y - border_size + 10, 
                                    border_size * 2, border_height)
            pygame.draw.rect(screen, (15, 20, 25), border_rect)  # Darker background
            pygame.draw.rect(screen, (50, 60, 70), border_rect, 1)  # Subtle border
            
            # Draw dial background (half circle)
            dial_rect = pygame.Rect(dial_x - dial_radius, dial_y - dial_radius, 
                                  dial_radius * 2, dial_radius * 2)
            
            # Draw background arc (180 degrees - half circle)
            import math
            center = (dial_x, dial_y)
            
            # Background arc
            arc_points = []
            for angle in range(180, 361):  # 180 to 360 degrees (bottom half)
                radian = math.radians(angle)
                px = center[0] + int((dial_radius - 3) * math.cos(radian))
                py = center[1] + int((dial_radius - 3) * math.sin(radian))
                arc_points.append((px, py))
            
            if len(arc_points) >= 3:
                pygame.draw.polygon(screen, (30, 35, 45), arc_points)
                pygame.draw.polygon(screen, (70, 80, 100), arc_points, 2)
            
            # Calculate needle angle based on LUFS value
            if value > -100:  # Valid LUFS value
                # Clamp value to dial range
                clamped_value = max(min_lufs, min(value, max_lufs))
                # Convert to angle (180 to 360 degrees)
                progress = (clamped_value - min_lufs) / lufs_range
                needle_angle = 180 + (progress * 180)  # 180 to 360 degrees
                
                # Draw colored arc based on LUFS value
                color = self.get_lufs_color(value)
                
                # Draw value arc
                value_points = []
                for angle in range(180, int(needle_angle) + 1):
                    radian = math.radians(angle)
                    px = center[0] + int((dial_radius - 8) * math.cos(radian))
                    py = center[1] + int((dial_radius - 8) * math.sin(radian))
                    value_points.append((px, py))
                
                if len(value_points) >= 2:
                    pygame.draw.lines(screen, color, False, value_points, 4)
                
                # Draw needle
                needle_radian = math.radians(needle_angle)
                needle_end_x = center[0] + int((dial_radius - 5) * math.cos(needle_radian))
                needle_end_y = center[1] + int((dial_radius - 5) * math.sin(needle_radian))
                pygame.draw.line(screen, (255, 255, 255), center, 
                               (needle_end_x, needle_end_y), 2)
                
                # Draw center dot
                pygame.draw.circle(screen, (200, 200, 200), center, 3)
            
            # Draw reference marks
            for ref_angle in [180, 225, 270, 315, 360]:  # 5 marks across the arc
                mark_radian = math.radians(ref_angle)
                mark_start_x = center[0] + int((dial_radius - 6) * math.cos(mark_radian))
                mark_start_y = center[1] + int((dial_radius - 6) * math.sin(mark_radian))
                mark_end_x = center[0] + int((dial_radius - 2) * math.cos(mark_radian))
                mark_end_y = center[1] + int((dial_radius - 2) * math.sin(mark_radian))
                pygame.draw.line(screen, (100, 110, 130), 
                               (mark_start_x, mark_start_y), (mark_end_x, mark_end_y), 1)
            
            # Label above dial
            if self.font_small:
                label_surf = self.font_small.render(label, True, (180, 190, 210))
                label_rect = label_surf.get_rect(centerx=dial_x, bottom=dial_y - dial_radius - 5)
                screen.blit(label_surf, label_rect)
            
            # Value below dial
            if self.font_small and value > -100:
                value_text = f"{value:+.1f}"
                value_surf = self.font_small.render(value_text, True, self.get_lufs_color(value))
                value_rect = value_surf.get_rect(centerx=dial_x, top=dial_y + 8)
                screen.blit(value_surf, value_rect)