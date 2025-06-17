"""
Real-time Spectrogram/Waterfall Display Panel for OMEGA-4 Audio Analyzer
Time-frequency visualization with scrolling waterfall effect
Professional visualization for frequency content over time
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import colorsys


class SpectrogramWaterfall:
    """Real-time spectrogram with scrolling waterfall visualization"""
    
    def __init__(self, sample_rate: int = 48000, fft_size: int = 2048):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        # Waterfall parameters
        self.waterfall_height = 200  # Number of time slices to keep
        self.waterfall_data = deque(maxlen=self.waterfall_height)
        
        # Frequency range and scaling
        self.min_freq = 20
        self.max_freq = 20000
        self.freq_scale = 'log'  # 'log' or 'linear'
        
        # Color mapping parameters
        self.dynamic_range = 80  # dB range for color mapping
        self.color_scheme = 'spectrum'  # 'spectrum', 'hot', 'cool', 'viridis'
        self.auto_gain = True
        self.gain_adjustment = 0.0
        
        # Peak tracking for auto-scaling
        self.peak_history = deque(maxlen=100)
        self.current_peak = 0.0
        self.current_floor = -80.0
        
        # Frequency bin mapping
        self.freq_bins = None
        self.freq_indices = None
        self._setup_frequency_mapping()
        
        # Display parameters
        self.show_frequency_grid = True
        self.show_time_grid = False
        self.grid_color = (80, 80, 100)
        
        # Performance optimization
        self.update_counter = 0
        self.update_interval = 1  # Update every frame
        
    def _setup_frequency_mapping(self):
        """Setup frequency bin mapping for display"""
        # Create frequency array for full spectrum
        nyquist = self.sample_rate / 2
        self.freq_bins = np.linspace(0, nyquist, self.fft_size // 2 + 1)
        
        # Find indices for desired frequency range
        min_idx = np.argmax(self.freq_bins >= self.min_freq)
        max_idx = np.argmax(self.freq_bins >= self.max_freq)
        if max_idx == 0:  # If max_freq is beyond Nyquist
            max_idx = len(self.freq_bins) - 1
        
        self.freq_indices = (min_idx, max_idx)
        self.display_freqs = self.freq_bins[min_idx:max_idx]
    
    def update(self, fft_data: np.ndarray, frequencies: np.ndarray):
        """Update waterfall with new FFT data"""
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return
        
        if fft_data is None or len(fft_data) == 0:
            return
        
        # Extract frequency range of interest
        min_idx, max_idx = self.freq_indices
        spectrum_slice = fft_data[min_idx:max_idx]
        
        # Convert to dB
        spectrum_db = 20 * np.log10(np.maximum(spectrum_slice, 1e-10))
        
        # Track peaks for auto-scaling
        current_max = np.max(spectrum_db)
        current_min = np.min(spectrum_db)
        
        self.peak_history.append((current_max, current_min))
        
        if self.auto_gain:
            # Calculate adaptive range
            recent_peaks = [p[0] for p in list(self.peak_history)[-20:]]
            recent_mins = [p[1] for p in list(self.peak_history)[-20:]]
            
            if recent_peaks and recent_mins:
                self.current_peak = np.percentile(recent_peaks, 95)  # 95th percentile
                self.current_floor = np.percentile(recent_mins, 5)   # 5th percentile
        
        # Normalize spectrum for color mapping
        normalized_spectrum = self._normalize_spectrum(spectrum_db)
        
        # Add to waterfall
        self.waterfall_data.append(normalized_spectrum)
    
    def _normalize_spectrum(self, spectrum_db: np.ndarray) -> np.ndarray:
        """Normalize spectrum for color mapping"""
        # Apply gain adjustment
        adjusted_spectrum = spectrum_db + self.gain_adjustment
        
        # Normalize to 0-1 range using current peak/floor
        peak_range = self.current_peak - self.current_floor
        if peak_range > 0:
            normalized = (adjusted_spectrum - self.current_floor) / peak_range
        else:
            normalized = np.zeros_like(adjusted_spectrum)
        
        # Clamp to 0-1 range
        return np.clip(normalized, 0.0, 1.0)
    
    def _frequency_to_display_y(self, freq: float, display_height: int) -> int:
        """Convert frequency to display Y coordinate"""
        if self.freq_scale == 'log':
            # Logarithmic frequency scaling
            log_min = math.log10(max(self.min_freq, 1))
            log_max = math.log10(self.max_freq)
            log_freq = math.log10(max(freq, 1))
            
            if log_max > log_min:
                normalized = (log_freq - log_min) / (log_max - log_min)
            else:
                normalized = 0.0
        else:
            # Linear frequency scaling
            normalized = (freq - self.min_freq) / (self.max_freq - self.min_freq)
        
        # Flip Y axis (high frequencies at top)
        return int(display_height * (1.0 - normalized))
    
    def _spectrum_to_color(self, intensity: float) -> Tuple[int, int, int]:
        """Convert normalized spectrum intensity to RGB color"""
        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))
        
        if self.color_scheme == 'spectrum':
            # Rainbow spectrum: blue -> green -> yellow -> red
            if intensity < 0.25:
                # Blue to cyan
                r = 0
                g = int(255 * (intensity / 0.25))
                b = 255
            elif intensity < 0.5:
                # Cyan to green
                r = 0
                g = 255
                b = int(255 * (1 - (intensity - 0.25) / 0.25))
            elif intensity < 0.75:
                # Green to yellow
                r = int(255 * ((intensity - 0.5) / 0.25))
                g = 255
                b = 0
            else:
                # Yellow to red
                r = 255
                g = int(255 * (1 - (intensity - 0.75) / 0.25))
                b = 0
                
        elif self.color_scheme == 'hot':
            # Hot colormap: black -> red -> yellow -> white
            if intensity < 0.33:
                r = int(255 * (intensity / 0.33))
                g = 0
                b = 0
            elif intensity < 0.66:
                r = 255
                g = int(255 * ((intensity - 0.33) / 0.33))
                b = 0
            else:
                r = 255
                g = 255
                b = int(255 * ((intensity - 0.66) / 0.34))
                
        elif self.color_scheme == 'cool':
            # Cool colormap: cyan to magenta
            r = int(255 * intensity)
            g = int(255 * (1 - intensity))
            b = 255
            
        elif self.color_scheme == 'viridis':
            # Viridis-like colormap
            h = 0.8 - 0.8 * intensity  # Hue from purple to yellow
            s = 0.9  # High saturation
            v = 0.2 + 0.8 * intensity  # Value from dark to bright
            
            rgb = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        
        else:
            # Grayscale fallback
            gray = int(255 * intensity)
            r, g, b = gray, gray, gray
        
        return (r, g, b)
    
    def get_frequency_labels(self, display_height: int) -> List[Tuple[float, int, str]]:
        """Get frequency labels for display"""
        labels = []
        
        if self.freq_scale == 'log':
            # Logarithmic frequency labels
            freq_labels = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        else:
            # Linear frequency labels
            freq_labels = np.linspace(self.min_freq, self.max_freq, 10)
        
        for freq in freq_labels:
            if self.min_freq <= freq <= self.max_freq:
                y_pos = self._frequency_to_display_y(freq, display_height)
                
                # Format frequency label
                if freq >= 1000:
                    label = f"{freq/1000:.0f}k" if freq % 1000 == 0 else f"{freq/1000:.1f}k"
                else:
                    label = f"{freq:.0f}"
                
                labels.append((freq, y_pos, label))
        
        return labels
    
    def set_color_scheme(self, scheme: str):
        """Set color scheme for waterfall display"""
        if scheme in ['spectrum', 'hot', 'cool', 'viridis']:
            self.color_scheme = scheme
    
    def set_frequency_range(self, min_freq: float, max_freq: float):
        """Set frequency range for display"""
        self.min_freq = max(1, min_freq)
        self.max_freq = min(self.sample_rate / 2, max_freq)
        self._setup_frequency_mapping()
    
    def set_frequency_scale(self, scale: str):
        """Set frequency scaling (log or linear)"""
        if scale in ['log', 'linear']:
            self.freq_scale = scale
    
    def adjust_gain(self, gain_db: float):
        """Adjust display gain"""
        self.gain_adjustment = gain_db
    
    def toggle_auto_gain(self):
        """Toggle automatic gain control"""
        self.auto_gain = not self.auto_gain
    
    def clear_waterfall(self):
        """Clear waterfall history"""
        self.waterfall_data.clear()


class SpectrogramWaterfallPanel:
    """OMEGA-4 Real-time Spectrogram/Waterfall Display Panel"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.waterfall = SpectrogramWaterfall(sample_rate)
        
        # Display state
        self.waterfall_info = {
            'peak_freq': 0.0,
            'peak_magnitude': 0.0,
            'frequency_range': (20, 20000),
            'dynamic_range': 80,
            'color_scheme': 'spectrum'
        }
        
        # Control state
        self.paused = False
        self.zoom_factor = 1.0
        self.scroll_offset = 0
        
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
    
    def update(self, fft_data: np.ndarray, frequencies: np.ndarray):
        """Update spectrogram with new FFT data"""
        if not self.paused and fft_data is not None and len(fft_data) > 0:
            self.waterfall.update(fft_data, frequencies)
            
            # Update display info
            if len(fft_data) > 0:
                peak_idx = np.argmax(fft_data)
                if peak_idx < len(frequencies):
                    self.waterfall_info['peak_freq'] = frequencies[peak_idx]
                    self.waterfall_info['peak_magnitude'] = fft_data[peak_idx]
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw spectrogram waterfall panel"""
        # Background
        pygame.draw.rect(screen, (10, 10, 15), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 2)
        
        # Title area
        title_height = int(40 * ui_scale)
        waterfall_area = (x + 2, y + title_height, width - 4, height - title_height - 2)
        
        # Title
        if self.font_medium:
            title_text = "Real-time Spectrogram"
            if self.paused:
                title_text += " [PAUSED]"
            
            title_color = (200, 200, 220) if not self.paused else (255, 200, 100)
            title_surf = self.font_medium.render(title_text, True, title_color)
            screen.blit(title_surf, (x + int(10 * ui_scale), y + int(8 * ui_scale)))
        
        # Draw waterfall
        self._draw_waterfall(screen, *waterfall_area, ui_scale)
        
        # Draw frequency axis
        self._draw_frequency_axis(screen, x + width - int(60 * ui_scale), 
                                 y + title_height, int(50 * ui_scale), 
                                 height - title_height, ui_scale)
        
        # Draw controls info
        if self.font_tiny:
            controls_y = y + height - int(15 * ui_scale)
            controls_text = f"Range: {self.waterfall.min_freq}-{self.waterfall.max_freq}Hz | " \
                          f"Scheme: {self.waterfall.color_scheme} | " \
                          f"Scale: {self.waterfall.freq_scale}"
            controls_surf = self.font_tiny.render(controls_text, True, (140, 140, 160))
            screen.blit(controls_surf, (x + int(10 * ui_scale), controls_y))
    
    def _draw_waterfall(self, screen: pygame.Surface, x: int, y: int, 
                       width: int, height: int, ui_scale: float):
        """Draw the main waterfall display"""
        if not self.waterfall.waterfall_data:
            return
        
        waterfall_data = list(self.waterfall.waterfall_data)
        num_time_slices = len(waterfall_data)
        
        if num_time_slices == 0:
            return
        
        # Calculate dimensions
        time_width = width - 60  # Leave space for frequency axis
        freq_height = height
        
        # Draw waterfall pixel by pixel
        if num_time_slices > 0 and len(waterfall_data[0]) > 0:
            time_step = max(1, time_width // num_time_slices)
            freq_bins = len(waterfall_data[0])
            freq_step = max(1, freq_height // freq_bins)
            
            # Create surface for efficient pixel drawing
            waterfall_surface = pygame.Surface((time_width, freq_height))
            
            for t_idx, spectrum in enumerate(waterfall_data):
                x_pos = t_idx * time_step
                
                for f_idx, intensity in enumerate(spectrum):
                    y_pos = f_idx * freq_step
                    
                    # Get color for this intensity
                    color = self.waterfall._spectrum_to_color(intensity)
                    
                    # Draw pixel block
                    pixel_rect = (x_pos, freq_height - y_pos - freq_step, 
                                 time_step, freq_step)
                    pygame.draw.rect(waterfall_surface, color, pixel_rect)
            
            # Blit waterfall surface to screen
            screen.blit(waterfall_surface, (x, y))
        
        # Draw time grid if enabled
        if self.waterfall.show_time_grid:
            self._draw_time_grid(screen, x, y, time_width, freq_height, ui_scale)
    
    def _draw_frequency_axis(self, screen: pygame.Surface, x: int, y: int, 
                           width: int, height: int, ui_scale: float):
        """Draw frequency axis with labels"""
        # Background
        pygame.draw.rect(screen, (20, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 1)
        
        # Get frequency labels
        freq_labels = self.waterfall.get_frequency_labels(height)
        
        if self.font_tiny:
            for freq, y_pos, label in freq_labels:
                # Draw tick mark
                tick_x = x + width - 10
                tick_y = y + y_pos
                pygame.draw.line(screen, (150, 150, 170), 
                               (tick_x, tick_y), (tick_x + 8, tick_y), 1)
                
                # Draw label
                label_surf = self.font_tiny.render(label, True, (180, 180, 200))
                label_rect = label_surf.get_rect(right=tick_x - 2, centery=tick_y)
                screen.blit(label_surf, label_rect)
                
                # Draw grid line if enabled
                if self.waterfall.show_frequency_grid:
                    grid_x = x - width + 60  # Extend into waterfall area
                    pygame.draw.line(screen, self.waterfall.grid_color, 
                                   (grid_x, tick_y), (tick_x, tick_y), 1)
    
    def _draw_time_grid(self, screen: pygame.Surface, x: int, y: int, 
                       width: int, height: int, ui_scale: float):
        """Draw time grid lines"""
        grid_spacing = width // 10  # 10 vertical lines
        
        for i in range(1, 10):
            grid_x = x + i * grid_spacing
            pygame.draw.line(screen, self.waterfall.grid_color, 
                           (grid_x, y), (grid_x, y + height), 1)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
    
    def cycle_color_scheme(self):
        """Cycle through color schemes"""
        schemes = ['spectrum', 'hot', 'cool', 'viridis']
        current_idx = schemes.index(self.waterfall.color_scheme)
        next_idx = (current_idx + 1) % len(schemes)
        self.waterfall.set_color_scheme(schemes[next_idx])
        self.waterfall_info['color_scheme'] = schemes[next_idx]
    
    def toggle_frequency_scale(self):
        """Toggle between log and linear frequency scale"""
        current_scale = self.waterfall.freq_scale
        new_scale = 'linear' if current_scale == 'log' else 'log'
        self.waterfall.set_frequency_scale(new_scale)
    
    def adjust_gain(self, delta_db: float):
        """Adjust display gain"""
        self.waterfall.adjust_gain(self.waterfall.gain_adjustment + delta_db)
    
    def reset_view(self):
        """Reset view parameters"""
        self.waterfall.set_frequency_range(20, 20000)
        self.waterfall.gain_adjustment = 0.0
        self.zoom_factor = 1.0
        self.scroll_offset = 0
    
    def get_results(self) -> Dict[str, Any]:
        """Get current spectrogram info"""
        return self.waterfall_info.copy()