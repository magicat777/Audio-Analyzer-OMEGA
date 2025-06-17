"""
Phase Correlation Panel - Stereo imaging and phase analysis
"""

import numpy as np
import pygame
from collections import deque
import math


class PhaseCorrelationPanel:
    """Panel for stereo phase correlation and imaging analysis"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.panel_height = 200
        self.is_frozen = False
        
        # Phase correlation data
        self.correlation = 0.0  # -1 to +1
        self.correlation_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        
        # Stereo width analysis
        self.stereo_width = 0.0  # 0 to 1
        self.balance = 0.0  # -1 (left) to +1 (right)
        
        # Frequency-dependent phase
        self.freq_bands = 10
        self.band_correlations = np.zeros(self.freq_bands)
        self.band_frequencies = []
        
        # Goniometer data
        self.gonio_points = deque(maxlen=1000)
        
        # Colors
        self.bg_color = (15, 18, 20)
        self.good_phase_color = (100, 255, 100)
        self.bad_phase_color = (255, 100, 100)
        self.neutral_color = (100, 100, 255)
        
        # Fonts (will be set by main app)
        self.fonts = None
        
        # Update control
        self.update_counter = 0
        self.update_interval = 1  # Update every frame for smooth visualization
        
        # Initialize frequency bands
        self._init_frequency_bands()
    
    def _init_frequency_bands(self):
        """Initialize frequency band boundaries"""
        # Logarithmic frequency bands from 20Hz to 20kHz
        min_freq = 20
        max_freq = min(20000, self.sample_rate / 2)
        self.band_frequencies = np.logspace(
            np.log10(min_freq), 
            np.log10(max_freq), 
            self.freq_bands + 1
        )
    
    def set_fonts(self, fonts):
        """Set fonts for rendering"""
        self.fonts = fonts
    
    def update(self, audio_data, spectrum=None):
        """Update phase correlation analysis"""
        if self.is_frozen:
            return
        
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return
        
        # For now, simulate stereo from mono by creating a delayed version
        # In real implementation, this would receive actual stereo data
        if len(audio_data) > 100:
            # Simulate stereo channels
            left_channel = audio_data
            # Create slightly delayed and attenuated right channel
            delay_samples = 10
            right_channel = np.zeros_like(audio_data)
            if len(audio_data) > delay_samples:
                right_channel[delay_samples:] = audio_data[:-delay_samples] * 0.9
                right_channel[:delay_samples] = audio_data[-delay_samples:] * 0.9
            
            # Calculate overall correlation
            self.correlation = self._calculate_correlation(left_channel, right_channel)
            self.correlation_history.append(self.correlation)
            
            # Calculate stereo width
            self.stereo_width = 1.0 - abs(self.correlation)
            
            # Calculate balance
            left_energy = np.sqrt(np.mean(left_channel**2))
            right_energy = np.sqrt(np.mean(right_channel**2))
            total_energy = left_energy + right_energy
            if total_energy > 0:
                self.balance = (right_energy - left_energy) / total_energy
            
            # Update goniometer points
            self._update_goniometer(left_channel, right_channel)
            
            # Calculate frequency-dependent correlation
            self._calculate_band_correlations(left_channel, right_channel)
    
    def _calculate_correlation(self, left, right):
        """Calculate correlation coefficient between channels"""
        if len(left) != len(right) or len(left) == 0:
            return 0.0
        
        # Normalize signals
        left_norm = left - np.mean(left)
        right_norm = right - np.mean(right)
        
        # Calculate correlation
        left_std = np.std(left_norm)
        right_std = np.std(right_norm)
        
        if left_std > 0 and right_std > 0:
            correlation = np.mean(left_norm * right_norm) / (left_std * right_std)
            return np.clip(correlation, -1.0, 1.0)
        
        return 0.0
    
    def _update_goniometer(self, left, right):
        """Update goniometer display points"""
        # Downsample for display
        step = max(1, len(left) // 100)
        
        for i in range(0, len(left), step):
            if i < len(left) and i < len(right):
                # Convert to mid/side
                mid = (left[i] + right[i]) / 2
                side = (left[i] - right[i]) / 2
                
                # Rotate 45 degrees for traditional goniometer display
                x = (mid - side) / math.sqrt(2)
                y = (mid + side) / math.sqrt(2)
                
                # Scale and add to points
                scale = 0.5
                self.gonio_points.append((x * scale, y * scale))
    
    def _calculate_band_correlations(self, left, right):
        """Calculate correlation for each frequency band"""
        # Compute FFTs
        left_fft = np.fft.rfft(left * np.hanning(len(left)))
        right_fft = np.fft.rfft(right * np.hanning(len(right)))
        
        freqs = np.fft.rfftfreq(len(left), 1/self.sample_rate)
        
        # Calculate correlation for each band
        for i in range(self.freq_bands):
            low_freq = self.band_frequencies[i]
            high_freq = self.band_frequencies[i + 1]
            
            # Find frequency bins for this band
            band_mask = (freqs >= low_freq) & (freqs < high_freq)
            
            if np.any(band_mask):
                left_band = left_fft[band_mask]
                right_band = right_fft[band_mask]
                
                # Calculate magnitude correlation for this band
                left_mag = np.abs(left_band)
                right_mag = np.abs(right_band)
                
                if len(left_mag) > 0:
                    self.band_correlations[i] = self._calculate_correlation(
                        left_mag, right_mag
                    )
    
    def draw(self, screen, x, y, width, height=None, panel_color=None):
        """Draw the phase correlation panel"""
        if not self.fonts:
            return
        
        # Use provided height or default
        if height is None:
            height = self.panel_height
        
        # Panel background
        if panel_color is None:
            panel_color = self.bg_color
        
        panel_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, panel_color, panel_rect)
        pygame.draw.rect(screen, (100, 100, 100), panel_rect, 1)
        
        # Title with more padding
        title = "Phase Correlation"
        if self.is_frozen:
            title += " [FROZEN]"
        title_surface = self.fonts['small'].render(title, True, (255, 255, 255))
        screen.blit(title_surface, (x + 15, y + 10))
        
        # Layout with increased padding
        padding = 20  # Increased padding
        meter_width = 140  # Slightly wider meters
        meter_height = 25  # Taller meters
        meter_spacing = 15  # Space between meters
        
        # Calculate positions with better spacing
        current_y = y + 40  # Start below title
        
        # Correlation meter
        self._draw_correlation_meter(screen, x + padding, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing
        
        # Stereo width indicator
        self._draw_width_indicator(screen, x + padding, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing
        
        # Balance meter
        self._draw_balance_meter(screen, x + padding, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing * 2  # Extra space before graphs
        
        # Goniometer - scale based on available height
        gonio_size = min(int(height * 0.35), width // 3)
        gonio_x = x + width - gonio_size - padding
        gonio_y = y + 40
        self._draw_goniometer(screen, gonio_x, gonio_y, gonio_size)
        
        # Frequency-dependent correlation
        graph_x = x + meter_width + padding * 2
        graph_y = y + 40
        graph_width = gonio_x - graph_x - padding
        graph_height = min(120, int(height * 0.3))
        self._draw_band_correlations(screen, graph_x, graph_y, graph_width, graph_height)
        
        # Add extra padding below goniometer
        history_y = max(current_y, gonio_y + gonio_size + padding * 2)  # Double padding
        
        # History graph - use remaining space but leave padding at bottom
        available_height = (y + height) - history_y - padding * 2  # Double padding at bottom
        history_height = max(80, available_height)  # Use all available space
        
        if history_height > 40:  # Only draw if there's enough space
            self._draw_correlation_history(screen, x + padding, history_y, 
                                          width - 2 * padding, history_height)
    
    def _draw_correlation_meter(self, screen, x, y, width, height):
        """Draw correlation meter"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Center line
        center_x = x + width // 2
        pygame.draw.line(screen, (80, 80, 80), 
                        (center_x, y), (center_x, y + height), 1)
        
        # Correlation indicator
        corr_x = x + int((self.correlation + 1) * width / 2)
        
        # Color based on correlation
        if abs(self.correlation) < 0.5:
            color = self.good_phase_color
        elif abs(self.correlation) < 0.8:
            color = self.neutral_color
        else:
            color = self.bad_phase_color
        
        pygame.draw.rect(screen, color, 
                        (corr_x - 2, y, 4, height))
        
        # Labels
        label = f"Correlation: {self.correlation:.2f}"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_width_indicator(self, screen, x, y, width, height):
        """Draw stereo width indicator"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Width bar
        width_pixels = int(self.stereo_width * width)
        pygame.draw.rect(screen, self.neutral_color, 
                        (x, y, width_pixels, height))
        
        # Label
        label = f"Width: {self.stereo_width * 100:.0f}%"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_balance_meter(self, screen, x, y, width, height):
        """Draw balance meter"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Center line
        center_x = x + width // 2
        pygame.draw.line(screen, (80, 80, 80), 
                        (center_x, y), (center_x, y + height), 1)
        
        # Balance indicator
        balance_x = x + int((self.balance + 1) * width / 2)
        pygame.draw.rect(screen, (200, 200, 100), 
                        (balance_x - 2, y, 4, height))
        
        # Labels
        if abs(self.balance) < 0.1:
            balance_text = "Center"
        elif self.balance < 0:
            balance_text = f"L {abs(self.balance * 100):.0f}%"
        else:
            balance_text = f"R {self.balance * 100:.0f}%"
        
        label = f"Balance: {balance_text}"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_goniometer(self, screen, x, y, size):
        """Draw goniometer (Lissajous) display"""
        # Background
        center_x = x + size // 2
        center_y = y + size // 2
        
        pygame.draw.circle(screen, (30, 30, 30), (center_x, center_y), size // 2)
        pygame.draw.circle(screen, (60, 60, 60), (center_x, center_y), size // 2, 1)
        
        # Grid lines
        pygame.draw.line(screen, (50, 50, 50), 
                        (center_x, y), (center_x, y + size), 1)
        pygame.draw.line(screen, (50, 50, 50), 
                        (x, center_y), (x + size, center_y), 1)
        
        # Draw diagonal lines (45 degree)
        diag_offset = int(size * 0.35)
        pygame.draw.line(screen, (40, 40, 40),
                        (center_x - diag_offset, center_y - diag_offset),
                        (center_x + diag_offset, center_y + diag_offset), 1)
        pygame.draw.line(screen, (40, 40, 40),
                        (center_x - diag_offset, center_y + diag_offset),
                        (center_x + diag_offset, center_y - diag_offset), 1)
        
        # Draw points
        if len(self.gonio_points) > 1:
            for point in self.gonio_points:
                px = center_x + int(point[0] * size * 0.4)
                py = center_y - int(point[1] * size * 0.4)  # Invert Y
                
                # Clip to circle
                dx = px - center_x
                dy = py - center_y
                if dx*dx + dy*dy <= (size//2 - 2)**2:
                    screen.set_at((px, py), self.good_phase_color)
        
        # Label
        label = "Goniometer"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_band_correlations(self, screen, x, y, width, height):
        """Draw frequency-dependent correlation"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Draw bars for each frequency band
        bar_width = width // self.freq_bands
        
        for i in range(self.freq_bands):
            bar_x = x + i * bar_width
            correlation = self.band_correlations[i]
            
            # Bar height (centered at middle)
            bar_height = int(abs(correlation) * height / 2)
            bar_y = y + height // 2
            
            # Color based on correlation
            if abs(correlation) < 0.5:
                color = self.good_phase_color
            elif abs(correlation) < 0.8:
                color = self.neutral_color
            else:
                color = self.bad_phase_color
            
            # Draw bar
            if correlation >= 0:
                pygame.draw.rect(screen, color,
                               (bar_x + 2, bar_y - bar_height, 
                                bar_width - 4, bar_height))
            else:
                pygame.draw.rect(screen, color,
                               (bar_x + 2, bar_y, 
                                bar_width - 4, bar_height))
        
        # Center line
        pygame.draw.line(screen, (80, 80, 80),
                        (x, y + height // 2), 
                        (x + width, y + height // 2), 1)
        
        # Label
        label = "Frequency Correlation"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_correlation_history(self, screen, x, y, width, height):
        """Draw correlation history graph"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Center line (correlation = 0)
        center_y = y + height // 2
        pygame.draw.line(screen, (50, 50, 50),
                        (x, center_y), (x + width, center_y), 1)
        
        # Draw history
        if len(self.correlation_history) > 1:
            points = []
            for i, corr in enumerate(self.correlation_history):
                px = x + (i * width // len(self.correlation_history))
                py = center_y - int(corr * height / 2)
                points.append((px, py))
            
            pygame.draw.lines(screen, self.neutral_color, False, points, 2)
        
        # Label
        label = "Correlation History"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def get_status(self):
        """Get current phase correlation status"""
        return {
            'correlation': self.correlation,
            'stereo_width': self.stereo_width,
            'balance': self.balance,
            'mono_compatible': abs(self.correlation) < 0.8
        }