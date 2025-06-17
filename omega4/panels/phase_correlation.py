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
        
        # For now, simulate stereo from mono by creating a more dynamic version
        # In real implementation, this would receive actual stereo data
        if len(audio_data) > 100:
            # Simulate stereo channels with frequency-dependent processing
            left_channel = audio_data.copy()
            right_channel = audio_data.copy()
            
            # Apply frequency-dependent delay and phase shifts for more realistic stereo
            # This creates a more dynamic stereo field that varies with the audio content
            fft = np.fft.rfft(audio_data)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
            
            # Create frequency-dependent modifications
            for i, freq in enumerate(freqs):
                if freq < 200:  # Bass frequencies - keep mostly centered
                    phase_shift = np.random.uniform(-0.1, 0.1)
                elif freq < 2000:  # Mid frequencies - moderate spread
                    phase_shift = np.random.uniform(-0.5, 0.5)
                else:  # High frequencies - wider spread
                    phase_shift = np.random.uniform(-1.0, 1.0)
                
                # Apply phase shift to create stereo difference
                fft[i] *= np.exp(1j * phase_shift)
            
            # Create right channel with modified phase
            right_channel = np.fft.irfft(fft, n=len(audio_data))
            
            # Add slight amplitude variation for balance changes
            time_factor = self.update_counter * 0.01
            balance_mod = np.sin(time_factor * 0.5) * 0.2  # Slow variation
            left_channel *= (1.0 + balance_mod * 0.5)
            right_channel *= (1.0 - balance_mod * 0.5)
            
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
        
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Use provided height or default
        if height is None:
            height = self.panel_height
        
        # Draw background
        draw_panel_background(screen, x, y, width, height)
        
        # Draw centered header
        if 'medium' in self.fonts:
            current_y = draw_panel_header(screen, "Phase Correlation", self.fonts['medium'],
                                        x, y, width, frozen=self.is_frozen)
        else:
            current_y = y + 35
        
        # Layout with increased padding
        padding = 20  # Increased padding
        meter_width = 140  # Slightly wider meters
        meter_height = 25  # Taller meters
        meter_spacing = 15  # Space between meters
        
        # Start immediately below header (2-3px spacing)
        current_y += 3
        
        # Phase correlation meters on left side
        left_x = x + padding
        
        # Correlation meter
        self._draw_correlation_meter(screen, left_x, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing
        
        # Stereo width indicator
        self._draw_width_indicator(screen, left_x, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing
        
        # Balance meter
        self._draw_balance_meter(screen, left_x, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing * 2  # Extra space before frequency correlation
        
        # Goniometer - positioned on right side with fixed size
        gonio_size = min(int(height * 0.35), 150)  # Fixed maximum size
        border_size = 8  # Match the border size in _draw_goniometer
        gonio_x = x + width - gonio_size - padding * 2 - border_size
        gonio_y = current_y - (meter_height + meter_spacing) * 3 - meter_spacing  # Align with meters
        self._draw_goniometer(screen, gonio_x, gonio_y, gonio_size)
        
        # Frequency-dependent correlation - positioned below goniometer with 10px gap
        gonio_bottom = gonio_y + gonio_size + border_size  # Bottom of goniometer including border
        freq_graph_y = gonio_bottom + 10  # 10px gap below goniometer
        freq_graph_width = width - 2 * padding
        freq_graph_height = min(50, int(height * 0.11))  # Reduced by 50%
        self._draw_band_correlations(screen, left_x, freq_graph_y, freq_graph_width, freq_graph_height)
        
        # Correlation history - positioned at bottom with proper spacing
        history_height = 80  # Slightly reduced height
        history_y = y + height - history_height - 10  # 10px from bottom
        history_width = width - 2 * padding  # Use consistent padding
        history_x = x + padding  # Use consistent padding
        
        self._draw_correlation_history(screen, history_x, history_y, 
                                      history_width, history_height)
    
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
        # Create bordered background similar to circle of fifths
        border_size = 8
        total_size = size + border_size * 2
        
        # Draw outer border/background
        border_rect = pygame.Rect(x - border_size, y - border_size, total_size, total_size)
        pygame.draw.rect(screen, (25, 30, 40), border_rect)
        pygame.draw.rect(screen, (70, 80, 100), border_rect, 2)
        
        # Inner background circle
        center_x = x + size // 2
        center_y = y + size // 2
        
        # Fill background
        pygame.draw.circle(screen, (30, 30, 30), (center_x, center_y), size // 2)
        
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
        
        # Add scale markings - concentric circles
        for scale in [0.25, 0.5, 0.75]:
            radius = int(size * scale / 2)
            pygame.draw.circle(screen, (40, 40, 40), (center_x, center_y), radius, 1)
        
        # Add L/R labels
        if self.fonts and 'tiny' in self.fonts:
            # Left label
            l_label = self.fonts['tiny'].render('L', True, (100, 100, 100))
            screen.blit(l_label, (x + 5, center_y - 8))
            
            # Right label
            r_label = self.fonts['tiny'].render('R', True, (100, 100, 100))
            screen.blit(r_label, (x + size - 15, center_y - 8))
            
            # M (Mid) and S (Side) labels for diagonal axes
            m_label = self.fonts['tiny'].render('M', True, (80, 80, 80))
            screen.blit(m_label, (center_x - 5, y + 5))
            
            s_label = self.fonts['tiny'].render('S', True, (80, 80, 80))
            screen.blit(s_label, (center_x - 5, y + size - 18))
        
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
        
        # Draw circle border (similar to circle of fifths)
        pygame.draw.circle(screen, (60, 60, 60), (center_x, center_y), size // 2, 2)
        
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
        
        # Add frequency labels
        if self.fonts and 'tiny' in self.fonts:
            # Add frequency labels for key bands
            freq_labels = [(0, "20Hz"), (2, "100Hz"), (5, "1kHz"), (7, "5kHz"), (9, "20kHz")]
            for idx, label in freq_labels:
                if idx < self.freq_bands:
                    label_x = x + idx * bar_width + bar_width // 2 - 15
                    freq_text = self.fonts['tiny'].render(label, True, (80, 80, 80))
                    screen.blit(freq_text, (label_x, y + height + 2))
            
            # Add correlation scale on the left
            scale_labels = [(0, "+1"), (height // 2, "0"), (height, "-1")]
            for y_offset, label in scale_labels:
                scale_text = self.fonts['tiny'].render(label, True, (80, 80, 80))
                screen.blit(scale_text, (x - 25, y + y_offset - 8))
        
        # Label
        label = "Frequency Correlation"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_correlation_history(self, screen, x, y, width, height):
        """Draw correlation history graph"""
        # Background with subtle color
        pygame.draw.rect(screen, (20, 25, 35), (x, y, width, height))
        # Single clean border
        pygame.draw.rect(screen, (60, 70, 90), (x, y, width, height), 1)
        
        # Title inside the box
        if self.fonts and 'tiny' in self.fonts:
            title = self.fonts['tiny'].render("Correlation History", True, (150, 170, 200))
            screen.blit(title, (x + 5, y + 3))
        
        # Graph area with more vertical space
        graph_y = y + 18  # Below title
        graph_height = height - 22  # More space for the graph
        
        # Center line (correlation = 0)
        center_y = graph_y + graph_height // 2
        pygame.draw.line(screen, (50, 50, 50),
                        (x + 5, center_y), (x + width - 5, center_y), 1)
        
        # Draw grid lines for reference (-1, -0.5, 0, 0.5, 1)
        for corr_val in [-1, -0.5, 0.5, 1]:
            grid_y = center_y - int(corr_val * graph_height / 2)
            if y + 20 < grid_y < y + height - 5:
                pygame.draw.line(screen, (40, 40, 50),
                               (x + 5, grid_y), (x + width - 5, grid_y), 1)
        
        # Draw history
        if len(self.correlation_history) > 1:
            points = []
            graph_width = width - 10
            for i, corr in enumerate(self.correlation_history):
                px = x + 5 + (i * graph_width // len(self.correlation_history))
                # Use more of the vertical space (changed from 2.2 to 2.0)
                py = center_y - int(corr * graph_height / 2.0)
                # Clamp with small margin
                py = max(graph_y + 1, min(graph_y + graph_height - 1, py))
                points.append((px, py))
            
            # Draw with gradient color based on correlation
            if len(points) > 1:
                for i in range(len(points) - 1):
                    # Get correlation value for color
                    corr = self.correlation_history[i]
                    if abs(corr) < 0.5:
                        color = self.good_phase_color
                    elif abs(corr) < 0.8:
                        color = self.neutral_color
                    else:
                        color = self.bad_phase_color
                    
                    pygame.draw.line(screen, color, points[i], points[i + 1], 2)
    
    def get_status(self):
        """Get current phase correlation status"""
        return {
            'correlation': self.correlation,
            'stereo_width': self.stereo_width,
            'balance': self.balance,
            'mono_compatible': abs(self.correlation) < 0.8
        }