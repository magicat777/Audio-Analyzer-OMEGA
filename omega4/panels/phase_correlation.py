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
        
        # Enhanced Goniometer with persistence/phosphor effect
        self.gonio_persistence = []  # Store multiple frames
        self.gonio_decay_rate = 0.95  # Phosphor decay
        self.gonio_max_frames = 30    # History depth
        
        # Display modes and controls
        self.display_mode = 'goniometer'  # or 'vectorscope', 'correlation_cloud'
        self.phase_rotation = 0.0  # Manual phase adjustment
        
        # Phase Statistics
        self.correlation_stats = {
            'min': 0.0,
            'max': 0.0,
            'avg': 0.0,
            'peak_hold_time': 0,
            'warnings': []
        }
        
        # Colors
        self.bg_color = (15, 18, 20)
        self.good_phase_color = (100, 255, 100)
        self.bad_phase_color = (255, 100, 100)
        self.neutral_color = (100, 100, 255)
        self.warning_color = (255, 255, 100)
        
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
            
            # Update goniometer points with persistence
            self._update_goniometer_enhanced(left_channel, right_channel)
            
            # Calculate frequency-dependent correlation
            self._calculate_band_correlations(left_channel, right_channel)
            
            # Update statistics
            self._update_statistics()
    
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
    
    def _update_goniometer_enhanced(self, left, right):
        """Update goniometer with persistence effect"""
        current_frame = []
        step = max(1, len(left) // 100)
        
        for i in range(0, len(left), step):
            if i < len(left) and i < len(right):
                mid = (left[i] + right[i]) / 2
                side = (left[i] - right[i]) / 2
                x = (mid - side) / math.sqrt(2)
                y = (mid + side) / math.sqrt(2)
                current_frame.append((x * 0.5, y * 0.5))
        
        # Add to persistence buffer
        self.gonio_persistence.append({
            'points': current_frame,
            'intensity': 1.0
        })
        
        # Decay old frames
        self.gonio_persistence = [
            {'points': frame['points'], 
             'intensity': frame['intensity'] * self.gonio_decay_rate}
            for frame in self.gonio_persistence 
            if frame['intensity'] > 0.1
        ][-self.gonio_max_frames:]
    
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
    
    def toggle_display_mode(self):
        """Switch between goniometer and vectorscope"""
        modes = ['goniometer', 'vectorscope', 'correlation_cloud']
        current_idx = modes.index(self.display_mode)
        self.display_mode = modes[(current_idx + 1) % len(modes)]
        return self.display_mode
    
    def _update_statistics(self):
        """Update correlation statistics"""
        if len(self.correlation_history) > 0:
            self.correlation_stats['min'] = min(self.correlation_history)
            self.correlation_stats['max'] = max(self.correlation_history)
            self.correlation_stats['avg'] = np.mean(self.correlation_history)
            
            # Check for phase issues
            self.correlation_stats['warnings'] = []
            if abs(self.correlation_stats['avg']) > 0.8:
                self.correlation_stats['warnings'].append("High correlation - check mono compatibility")
            if self.correlation < -0.5:
                self.correlation_stats['warnings'].append("Phase cancellation detected")
    
    def reset_statistics(self):
        """Reset correlation statistics"""
        self.correlation_stats = {
            'min': 0.0,
            'max': 0.0,
            'avg': 0.0,
            'peak_hold_time': 0,
            'warnings': []
        }
        self.correlation_history.clear()
    
    def export_phase_data(self, filename=None):
        """Export phase correlation data to CSV"""
        import csv
        from datetime import datetime
        import os
        
        if filename is None:
            filename = f"omega_phase_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(os.path.expanduser("~"), filename)
        
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Timestamp', 'Correlation', 'Width', 'Balance', 'Mono_Score'])
                
                for i, corr in enumerate(self.correlation_history):
                    mono_score = max(0, (1 - abs(corr)) * 100)
                    writer.writerow([i/60.0, corr, self.stereo_width, self.balance, mono_score])
            
            return filepath
        except Exception as e:
            print(f"Error exporting phase data: {e}")
            return None
    
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
        current_y += meter_height + meter_spacing
        
        # Mono compatibility indicator
        self._draw_mono_compatibility(screen, left_x, current_y, meter_width, meter_height)
        current_y += meter_height + meter_spacing
        
        # Goniometer/Vectorscope - positioned on right side with fixed size
        gonio_size = min(int(height * 0.35), 150)  # Fixed maximum size
        border_size = 8  # Match the border size in _draw_goniometer
        gonio_x = x + width - gonio_size - padding * 2 - border_size
        gonio_y = current_y - (meter_height + meter_spacing) * 3 - meter_spacing  # Align with meters
        
        # Draw based on display mode
        if self.display_mode == 'vectorscope':
            self._draw_vectorscope(screen, gonio_x, gonio_y, gonio_size)
        elif self.display_mode == 'correlation_cloud':
            self._draw_correlation_cloud(screen, gonio_x, gonio_y, gonio_size)
        else:  # Default goniometer
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
        
        # Phase alerts at the top
        self._draw_phase_alerts(screen, x, y + 5, width)
    
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
    
    def _draw_mono_compatibility(self, screen, x, y, width, height):
        """Draw mono compatibility indicator with traffic light system"""
        # Calculate mono compatibility score (0-100%)
        mono_score = max(0, (1 - abs(self.correlation)) * 100)
        
        # Traffic light colors
        if mono_score > 80:
            color = (0, 255, 0)  # Green - excellent
            status = "Excellent"
        elif mono_score > 60:
            color = (255, 255, 0)  # Yellow - good
            status = "Good"
        elif mono_score > 40:
            color = (255, 165, 0)  # Orange - caution
            status = "Caution"
        else:
            color = (255, 0, 0)  # Red - poor
            status = "Poor"
        
        # Draw indicator
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Fill bar
        fill_width = int(mono_score * width / 100)
        pygame.draw.rect(screen, color, (x, y, fill_width, height))
        
        # Label
        label = f"Mono Compatibility: {mono_score:.0f}% - {status}"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_phase_alerts(self, screen, x, y, width):
        """Draw real-time phase alerts"""
        if abs(self.correlation) > 0.9:
            # Red flashing alert
            if (pygame.time.get_ticks() // 500) % 2:
                alert_surf = self.fonts['small'].render(
                    "⚠ PHASE WARNING: High Correlation!", 
                    True, (255, 50, 50)
                )
                screen.blit(alert_surf, (x + width//2 - alert_surf.get_width()//2, y))
        elif self.correlation < -0.7:
            # Phase cancellation warning
            if (pygame.time.get_ticks() // 300) % 2:
                alert_surf = self.fonts['small'].render(
                    "⚠ PHASE CANCELLATION DETECTED!", 
                    True, (255, 100, 50)
                )
                screen.blit(alert_surf, (x + width//2 - alert_surf.get_width()//2, y))
    
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
        
        # Draw points with persistence/phosphor effect
        for frame in self.gonio_persistence:
            intensity = frame['intensity']
            # Adjust color based on intensity for phosphor effect
            r = int(self.good_phase_color[0] * intensity)
            g = int(self.good_phase_color[1] * intensity)
            b = int(self.good_phase_color[2] * intensity)
            color = (r, g, b)
            
            for point in frame['points']:
                px = center_x + int(point[0] * size * 0.4)
                py = center_y - int(point[1] * size * 0.4)  # Invert Y
                
                # Clip to circle
                dx = px - center_x
                dy = py - center_y
                if dx*dx + dy*dy <= (size//2 - 2)**2:
                    # Use different drawing based on intensity
                    if intensity > 0.8:
                        # Recent points - brighter, larger
                        pygame.draw.circle(screen, color, (px, py), 2)
                    elif intensity > 0.5:
                        # Medium age points
                        pygame.draw.circle(screen, color, (px, py), 1)
                    else:
                        # Old points - single pixel
                        screen.set_at((px, py), color)
        
        # Draw circle border (similar to circle of fifths)
        pygame.draw.circle(screen, (60, 60, 60), (center_x, center_y), size // 2, 2)
        
        # Label
        label = "Goniometer"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_vectorscope(self, screen, x, y, size):
        """Draw vectorscope display"""
        # Create bordered background
        border_size = 8
        total_size = size + border_size * 2
        
        # Draw outer border/background
        border_rect = pygame.Rect(x - border_size, y - border_size, total_size, total_size)
        pygame.draw.rect(screen, (30, 25, 35), border_rect)
        pygame.draw.rect(screen, (90, 70, 110), border_rect, 2)
        
        # Inner background circle
        center_x = x + size // 2
        center_y = y + size // 2
        
        # Fill background
        pygame.draw.circle(screen, (25, 20, 30), (center_x, center_y), size // 2)
        
        # Polar grid for vectorscope
        for radius_percent in [0.25, 0.5, 0.75, 1.0]:
            radius = int(size * radius_percent / 2)
            alpha = 60 if radius_percent == 1.0 else 40
            circle_color = (60, 50, 70) if radius_percent == 1.0 else (40, 35, 45)
            pygame.draw.circle(screen, circle_color, (center_x, center_y), radius, 1)
        
        # Angle lines (every 30 degrees)
        for angle in range(0, 360, 30):
            angle_rad = np.radians(angle)
            end_x = center_x + int(np.cos(angle_rad) * size // 2)
            end_y = center_y + int(np.sin(angle_rad) * size // 2)
            pygame.draw.line(screen, (40, 35, 45), (center_x, center_y), (end_x, end_y), 1)
        
        # Phase angle indicators
        if self.fonts and 'tiny' in self.fonts:
            for angle, label in [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]:
                angle_rad = np.radians(angle)
                label_x = center_x + int(np.cos(angle_rad) * (size // 2 + 15))
                label_y = center_y + int(np.sin(angle_rad) * (size // 2 + 15))
                text = self.fonts['tiny'].render(label, True, (100, 90, 110))
                text_rect = text.get_rect(center=(label_x, label_y))
                screen.blit(text, text_rect)
        
        # Draw correlation vector
        if abs(self.correlation) > 0.01:
            # Calculate vector angle and magnitude
            vector_angle = np.arccos(np.clip(abs(self.correlation), 0, 1))
            if self.correlation < 0:
                vector_angle = np.pi - vector_angle
            
            vector_magnitude = abs(self.correlation) * (size // 2 - 10)
            vector_x = center_x + int(np.cos(vector_angle) * vector_magnitude)
            vector_y = center_y - int(np.sin(vector_angle) * vector_magnitude)
            
            # Color based on correlation quality
            if abs(self.correlation) < 0.5:
                vector_color = (100, 255, 100)
            elif abs(self.correlation) < 0.8:
                vector_color = (255, 255, 100)
            else:
                vector_color = (255, 100, 100)
            
            # Draw vector with arrowhead
            pygame.draw.line(screen, vector_color, (center_x, center_y), (vector_x, vector_y), 3)
            
            # Draw arrowhead
            arrow_size = 8
            arrow_angle1 = vector_angle + np.pi * 0.75
            arrow_angle2 = vector_angle - np.pi * 0.75
            
            arrow_x1 = vector_x + int(np.cos(arrow_angle1) * arrow_size)
            arrow_y1 = vector_y - int(np.sin(arrow_angle1) * arrow_size)
            arrow_x2 = vector_x + int(np.cos(arrow_angle2) * arrow_size)
            arrow_y2 = vector_y - int(np.sin(arrow_angle2) * arrow_size)
            
            pygame.draw.polygon(screen, vector_color, 
                              [(vector_x, vector_y), (arrow_x1, arrow_y1), (arrow_x2, arrow_y2)])
        
        # Draw border
        pygame.draw.circle(screen, (80, 70, 90), (center_x, center_y), size // 2, 2)
        
        # Label
        label = f"Vectorscope ({self.correlation:.2f})"
        label_surface = self.fonts['tiny'].render(label, True, (180, 160, 200))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_correlation_cloud(self, screen, x, y, size):
        """Draw correlation cloud display"""
        # Create bordered background
        border_size = 8
        total_size = size + border_size * 2
        
        # Draw outer border/background
        border_rect = pygame.Rect(x - border_size, y - border_size, total_size, total_size)
        pygame.draw.rect(screen, (20, 30, 25), border_rect)
        pygame.draw.rect(screen, (70, 110, 90), border_rect, 2)
        
        # Inner background
        center_x = x + size // 2
        center_y = y + size // 2
        pygame.draw.circle(screen, (15, 25, 20), (center_x, center_y), size // 2)
        
        # Grid
        pygame.draw.line(screen, (40, 60, 50), 
                        (center_x, y), (center_x, y + size), 1)
        pygame.draw.line(screen, (40, 60, 50), 
                        (x, center_y), (x + size, center_y), 1)
        
        # Correlation zones
        zones = [
            (0.8, (50, 80, 60)),   # Good zone
            (0.6, (80, 80, 50)),   # Moderate zone  
            (0.3, (80, 60, 50)),   # Caution zone
        ]
        
        for threshold, color in zones:
            zone_radius = int(threshold * size // 2)
            pygame.draw.circle(screen, color, (center_x, center_y), zone_radius, 1)
        
        # Draw correlation history as cloud points
        if len(self.correlation_history) > 10:
            cloud_points = list(self.correlation_history)[-50:]  # Last 50 points
            
            for i, corr in enumerate(cloud_points):
                # Add some spread for cloud effect
                spread = (i / len(cloud_points)) * 10  # Spread over time
                angle = (i * 137.5) % 360  # Golden angle for good distribution
                angle_rad = np.radians(angle)
                
                point_x = center_x + int(corr * size * 0.4) + int(np.cos(angle_rad) * spread)
                point_y = center_y + int(np.sin(angle_rad) * spread)
                
                # Color based on age and correlation
                age_factor = i / len(cloud_points)
                if abs(corr) < 0.5:
                    point_color = (int(100 * age_factor), int(200 * age_factor), int(100 * age_factor))
                elif abs(corr) < 0.8:
                    point_color = (int(200 * age_factor), int(200 * age_factor), int(100 * age_factor))
                else:
                    point_color = (int(200 * age_factor), int(100 * age_factor), int(100 * age_factor))
                
                # Clip to circle
                dx = point_x - center_x
                dy = point_y - center_y
                if dx*dx + dy*dy <= (size//2 - 5)**2:
                    if age_factor > 0.8:
                        pygame.draw.circle(screen, point_color, (point_x, point_y), 2)
                    elif age_factor > 0.5:
                        pygame.draw.circle(screen, point_color, (point_x, point_y), 1)
                    else:
                        if 0 <= point_x < screen.get_width() and 0 <= point_y < screen.get_height():
                            screen.set_at((point_x, point_y), point_color)
        
        # Current correlation indicator
        if abs(self.correlation) > 0.01:
            current_x = center_x + int(self.correlation * size * 0.4)
            current_y = center_y
            pygame.draw.circle(screen, (255, 255, 255), (current_x, current_y), 4)
            pygame.draw.circle(screen, (100, 255, 255), (current_x, current_y), 2)
        
        # Draw border
        pygame.draw.circle(screen, (60, 100, 80), (center_x, center_y), size // 2, 2)
        
        # Label
        label = "Correlation Cloud"
        label_surface = self.fonts['tiny'].render(label, True, (150, 200, 180))
        screen.blit(label_surface, (x, y - 15))
    
    def _draw_band_correlations(self, screen, x, y, width, height):
        """Draw frequency-dependent correlation"""
        # Check display mode and call appropriate method
        if hasattr(self, 'display_mode') and self.display_mode == 'enhanced':
            self._draw_band_correlations_enhanced(screen, x, y, width, height)
        else:
            self._draw_band_correlations_standard(screen, x, y, width, height)
    
    def _draw_band_correlations_standard(self, screen, x, y, width, height):
        """Draw standard frequency-dependent correlation"""
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
    
    def _draw_band_correlations_enhanced(self, screen, x, y, width, height):
        """Draw enhanced frequency correlation with gradients"""
        # Background
        pygame.draw.rect(screen, (20, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 1)
        
        # Enhanced gradient visualization
        bar_width = max(1, width // self.freq_bands)
        
        for i in range(self.freq_bands):
            bar_x = x + i * bar_width
            correlation = self.band_correlations[i]
            
            # Enhanced color mapping with gradients
            if abs(correlation) < 0.3:
                # Green gradient for good correlation
                intensity = abs(correlation) / 0.3
                color = (int(50 + intensity * 50), int(200 + intensity * 55), int(50 + intensity * 50))
            elif abs(correlation) < 0.7:
                # Yellow gradient for moderate correlation
                intensity = (abs(correlation) - 0.3) / 0.4
                color = (int(200 + intensity * 55), int(200 + intensity * 55), int(50))
            else:
                # Red gradient for poor correlation
                intensity = min(1.0, (abs(correlation) - 0.7) / 0.3)
                color = (int(200 + intensity * 55), int(50), int(50))
            
            # Create gradient effect
            bar_height = int(abs(correlation) * height / 2)
            bar_y = y + height // 2
            
            # Draw gradient bars with multiple segments
            segments = min(bar_height, 20)
            if segments > 0:
                segment_height = max(1, bar_height // segments)
                
                for seg in range(segments):
                    seg_y = bar_y + seg * segment_height if correlation < 0 else bar_y - (seg + 1) * segment_height
                    gradient_factor = 1.0 - (seg / segments) * 0.4
                    seg_color = (int(color[0] * gradient_factor), 
                               int(color[1] * gradient_factor), 
                               int(color[2] * gradient_factor))
                    
                    pygame.draw.rect(screen, seg_color,
                                   (bar_x + 1, seg_y, bar_width - 2, segment_height))
        
        # Enhanced center line
        pygame.draw.line(screen, (100, 100, 120),
                        (x, y + height // 2), 
                        (x + width, y + height // 2), 2)
        
        # Enhanced labels with better positioning
        if self.fonts and 'tiny' in self.fonts:
            freq_labels = [(0, "20"), (2, "100"), (4, "500"), (6, "2k"), (8, "10k"), (9, "20k")]
            for idx, label in freq_labels:
                if idx < self.freq_bands:
                    label_x = x + idx * bar_width + bar_width // 2
                    freq_text = self.fonts['tiny'].render(label, True, (120, 120, 140))
                    text_rect = freq_text.get_rect(centerx=label_x, top=y + height + 2)
                    screen.blit(freq_text, text_rect)
        
        # Enhanced title
        label = "Enhanced Frequency Correlation"
        label_surface = self.fonts['tiny'].render(label, True, (180, 180, 200))
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