"""
Display Interface for OMEGA-4 Spectrum Analyzer
Phase 2: Extract display layer with clear interface boundary
"""

import pygame
import numpy as np
from typing import Dict, Any, Tuple, Optional

class SpectrumDisplay:
    """
    Handles all visualization and rendering for the spectrum analyzer.
    This is a clear interface that separates display from processing.
    """
    
    def __init__(self, screen: pygame.Surface, width: int, height: int, bars: int):
        """
        Initialize the display with pygame screen and dimensions.
        
        Args:
            screen: Pygame screen surface
            width: Display width
            height: Display height  
            bars: Number of spectrum bars
        """
        self.screen = screen
        self.width = width
        self.height = height
        self.bars = bars
        
        # Display state - will be moved from main
        self.show_help = False
        
        # Initialize display components
        self._setup_display()
        
    def _setup_display(self):
        """Initialize display components (fonts, colors, etc.)"""
        # Background color
        self.background_color = (8, 10, 15)  # Professional dark background
        
        # Initialize fonts (will be properly sized later)
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None
        
        # Grid font - temporary initialization
        self.font_grid = pygame.font.Font(None, 18)
        
        # Generate color gradient for spectrum bars
        self.colors = self._generate_professional_colors()
        
    def render_frame(self, spectrum_data: np.ndarray, analysis_results: Dict[str, Any],
                    vis_params: Optional[Dict[str, Any]] = None):
        """
        Main entry point for rendering a frame.
        
        Args:
            spectrum_data: Array of spectrum bar heights (0.0 to 1.0)
            analysis_results: Dictionary containing analyzer results
            vis_params: Visualization parameters (positions, sizes, etc.)
        """
        # Clear the screen first
        self.clear_screen()
        
        # Draw spectrum bars if we have the parameters
        if vis_params:
            self.draw_spectrum_bars(spectrum_data, vis_params)
        
    def clear_screen(self):
        """Clear the screen with background color"""
        self.screen.fill(self.background_color)
        
    def update_dimensions(self, width: int, height: int):
        """Update display dimensions (for window resize)"""
        self.width = width
        self.height = height
        
    def resize(self, width: int, height: int):
        """
        Resize the display (alias for update_dimensions).
        Called when window is resized.
        """
        self.update_dimensions(width, height)
        # Regenerate colors for new bar count if needed
        self.colors = self._generate_professional_colors()
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """
        Set fonts for the display.
        
        Args:
            fonts: Dictionary with font names as keys:
                   'large', 'medium', 'small', 'tiny', 'grid'
        """
        self.font_large = fonts.get('large')
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        self.font_tiny = fonts.get('tiny')
        self.font_grid = fonts.get('grid', self.font_grid)  # Keep default if not provided
        
    def draw_spectrum_bars(self, spectrum_data: np.ndarray, vis_params: Dict[str, Any]):
        """
        Draw spectrum bars.
        
        Args:
            spectrum_data: Bar heights (0.0 to 1.0)
            vis_params: Dictionary with visualization parameters:
                - vis_start_x: Starting X position
                - vis_start_y: Starting Y position  
                - vis_width: Width of visualization area
                - vis_height: Height of visualization area
                - center_y: Center Y position for bars
                - max_bar_height: Maximum bar height in pixels
        """
        # For now, just draw simple bars without effects
        # We'll gradually add complexity
        
        vis_start_x = vis_params.get('vis_start_x', 0)
        vis_width = vis_params.get('vis_width', self.width)
        center_y = vis_params.get('center_y', self.height // 2)
        max_bar_height = vis_params.get('max_bar_height', 200)
        spectrum_top = vis_params.get('spectrum_top', vis_params.get('vis_start_y', 0))
        spectrum_bottom = vis_params.get('spectrum_bottom', vis_params.get('vis_start_y', 0) + vis_params.get('vis_height', self.height))
        
        bar_width = vis_width / len(spectrum_data)
        
        # Simple bar drawing for now
        for i in range(len(spectrum_data)):
            if spectrum_data[i] > 0.01:
                height = int(spectrum_data[i] * max_bar_height)
                x = vis_start_x + int(i * bar_width)
                width = max(1, int(bar_width))
                
                # Use gradient color
                color = self.colors[i] if i < len(self.colors) else (0, 100, 255)
                
                # Draw upper bar (ensure it doesn't go above spectrum_top)
                upper_y = max(spectrum_top, center_y - height)
                upper_height = center_y - upper_y
                if upper_height > 0:
                    pygame.draw.rect(self.screen, color, 
                                   (x, upper_y, width, upper_height))
                
                # Draw lower bar (ensure it doesn't go below spectrum_bottom)
                lower_height = min(height, spectrum_bottom - center_y)
                if lower_height > 0:
                    lower_color = tuple(int(c * 0.75) for c in color)
                    pygame.draw.rect(self.screen, lower_color,
                                   (x, center_y, width, lower_height))
                               
    def _generate_professional_colors(self):
        """Generate professional color gradient for spectrum bars"""
        colors = []
        for i in range(self.bars):
            hue = i / self.bars

            # Smooth gradient: Purple -> Red -> Orange -> Yellow -> Green -> Cyan -> Blue
            if hue < 0.167:  # Purple to Red
                t = hue / 0.167
                r = int(150 + 105 * t)
                g = int(50 * (1 - t))
                b = int(200 * (1 - t))
            elif hue < 0.333:  # Red to Orange to Yellow
                t = (hue - 0.167) / 0.166
                r = 255
                g = int(255 * t)
                b = 0
            elif hue < 0.5:  # Yellow to Green
                t = (hue - 0.333) / 0.167
                r = int(255 * (1 - t))
                g = 255
                b = int(100 * t)
            elif hue < 0.667:  # Green to Cyan
                t = (hue - 0.5) / 0.167
                r = 0
                g = int(255 - 55 * t)
                b = int(100 + 155 * t)
            elif hue < 0.833:  # Cyan to Blue
                t = (hue - 0.667) / 0.166
                r = int(100 * t)
                g = int(200 * (1 - t))
                b = 255
            else:  # Blue to Light Blue/White
                t = (hue - 0.833) / 0.167
                r = int(100 + 155 * t)
                g = int(200 * t)
                b = 255

            colors.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))

        return colors
        
    def draw_grid_and_labels(self, vis_params: Dict[str, Any], ui_scale: float = 1.0):
        """
        Draw the dB grid and labels.
        
        Args:
            vis_params: Visualization parameters including positions
            ui_scale: UI scaling factor for fonts
        """
        # Extract parameters
        spectrum_left = vis_params.get('spectrum_left', 0)
        spectrum_right = vis_params.get('spectrum_right', self.width)
        spectrum_top = vis_params.get('spectrum_top', 100)
        spectrum_bottom = vis_params.get('spectrum_bottom', self.height - 100)
        center_y = vis_params.get('center_y', (spectrum_top + spectrum_bottom) // 2)
        
        # Amplitude grid - show fewer labels if window is cramped
        if self.height < 900:
            db_levels = [0, -20, -40, -60]  # Minimal set for small windows
        else:
            db_levels = [0, -10, -20, -30, -40, -50, -60]  # Full set for large windows
        
        spectrum_height = spectrum_bottom - spectrum_top
        
        # Draw vertical frequency grid lines first (behind horizontal lines)
        key_frequencies = [60, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for freq in key_frequencies:
            # Simple log scale mapping
            import math
            if freq >= 20:
                log_pos = (math.log10(freq) - math.log10(20)) / (math.log10(20000) - math.log10(20))
                x_pos = int(spectrum_left + log_pos * (spectrum_right - spectrum_left))
                if spectrum_left <= x_pos <= spectrum_right:
                    pygame.draw.line(
                        self.screen, (30, 35, 45),
                        (x_pos, spectrum_top), (x_pos, spectrum_bottom), 1
                    )
        
        # Draw center line (0dB) more prominently
        pygame.draw.line(
            self.screen, (60, 70, 90), 
            (spectrum_left, center_y), (spectrum_right, center_y), 2
        )
        
        # Draw 0dB label at center
        label_text = "  0"
        label = self.font_grid.render(label_text, True, (150, 160, 170))
        label_rect = label.get_rect(right=spectrum_left - 5, centery=center_y)
        if label_rect.left >= 5:
            bg_padding = int(2 * ui_scale)
            pygame.draw.rect(
                self.screen, (15, 20, 30), 
                label_rect.inflate(bg_padding * 2, bg_padding)
            )
            self.screen.blit(label, label_rect)
        
        # Draw other dB levels both above and below center
        for db in db_levels[1:]:  # Skip 0dB as we already drew it
            # Calculate distance from center (0 to 1)
            normalized_pos = abs(db) / 60.0
            
            # Upper position (above center line)
            upper_y = int(center_y - (spectrum_height / 2) * normalized_pos)
            # Lower position (below center line)
            lower_y = int(center_y + (spectrum_height / 2) * normalized_pos)
            
            # Draw grid lines
            if spectrum_top <= upper_y <= center_y:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), 
                    (spectrum_left, upper_y), (spectrum_right, upper_y), 1
                )
            if center_y <= lower_y <= spectrum_bottom:
                pygame.draw.line(
                    self.screen, (40, 45, 55, 80), 
                    (spectrum_left, lower_y), (spectrum_right, lower_y), 1
                )
            
            # Draw labels
            label_text = f"{db:3d}"
            label = self.font_grid.render(label_text, True, (120, 120, 130))
            
            # Upper label
            if spectrum_top <= upper_y <= center_y:
                label_rect = label.get_rect(right=spectrum_left - 5, centery=upper_y)
                if label_rect.left >= 5:
                    bg_padding = int(2 * ui_scale)
                    pygame.draw.rect(
                        self.screen, (15, 20, 30), 
                        label_rect.inflate(bg_padding * 2, bg_padding)
                    )
                    self.screen.blit(label, label_rect)
            
            # Lower label
            if center_y <= lower_y <= spectrum_bottom:
                label_rect = label.get_rect(right=spectrum_left - 5, centery=lower_y)
                if label_rect.left >= 5:
                    bg_padding = int(2 * ui_scale)
                    pygame.draw.rect(
                        self.screen, (15, 20, 30), 
                        label_rect.inflate(bg_padding * 2, bg_padding)
                    )
                    self.screen.blit(label, label_rect)
    def draw_frequency_scale(self, vis_params: Dict[str, Any], 
                           freq_to_x_func=None,
                           ui_scale: float = 1.0):
        """
        Draw the frequency scale with tick marks and labels.
        
        Args:
            vis_params: Visualization parameters
            freq_to_x_func: Function to convert frequency to x position
            ui_scale: UI scaling factor
        """
        # Extract parameters
        spectrum_left = vis_params.get("spectrum_left", 0)
        spectrum_right = vis_params.get("spectrum_right", self.width)
        spectrum_bottom = vis_params.get("spectrum_bottom", self.height - 100)
        scale_y = vis_params.get("scale_y", spectrum_bottom + 5)
        
        # Key frequencies
        primary_frequencies = [20, 60, 120, 250, 500, 1000, 2000, 4000, 6000, 10000, 20000]
        secondary_frequencies = [30, 40, 80, 150, 300, 700, 1500, 3000, 5000, 8000, 15000]
        
        # Draw background for scale area
        scale_bg_rect = pygame.Rect(spectrum_left - 5, scale_y - 5, 
                                   spectrum_right - spectrum_left + 10, 45)
        pygame.draw.rect(self.screen, (15, 20, 30), scale_bg_rect)
        
        # Draw scale line
        pygame.draw.line(self.screen, (120, 120, 140), 
                        (spectrum_left, scale_y), (spectrum_right, scale_y), 3)
        
        # If no frequency mapping function provided, use simple linear mapping
        if freq_to_x_func is None:
            def freq_to_x_func(freq):
                # Simple log mapping
                import math
                if freq <= 20:
                    return spectrum_left
                if freq >= 20000:
                    return spectrum_right
                log_pos = (math.log10(freq) - math.log10(20)) / (math.log10(20000) - math.log10(20))
                return spectrum_left + log_pos * (spectrum_right - spectrum_left)
        
        # Draw secondary tick marks
        for freq in secondary_frequencies:
            x_pos = int(freq_to_x_func(freq))
            if spectrum_left <= x_pos <= spectrum_right:
                tick_height = int(5 * ui_scale)
                pygame.draw.line(self.screen, (140, 140, 150), 
                               (x_pos, scale_y), (x_pos, scale_y + tick_height), 2)
        
        # Draw primary marks with labels
        last_label_x = -100
        for freq in primary_frequencies:
            x_pos = int(freq_to_x_func(freq))
            
            if x_pos < spectrum_left or x_pos > spectrum_right:
                continue
                
            # Draw tick mark
            tick_height = int(10 * ui_scale)
            if freq in [60, 250, 500, 2000, 6000]:
                pygame.draw.line(self.screen, (200, 200, 220), 
                               (x_pos, scale_y - 3), (x_pos, scale_y + tick_height + 3), 3)
            else:
                pygame.draw.line(self.screen, (170, 170, 180), 
                               (x_pos, scale_y), (x_pos, scale_y + tick_height), 2)
            
            # Skip label if too close
            min_spacing = max(35, int(40 * ui_scale))
            if abs(x_pos - last_label_x) < min_spacing:
                continue
                
            # Format frequency text
            if freq >= 10000:
                freq_text = f"{freq/1000:.0f}k"
            elif freq >= 1000:
                freq_text = f"{freq/1000:.0f}k" if freq % 1000 == 0 else f"{freq/1000:.1f}k"
            else:
                freq_text = f"{freq}"
                
            # Draw label
            label = self.font_grid.render(freq_text, True, (220, 220, 230))
            label_rect = label.get_rect(centerx=x_pos, top=scale_y + tick_height + 3)
            
            bg_padding = int(2 * ui_scale)
            pygame.draw.rect(self.screen, (20, 25, 35), 
                           label_rect.inflate(bg_padding * 2, bg_padding))
            self.screen.blit(label, label_rect)
            
            last_label_x = x_pos
            
    def draw_frequency_band_separators(self, vis_params: Dict[str, Any], 
                                      freq_to_x_func=None,
                                      show_separators: bool = True):
        """
        Draw vertical lines to separate frequency bands.
        
        Args:
            vis_params: Visualization parameters
            freq_to_x_func: Function to convert frequency to x position
            show_separators: Whether to show the separators
        """
        if not show_separators:
            return
            
        # Extract parameters
        spectrum_top = vis_params.get("spectrum_top", 290)
        spectrum_bottom = vis_params.get("spectrum_bottom", self.height - 50)
        
        # Band boundaries with labels and colors
        band_boundaries = [
            (60, "SUB", (120, 80, 140)),
            (250, "BASS", (140, 100, 120)),
            (500, "L-MID", (120, 120, 100)),
            (2000, "MID", (100, 140, 120)),
            (6000, "H-MID", (100, 120, 140)),
        ]
        
        # If no frequency mapping function provided, use simple log mapping
        if freq_to_x_func is None:
            def freq_to_x_func(freq):
                import math
                spectrum_left = vis_params.get("spectrum_left", 0)
                spectrum_right = vis_params.get("spectrum_right", self.width)
                if freq <= 20:
                    return spectrum_left
                if freq >= 20000:
                    return spectrum_right
                log_pos = (math.log10(freq) - math.log10(20)) / (math.log10(20000) - math.log10(20))
                return spectrum_left + log_pos * (spectrum_right - spectrum_left)
        
        # Draw each separator
        for freq, label, color in band_boundaries:
            x_pos = int(freq_to_x_func(freq))
            
            # Draw separator line with transparency
            pygame.draw.line(
                self.screen, color + (128,),
                (x_pos, spectrum_top), (x_pos, spectrum_bottom), 2
            )
            
            # Draw label (using tiny font if available)
            if hasattr(self, 'font_tiny') and self.font_tiny:
                label_surface = self.font_tiny.render(label, True, color)
            else:
                # Fallback to grid font
                label_surface = self.font_grid.render(label, True, color)
            
            self.screen.blit(label_surface, (x_pos + 5, spectrum_top + 5))
            
    def draw_peak_hold_indicators(self, peak_data: np.ndarray, vis_params: Dict[str, Any]):
        """
        Draw peak hold indicators for frequency bars.
        
        Args:
            peak_data: Array of peak hold values (0.0 to 1.0)
            vis_params: Visualization parameters matching spectrum bars
        """
        # Extract parameters
        vis_start_x = vis_params.get('vis_start_x', 0)
        vis_width = vis_params.get('vis_width', self.width)
        center_y = vis_params.get('center_y', self.height // 2)
        max_bar_height = vis_params.get('max_bar_height', 200)
        
        # Calculate bar width
        bar_width = vis_width / len(peak_data)
        
        # Draw peak indicators
        for i in range(len(peak_data)):
            peak = peak_data[i]
            if peak > 0.05:  # Only draw significant peaks
                x = vis_start_x + i * bar_width
                # Clamp peak height
                clamped_peak = min(peak, 1.0)
                peak_height = int(clamped_peak * max_bar_height)
                
                # Draw peak hold line above center (upper spectrum)
                upper_y = center_y - peak_height
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255, 180),
                    (int(x), upper_y),
                    (int(x + bar_width - 1), upper_y),
                    1
                )
                
                # Draw peak hold line below center (lower spectrum - mirrored)
                lower_y = center_y + peak_height
                pygame.draw.line(
                    self.screen,
                    (200, 200, 200, 140),  # Slightly dimmer for lower mirror
                    (int(x), lower_y),
                    (int(x + bar_width - 1), lower_y),
                    1
                )
                
    def draw_sub_bass_indicator(self, sub_bass_energy: float, 
                               warning_active: bool,
                               position: Dict[str, int]):
        """
        Draw sub-bass energy indicator.
        
        Args:
            sub_bass_energy: Energy level (0.0 to 1.0)
            warning_active: Whether to show warning
            position: Dict with 'x', 'y', 'width', 'height' keys
        """
        x_pos = position.get('x', 10)
        y_pos = position.get('y', self.height // 2 - 100)
        meter_width = position.get('width', 25)
        meter_height = position.get('height', 200)
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), 
                        (x_pos, y_pos, meter_width, meter_height))
        pygame.draw.rect(self.screen, (100, 100, 120), 
                        (x_pos, y_pos, meter_width, meter_height), 1)
        
        # Energy level
        level_height = int(sub_bass_energy * meter_height)
        if level_height > 0:
            # Color based on level
            if sub_bass_energy > 0.8:
                color = (255, 100, 100)  # Red warning
            elif sub_bass_energy > 0.6:
                color = (255, 200, 100)  # Orange caution
            else:
                color = (100, 255, 150)  # Green normal
                
            pygame.draw.rect(
                self.screen, color,
                (x_pos, y_pos + meter_height - level_height, 
                 meter_width, level_height)
            )
        
        # Labels
        if self.font_tiny:
            label_text = self.font_tiny.render("SUB", True, (150, 150, 170))
            label_rect = label_text.get_rect(centerx=x_pos + meter_width // 2,
                                            bottom=y_pos - 5)
            self.screen.blit(label_text, label_rect)
        
        # Warning indicator
        if warning_active and self.font_small:
            warning_text = self.font_small.render("SUB!", True, (255, 100, 100))
            warning_rect = warning_text.get_rect(centerx=x_pos + meter_width // 2,
                                               bottom=y_pos - 25)
            self.screen.blit(warning_text, warning_rect)
            
    def draw_adaptive_allocation_indicator(self, enabled: bool,
                                         content_type: str,
                                         allocation: float,
                                         position: Dict[str, int]):
        """
        Draw adaptive frequency allocation indicator.
        
        Args:
            enabled: Whether adaptive allocation is enabled
            content_type: Current content type ('music', 'speech', 'mixed', 'instrumental')
            allocation: Current allocation percentage (0.0 to 1.0)
            position: Dict with 'x', 'y', 'width', 'height' keys
        """
        x_pos = position.get('x', self.width - 170)
        y_pos = position.get('y', 80)
        width = position.get('width', 150)
        height = position.get('height', 60)
        
        # Background
        bg_color = (20, 25, 35) if enabled else (35, 25, 20)
        pygame.draw.rect(self.screen, bg_color, (x_pos, y_pos, width, height))
        pygame.draw.rect(self.screen, (80, 80, 100), (x_pos, y_pos, width, height), 1)
        
        # Status text
        if self.font_small:
            status_color = (100, 255, 150) if enabled else (150, 150, 150)
            status_text = "ADAPTIVE" if enabled else "FIXED"
            status_surface = self.font_small.render(status_text, True, status_color)
            self.screen.blit(status_surface, (x_pos + 5, y_pos + 5))
        
        # Content details
        if self.font_tiny:
            if enabled:
                # Content type color
                content_color = {
                    'music': (255, 150, 100),
                    'speech': (150, 200, 255),
                    'mixed': (255, 255, 150),
                    'instrumental': (200, 150, 255)
                }.get(content_type, (200, 200, 200))
                
                content_surface = self.font_tiny.render(
                    f"{content_type.upper()}", True, content_color
                )
                self.screen.blit(content_surface, (x_pos + 5, y_pos + 25))
                
                # Allocation percentage
                allocation_text = f"{allocation * 100:.0f}% LOW"
                allocation_surface = self.font_tiny.render(
                    allocation_text, True, (200, 200, 200)
                )
                self.screen.blit(allocation_surface, (x_pos + 5, y_pos + 40))
            else:
                # Fixed allocation
                fixed_surface = self.font_tiny.render("75% LOW", True, (150, 150, 150))
                self.screen.blit(fixed_surface, (x_pos + 5, y_pos + 25))
                
    def draw_technical_overlay(self, tech_data: Dict[str, Any], 
                             position: Dict[str, int],
                             show_overlay: bool = True):
        """
        Draw technical analysis overlay.
        
        Args:
            tech_data: Dictionary containing technical analysis data:
                - bass_energy, mid_energy, high_energy
                - spectral_tilt, tilt_description
                - crest_factor, dynamic_range
                - room_modes (list of dicts with freq, duration)
            position: Dict with 'x', 'y', 'width', 'height' keys
            show_overlay: Whether to show the overlay
        """
        if not show_overlay:
            return
            
        x_pos = position.get('x', self.width - 330)
        y_pos = position.get('y', 500)
        width = position.get('width', 320)
        height = position.get('height', 350)
        
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (x_pos, y_pos))
        
        # Border
        pygame.draw.rect(self.screen, (90, 90, 110), 
                        (x_pos, y_pos, width, height), 2)
        
        y_offset = y_pos + 20
        line_height = 25
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("Technical Analysis", True, (200, 200, 220))
            self.screen.blit(title_text, (x_pos + 20, y_offset))
            y_offset += 40
        
        # Helper function for text rendering
        def draw_text(text: str, x: int, y: int, color=(180, 180, 200)):
            if self.font_small:
                text_surface = self.font_small.render(text, True, color)
                self.screen.blit(text_surface, (x, y))
        
        # Tonal balance
        bass_energy = tech_data.get('bass_energy', 0)
        mid_energy = tech_data.get('mid_energy', 0)
        high_energy = tech_data.get('high_energy', 0)
        
        draw_text(f"Bass (20-250Hz): {bass_energy:.1%}", x_pos + 20, y_offset)
        y_offset += line_height
        draw_text(f"Mids (250-2kHz): {mid_energy:.1%}", x_pos + 20, y_offset)
        y_offset += line_height
        draw_text(f"Highs (2k-20kHz): {high_energy:.1%}", x_pos + 20, y_offset)
        y_offset += line_height * 1.5
        
        # Spectral tilt
        tilt = tech_data.get('spectral_tilt', 0)
        tilt_desc = tech_data.get('tilt_description', 'Balanced')
        draw_text(f"Spectral Tilt: {tilt:.1f}dB/oct ({tilt_desc})", x_pos + 20, y_offset)
        y_offset += line_height * 1.5
        
        # Dynamics
        crest_factor = tech_data.get('crest_factor', 0)
        dynamic_range = tech_data.get('dynamic_range', 0)
        
        draw_text(f"Crest Factor: {crest_factor:.1f}dB", x_pos + 20, y_offset)
        y_offset += line_height
        draw_text(f"Dynamic Range: {dynamic_range:.1f}dB", x_pos + 20, y_offset)
        y_offset += line_height * 1.5
        
        # Room modes
        room_modes = tech_data.get('room_modes', [])
        if room_modes:
            draw_text(f"Room Modes Detected: {len(room_modes)}", 
                     x_pos + 20, y_offset, (255, 150, 150))
            y_offset += line_height
            
            for mode in room_modes[:3]:  # Show first 3
                freq = mode.get('freq', 0)
                duration = mode.get('duration', 0)
                draw_text(f"  {freq:.1f}Hz ({duration:.1f}s)", 
                         x_pos + 40, y_offset, (200, 120, 120))
                y_offset += line_height
                
    def draw_analysis_grid(self, vis_params: Dict[str, Any],
                          freq_to_x_func=None,
                          show_grid: bool = True):
        """
        Draw vertical frequency analysis grid lines.
        
        Args:
            vis_params: Visualization parameters
            freq_to_x_func: Function to convert frequency to x position
            show_grid: Whether to show the grid
        """
        if not show_grid:
            return
            
        # Extract parameters
        spectrum_left = vis_params.get('spectrum_left', 0)
        spectrum_right = vis_params.get('spectrum_right', self.width)
        spectrum_top = vis_params.get('spectrum_top', 290)
        spectrum_bottom = vis_params.get('spectrum_bottom', self.height - 50)
        
        # If no frequency mapping function provided, use simple log mapping
        if freq_to_x_func is None:
            def freq_to_x_func(freq):
                import math
                if freq <= 20:
                    return spectrum_left
                if freq >= 20000:
                    return spectrum_right
                log_pos = (math.log10(freq) - math.log10(20)) / (math.log10(20000) - math.log10(20))
                return spectrum_left + log_pos * (spectrum_right - spectrum_left)
        
        # Musical octave frequencies for vertical lines
        octave_frequencies = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        for freq in octave_frequencies:
            x_pos = int(freq_to_x_func(freq))
            if spectrum_left <= x_pos <= spectrum_right:
                pygame.draw.line(
                    self.screen,
                    (40, 40, 50, 80),  # Semi-transparent grid color
                    (x_pos, spectrum_top),
                    (x_pos, spectrum_bottom),
                    1
                )
                
    def draw_header(self, header_data: Dict[str, Any], 
                   header_height: int = 280):
        """
        Draw the main application header.
        
        Args:
            header_data: Dictionary containing header information:
                - title, subtitle, features
                - audio_source, sample_rate, bars
                - latency, fps, quality_mode
                - gain, auto_gain, active_features
            header_height: Height of the header area
        """
        # Background already drawn by main app
        
        # Layout columns
        col1_x = 20
        col2_x = int(self.width * 0.35)
        col3_x = int(self.width * 0.65)
        
        ui_scale = header_data.get('ui_scale', 1.0)
        row_height = int(25 * ui_scale)
        y_start = 15
        
        # Column 1: Basic Info
        y = y_start
        if self.font_large:
            title = header_data.get('title', 'Professional Audio Analyzer')
            title_surf = self.font_large.render(title, True, (255, 255, 255))
            self.screen.blit(title_surf, (col1_x, y))
        y += int(40 * ui_scale)
        
        # Subtitle
        if self.font_small:
            subtitle = header_data.get('subtitle', '')
            if subtitle:
                # Check if subtitle fits in one line
                subtitle_surf = self.font_small.render(subtitle, True, (180, 200, 220))
                if subtitle_surf.get_width() > self.width - col1_x - 100:
                    # Split subtitle
                    parts = subtitle.split(' • ')
                    mid = len(parts) // 2
                    sub1 = ' • '.join(parts[:mid])
                    sub2 = ' • '.join(parts[mid:])
                    surf1 = self.font_small.render(sub1, True, (180, 200, 220))
                    surf2 = self.font_small.render(sub2, True, (180, 200, 220))
                    self.screen.blit(surf1, (col1_x, y))
                    y += row_height
                    self.screen.blit(surf2, (col1_x, y))
                else:
                    self.screen.blit(subtitle_surf, (col1_x, y))
        y += int(35 * ui_scale)
        
        # Features
        if self.font_tiny:
            features = header_data.get('features', [])
            for i, feature_line in enumerate(features[:3]):
                feat_surf = self.font_tiny.render(feature_line, True, (150, 170, 190))
                self.screen.blit(feat_surf, (col1_x, y + i * row_height))
        
        # Column 2: System Status
        if col2_x < self.width - 200 and self.font_small:
            y = y_start + int(60 * ui_scale)
            
            # Audio source
            audio_source = header_data.get('audio_source')
            if audio_source:
                source_surf = self.font_small.render(
                    f"🎤 Audio Source: {audio_source}", True, (120, 200, 120)
                )
                self.screen.blit(source_surf, (col2_x, y))
                y += row_height
            
            # Technical info
            if self.font_tiny:
                sample_rate = header_data.get('sample_rate', 48000)
                bars = header_data.get('bars', 128)
                tech_surf = self.font_tiny.render(
                    f"📊 {sample_rate}Hz • {bars} bars", True, (140, 160, 180)
                )
                self.screen.blit(tech_surf, (col2_x, y))
                y += row_height
                
                # FFT info
                fft_info = header_data.get('fft_info', 'Multi-FFT: 8192/4096/2048/1024')
                fft_surf = self.font_tiny.render(fft_info, True, (140, 160, 180))
                self.screen.blit(fft_surf, (col2_x, y))
        
        # Column 3: Performance & Status
        if col3_x < self.width - 150:
            y = y_start + int(60 * ui_scale)
            
            # Latency
            if self.font_small:
                latency = header_data.get('latency', 0)
                latency_color = (
                    (120, 200, 120) if latency < 10
                    else (200, 200, 120) if latency < 20 
                    else (200, 120, 120)
                )
                latency_surf = self.font_small.render(
                    f"⚡ Peak Latency: {latency:.1f}ms", True, latency_color
                )
                self.screen.blit(latency_surf, (col3_x, y))
                y += row_height
            
            if self.font_tiny:
                # FPS
                fps = header_data.get('fps', 60)
                fps_color = (120, 200, 120) if fps > 55 else (200, 200, 120)
                fps_surf = self.font_tiny.render(f"📈 FPS: {fps:.1f}", True, fps_color)
                self.screen.blit(fps_surf, (col3_x, y))
                y += row_height
                
                # Quality mode
                quality_mode = header_data.get('quality_mode', 'quality')
                if quality_mode == 'performance':
                    mode_surf = self.font_tiny.render(
                        "🚀 Performance Mode", True, (255, 150, 100)
                    )
                else:
                    mode_surf = self.font_tiny.render(
                        "✨ Quality Mode", True, (100, 255, 100)
                    )
                self.screen.blit(mode_surf, (col3_x, y))
                y += row_height
            
            # Gain
            if self.font_small:
                gain_db = header_data.get('gain_db', 0)
                auto_gain = header_data.get('auto_gain', False)
                gain_color = (120, 200, 120) if auto_gain else (200, 200, 120)
                gain_text = f"🎚️ Gain: {gain_db:+.1f}dB"
                if auto_gain:
                    gain_text += " (Auto)"
                gain_surf = self.font_small.render(gain_text, True, gain_color)
                self.screen.blit(gain_surf, (col3_x, y))
                y += row_height
            
            # Active features
            if self.font_tiny:
                active_features = header_data.get('active_features', [])
                if active_features:
                    feat_text = f"🔧 Active: {' • '.join(active_features[:3])}"
                    if len(active_features) > 3:
                        feat_text += f" (+{len(active_features) - 3})"
                    feat_surf = self.font_tiny.render(feat_text, True, (100, 150, 200))
                    self.screen.blit(feat_surf, (col3_x, y))
                    
    def draw_help_menu(self, help_sections: list, show_help: bool = True):
        """
        Draw keyboard shortcuts help overlay.
        
        Args:
            help_sections: List of tuples (section_title, shortcuts_list)
                          where shortcuts_list is list of (key, description)
            show_help: Whether to show the help menu
        """
        if not show_help:
            return
            
        # Semi-transparent overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(240)
        overlay.fill((20, 20, 30))
        self.screen.blit(overlay, (0, 0))
        
        # Layout settings
        col_width = 400
        start_x = 50
        start_y = 50
        line_height = 25
        section_gap = 35
        
        # Title
        if self.font_large:
            title_text = self.font_large.render(
                "OMEGA-2 Audio Analyzer - Keyboard Shortcuts", 
                True, (255, 255, 255)
            )
            title_rect = title_text.get_rect(centerx=self.width // 2, y=start_y)
            self.screen.blit(title_text, title_rect)
        
        # Calculate layout for 3 columns
        total_items = sum(len(section[1]) + 2 for section in help_sections)
        items_per_col = (total_items + 2) // 3
        
        col = 0
        x = start_x
        y = start_y + 80
        items_in_col = 0
        
        for section_title, shortcuts in help_sections:
            # Check if need new column
            if items_in_col > 0 and items_in_col + len(shortcuts) + 2 > items_per_col:
                col += 1
                x = start_x + col * col_width
                y = start_y + 80
                items_in_col = 0
            
            # Section title
            if self.font_medium:
                section_surf = self.font_medium.render(
                    section_title, True, (200, 200, 255)
                )
                self.screen.blit(section_surf, (x, y))
            y += line_height + 5
            items_in_col += 1
            
            # Shortcuts
            if self.font_small:
                for key, desc in shortcuts:
                    key_surf = self.font_small.render(
                        f"{key:>8}", True, (255, 200, 100)
                    )
                    desc_surf = self.font_small.render(
                        f" - {desc}", True, (200, 200, 200)
                    )
                    self.screen.blit(key_surf, (x, y))
                    self.screen.blit(desc_surf, (x + 80, y))
                    y += line_height
                    items_in_col += 1
            
            y += section_gap - line_height
            items_in_col += 1
        
        # Footer
        if self.font_medium:
            footer_text = self.font_medium.render(
                "Press ? or / to close help", 
                True, (150, 150, 200)
            )
            footer_rect = footer_text.get_rect(
                centerx=self.width // 2, 
                bottom=self.height - 30
            )
            self.screen.blit(footer_text, footer_rect)
            
    def draw_ab_comparison(self, reference_spectrum: np.ndarray,
                          current_spectrum: np.ndarray,
                          vis_params: Dict[str, Any]):
        """
        Draw A/B comparison overlay showing reference spectrum.
        
        Args:
            reference_spectrum: Reference spectrum data (0.0 to 1.0)
            current_spectrum: Current spectrum for comparison
            vis_params: Visualization parameters matching spectrum bars
        """
        if reference_spectrum is None or len(reference_spectrum) == 0:
            return
            
        # Extract parameters
        vis_start_x = vis_params.get('vis_start_x', 20)
        vis_width = vis_params.get('vis_width', self.width - 40)
        center_y = vis_params.get('center_y', self.height // 2)
        max_bar_height = vis_params.get('max_bar_height', 200)
        
        bar_width = vis_width / len(reference_spectrum)
        
        # Draw reference spectrum as thin lines
        for i, ref_value in enumerate(reference_spectrum):
            if ref_value > 0.01:
                x = vis_start_x + i * bar_width
                ref_height = int(ref_value * max_bar_height)
                
                # Upper reference line
                pygame.draw.line(
                    self.screen,
                    (200, 200, 200, 150),
                    (int(x), center_y - ref_height),
                    (int(x + bar_width), center_y - ref_height),
                    2
                )
                
                # Lower reference line (mirrored)
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150, 120),
                    (int(x), center_y),
                    (int(x + bar_width), center_y + ref_height),
                    2
                )
        
        # Calculate and show difference
        if len(current_spectrum) == len(reference_spectrum) and self.font_medium:
            current_rms = np.sqrt(np.mean(current_spectrum**2))
            ref_rms = np.sqrt(np.mean(reference_spectrum**2))
            
            if ref_rms > 0 and current_rms > 0:
                diff_db = 20 * np.log10(current_rms / ref_rms)
                diff_text = f"Δ: {diff_db:+.1f}dB"
                
                # Color based on difference
                if abs(diff_db) < 3:
                    diff_color = (100, 255, 100)
                elif abs(diff_db) < 6:
                    diff_color = (255, 200, 100)
                else:
                    diff_color = (255, 100, 100)
                
                # Draw difference text
                diff_surf = self.font_medium.render(diff_text, True, diff_color)
                self.screen.blit(diff_surf, (vis_start_x + vis_width - 100, center_y - max_bar_height - 30))
                
    def draw_voice_info(self, voice_data: Dict[str, Any],
                       position: Dict[str, int]):
        """
        Draw voice detection information overlay.
        
        Args:
            voice_data: Dictionary containing voice detection data:
                - has_voice, pitch, confidence
                - gender, gender_confidence
                - frequency_characteristics
            position: Dict with 'x', 'y', 'width', 'height' keys
        """
        if not voice_data:
            return
            
        x_pos = position.get('x', self.width - 330)
        y_pos = position.get('y', 500)
        width = position.get('width', 320)
        height = position.get('height', 220)
        
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        overlay.fill((20, 25, 35))
        self.screen.blit(overlay, (x_pos, y_pos))
        
        # Border
        pygame.draw.rect(self.screen, (80, 90, 110),
                        (x_pos, y_pos, width, height), 2)
        
        y_offset = y_pos + 20
        line_height = 25
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("Voice Detection", True, (200, 200, 220))
            self.screen.blit(title_text, (x_pos + 20, y_offset))
            y_offset += 40
        
        # Voice status
        if self.font_small:
            has_voice = voice_data.get('has_voice', False)
            confidence = voice_data.get('confidence', 0.0)
            
            # Status with color
            if has_voice:
                status_color = (100, 255, 100)
                status_text = "Voice: Detected"
            else:
                status_color = (200, 200, 200)
                status_text = "Voice: Not detected"
            
            status_surf = self.font_small.render(status_text, True, status_color)
            self.screen.blit(status_surf, (x_pos + 20, y_offset))
            y_offset += line_height
            
            # Confidence bar
            conf_text = f"Confidence: {confidence:.1%}"
            conf_surf = self.font_small.render(conf_text, True, (180, 180, 200))
            self.screen.blit(conf_surf, (x_pos + 20, y_offset))
            y_offset += line_height
            
            # Draw confidence bar
            bar_x = x_pos + 20
            bar_y = y_offset
            bar_width = width - 40
            bar_height = 8
            
            # Background
            pygame.draw.rect(self.screen, (40, 40, 50),
                           (bar_x, bar_y, bar_width, bar_height))
            # Fill
            fill_width = int(bar_width * confidence)
            if fill_width > 0:
                fill_color = (100, 255, 100) if confidence > 0.7 else (255, 200, 100) if confidence > 0.4 else (255, 100, 100)
                pygame.draw.rect(self.screen, fill_color,
                               (bar_x, bar_y, fill_width, bar_height))
            y_offset += bar_height + 10
            
            # Pitch info
            pitch = voice_data.get('pitch', 0)
            if pitch > 0:
                pitch_surf = self.font_small.render(
                    f"Pitch: {pitch:.0f}Hz", True, (150, 200, 255)
                )
                self.screen.blit(pitch_surf, (x_pos + 20, y_offset))
                y_offset += line_height
            
            # Gender detection
            gender = voice_data.get('gender', 'Unknown')
            gender_conf = voice_data.get('gender_confidence', 0.0)
            if gender != 'Unknown':
                gender_color = (255, 150, 200) if gender == 'Female' else (150, 200, 255)
                gender_surf = self.font_small.render(
                    f"Gender: {gender} ({gender_conf:.0%})",
                    True, gender_color
                )
                self.screen.blit(gender_surf, (x_pos + 20, y_offset))
                y_offset += line_height
            
            # Frequency characteristics
            freq_chars = voice_data.get('frequency_characteristics', {})
            if freq_chars and self.font_tiny:
                y_offset += 5
                brightness = freq_chars.get('brightness', 0)
                warmth = freq_chars.get('warmth', 0)
                
                if brightness > 0:
                    bright_surf = self.font_tiny.render(
                        f"Brightness: {brightness:.1f}",
                        True, (200, 200, 150)
                    )
                    self.screen.blit(bright_surf, (x_pos + 20, y_offset))
                    y_offset += 20
                
                if warmth > 0:
                    warmth_surf = self.font_tiny.render(
                        f"Warmth: {warmth:.1f}",
                        True, (200, 150, 150)
                    )
                    self.screen.blit(warmth_surf, (x_pos + 20, y_offset))
