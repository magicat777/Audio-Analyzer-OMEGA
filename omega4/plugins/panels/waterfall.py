"""
Waterfall Display Panel Plugin for OMEGA-4
3D-style frequency waterfall visualization
"""

import numpy as np
import pygame
from collections import deque
from typing import Dict, Any, List, Tuple

from omega4.plugins.base import PanelPlugin, PluginMetadata, PluginType


class WaterfallPanel(PanelPlugin):
    """3D waterfall frequency display panel"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Waterfall",
            version="1.0.0",
            author="OMEGA-4 Team",
            description="3D waterfall display showing frequency spectrum over time",
            plugin_type=PluginType.PANEL,
            config_schema={
                "history_size": {"type": "int", "default": 50, "min": 20, "max": 100},
                "perspective": {"type": "float", "default": 0.7, "min": 0.5, "max": 0.9},
                "color_scheme": {"type": "str", "default": "heat", "options": ["heat", "cool", "rainbow", "mono"]},
                "peak_hold": {"type": "bool", "default": True},
                "grid_lines": {"type": "bool", "default": True}
            }
        )
    
    def initialize(self, config: Dict = None) -> bool:
        """Initialize the waterfall panel"""
        if not super().initialize(config):
            return False
            
        # Configuration
        self.history_size = self._config.get("history_size", 50)
        self.perspective = self._config.get("perspective", 0.7)
        self.color_scheme = self._config.get("color_scheme", "heat")
        self.peak_hold = self._config.get("peak_hold", True)
        self.grid_lines = self._config.get("grid_lines", True)
        
        # Data storage
        self.spectrum_history = deque(maxlen=self.history_size)
        self.peak_values = None
        self.peak_decay = 0.95
        
        # Rendering
        self.surface = None
        self.needs_redraw = True
        
        return True
    
    def update(self, data: Dict[str, Any]):
        """Update waterfall with new spectrum data"""
        if not self._enabled:
            return
            
        band_values = data.get("band_values")
        if band_values is None:
            return
            
        # Add to history
        self.spectrum_history.append(band_values.copy())
        
        # Update peak hold
        if self.peak_hold:
            if self.peak_values is None:
                self.peak_values = band_values.copy()
            else:
                # Update peaks
                self.peak_values = np.maximum(self.peak_values * self.peak_decay, band_values)
                
        self.needs_redraw = True
    
    def draw(self, screen, x: int, y: int, width: int, height: int):
        """Draw the waterfall display"""
        if not self._enabled or not self._visible:
            return
            
        # Create or resize surface if needed
        if self.surface is None or self.surface.get_size() != (width, height):
            self.surface = pygame.Surface((width, height))
            self.needs_redraw = True
            
        if self.needs_redraw and len(self.spectrum_history) > 0:
            self._redraw_waterfall()
            self.needs_redraw = False
            
        # Draw to screen
        screen.blit(self.surface, (x, y))
        
        # Draw border
        pygame.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)
        
        # Draw title
        if "small" in self._font_cache:
            font = self._font_cache["small"]
            title = font.render("Waterfall Display", True, (255, 255, 255))
            screen.blit(title, (x + 5, y + 5))
    
    def _redraw_waterfall(self):
        """Redraw the waterfall surface"""
        self.surface.fill((0, 0, 0))
        
        width, height = self.surface.get_size()
        
        # Calculate 3D projection parameters
        base_y = height * 0.8
        top_y = height * 0.2
        depth_scale = (base_y - top_y) / len(self.spectrum_history)
        
        # Draw from back to front
        for i, spectrum in enumerate(self.spectrum_history):
            # Calculate Y position and scaling
            y_pos = base_y - i * depth_scale
            scale = 1.0 - (i / len(self.spectrum_history)) * (1.0 - self.perspective)
            
            # Draw this spectrum line
            self._draw_spectrum_line(spectrum, y_pos, scale, width, i)
            
        # Draw peak hold line if enabled
        if self.peak_hold and self.peak_values is not None:
            self._draw_peak_line(self.peak_values, base_y, width)
            
        # Draw grid if enabled
        if self.grid_lines:
            self._draw_grid(width, height, base_y, top_y)
    
    def _draw_spectrum_line(self, spectrum: np.ndarray, y_pos: float, scale: float, 
                           width: int, depth_index: int):
        """Draw a single spectrum line with 3D effect"""
        num_bands = len(spectrum)
        band_width = width / num_bands * scale
        x_offset = (width - width * scale) / 2
        
        # Get color based on depth
        base_color = self._get_color_for_depth(depth_index)
        
        points = []
        for i, value in enumerate(spectrum):
            x = x_offset + i * band_width
            
            # Scale height based on perspective
            bar_height = value * 100 * scale
            
            # Create points for filled polygon
            if i == 0:
                points.append((x, y_pos))
            points.append((x, y_pos - bar_height))
            if i == num_bands - 1:
                points.append((x + band_width, y_pos - bar_height))
                points.append((x + band_width, y_pos))
                
        # Draw filled spectrum
        if len(points) > 2:
            # Adjust color brightness based on average amplitude
            avg_amplitude = np.mean(spectrum)
            color = self._adjust_brightness(base_color, avg_amplitude)
            
            pygame.draw.polygon(self.surface, color, points)
            
            # Draw outline for better definition
            outline_color = self._adjust_brightness(base_color, avg_amplitude * 1.5)
            pygame.draw.lines(self.surface, outline_color, False, 
                            points[1:-1], 1)
    
    def _draw_peak_line(self, peaks: np.ndarray, y_pos: float, width: int):
        """Draw peak hold line"""
        num_bands = len(peaks)
        band_width = width / num_bands
        
        points = []
        for i, value in enumerate(peaks):
            x = i * band_width + band_width / 2
            y = y_pos - value * 100
            points.append((x, y))
            
        if len(points) > 1:
            pygame.draw.lines(self.surface, (255, 255, 0), False, points, 2)
    
    def _draw_grid(self, width: int, height: int, base_y: float, top_y: float):
        """Draw perspective grid lines"""
        grid_color = (40, 40, 40)
        
        # Horizontal lines (time)
        num_h_lines = 5
        for i in range(num_h_lines):
            y = base_y - (base_y - top_y) * i / (num_h_lines - 1)
            scale = 1.0 - i / (num_h_lines - 1) * (1.0 - self.perspective)
            x_offset = (width - width * scale) / 2
            
            pygame.draw.line(self.surface, grid_color,
                           (x_offset, y), (width - x_offset, y), 1)
            
        # Vertical lines (frequency)
        num_v_lines = 10
        for i in range(num_v_lines):
            x_base = width * i / (num_v_lines - 1)
            
            # Calculate perspective-adjusted positions
            x_top = width / 2 + (x_base - width / 2) * self.perspective
            
            pygame.draw.line(self.surface, grid_color,
                           (x_base, base_y), (x_top, top_y), 1)
    
    def _get_color_for_depth(self, depth_index: int) -> Tuple[int, int, int]:
        """Get color based on depth and color scheme"""
        t = depth_index / max(1, len(self.spectrum_history) - 1)
        
        if self.color_scheme == "heat":
            # Red to yellow gradient
            r = 255
            g = int(255 * (1 - t))
            b = 0
        elif self.color_scheme == "cool":
            # Blue to cyan gradient
            r = 0
            g = int(255 * (1 - t))
            b = 255
        elif self.color_scheme == "rainbow":
            # Full spectrum
            hue = t * 270  # 0 to 270 degrees (red to violet)
            r, g, b = self._hsv_to_rgb(hue, 1.0, 1.0)
        else:  # mono
            # Grayscale
            intensity = int(255 * (1 - t * 0.5))
            r = g = b = intensity
            
        return (r, g, b)
    
    def _adjust_brightness(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Adjust color brightness"""
        r, g, b = color
        factor = max(0, min(2, factor))
        
        return (
            int(min(255, r * factor)),
            int(min(255, g * factor)),
            int(min(255, b * factor))
        )
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB"""
        h = h / 60.0
        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
    
    def on_config_change(self):
        """Handle configuration changes"""
        # Update settings
        self.history_size = self._config.get("history_size", 50)
        self.perspective = self._config.get("perspective", 0.7)
        self.color_scheme = self._config.get("color_scheme", "heat")
        self.peak_hold = self._config.get("peak_hold", True)
        self.grid_lines = self._config.get("grid_lines", True)
        
        # Resize history if needed
        if len(self.spectrum_history) > self.history_size:
            new_history = deque(maxlen=self.history_size)
            for _ in range(self.history_size):
                new_history.append(self.spectrum_history.popleft())
            self.spectrum_history = new_history
            
        self.needs_redraw = True
    
    def handle_event(self, event) -> bool:
        """Handle user events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                # Toggle peak hold
                self.peak_hold = not self.peak_hold
                self._config["peak_hold"] = self.peak_hold
                if not self.peak_hold:
                    self.peak_values = None
                return True
            elif event.key == pygame.K_g:
                # Toggle grid
                self.grid_lines = not self.grid_lines
                self._config["grid_lines"] = self.grid_lines
                self.needs_redraw = True
                return True
                
        return False