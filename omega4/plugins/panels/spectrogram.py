"""
Spectrogram Panel Plugin for OMEGA-4
Example panel plugin showing time-frequency visualization
"""

import numpy as np
import pygame
from collections import deque
from typing import Dict, Any

from omega4.plugins.base import PanelPlugin, PluginMetadata, PluginType


class SpectrogramPanel(PanelPlugin):
    """Real-time spectrogram visualization panel"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Spectrogram",
            version="1.0.0",
            author="OMEGA-4 Team",
            description="Real-time spectrogram showing time-frequency representation",
            plugin_type=PluginType.PANEL,
            config_schema={
                "history_size": {"type": "int", "default": 200, "min": 50, "max": 500},
                "color_map": {"type": "str", "default": "viridis", "options": ["viridis", "plasma", "inferno", "magma"]},
                "db_range": {"type": "float", "default": 80.0, "min": 40.0, "max": 120.0},
                "interpolation": {"type": "bool", "default": True}
            }
        )
    
    def initialize(self, config: Dict = None) -> bool:
        """Initialize the spectrogram panel"""
        if not super().initialize(config):
            return False
            
        # Configuration
        self.history_size = self._config.get("history_size", 200)
        self.color_map = self._config.get("color_map", "viridis")
        self.db_range = self._config.get("db_range", 80.0)
        self.interpolation = self._config.get("interpolation", True)
        
        # Data storage
        self.spectrogram_data = deque(maxlen=self.history_size)
        self.frequency_bins = None
        self.time_axis = None
        
        # Color maps
        self.color_maps = {
            "viridis": self._generate_viridis_colormap(),
            "plasma": self._generate_plasma_colormap(),
            "inferno": self._generate_inferno_colormap(),
            "magma": self._generate_magma_colormap()
        }
        
        # Surface for rendering
        self.surface = None
        self.needs_redraw = True
        
        return True
    
    def update(self, data: Dict[str, Any]):
        """Update spectrogram with new FFT data"""
        if not self._enabled:
            return
            
        fft_data = data.get("fft_data")
        if fft_data is None:
            return
            
        # Convert to dB scale
        magnitude_db = 20 * np.log10(fft_data + 1e-10)
        
        # Add to history
        self.spectrogram_data.append(magnitude_db)
        self.needs_redraw = True
        
        # Update frequency bins if needed
        if self.frequency_bins is None:
            sample_rate = data.get("sample_rate", 48000)
            self.frequency_bins = np.fft.rfftfreq(len(fft_data) * 2 - 1, 1 / sample_rate)
    
    def draw(self, screen, x: int, y: int, width: int, height: int):
        """Draw the spectrogram panel"""
        if not self._enabled or not self._visible:
            return
            
        # Create or resize surface if needed
        if self.surface is None or self.surface.get_size() != (width, height):
            self.surface = pygame.Surface((width, height))
            self.needs_redraw = True
            
        if self.needs_redraw and len(self.spectrogram_data) > 0:
            self._redraw_spectrogram()
            self.needs_redraw = False
            
        # Draw to screen
        screen.blit(self.surface, (x, y))
        
        # Draw border
        pygame.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)
        
        # Draw title
        if "small" in self._font_cache:
            font = self._font_cache["small"]
            title = font.render("Spectrogram", True, (255, 255, 255))
            screen.blit(title, (x + 5, y + 5))
            
        # Draw axes labels
        self._draw_axes_labels(screen, x, y, width, height)
    
    def _redraw_spectrogram(self):
        """Redraw the spectrogram surface"""
        width, height = self.surface.get_size()
        
        # Convert deque to numpy array
        data_array = np.array(list(self.spectrogram_data)).T
        
        if data_array.size == 0:
            return
            
        # Normalize to 0-1 range
        vmin = -self.db_range
        vmax = 0
        normalized = (data_array - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Get color map
        colormap = self.color_maps.get(self.color_map, self.color_maps["viridis"])
        
        # Create pixel array
        pixels = pygame.surfarray.pixels3d(self.surface)
        
        # Map data to pixels
        data_height, data_width = normalized.shape
        
        for y in range(height):
            # Map y to frequency bin (inverted - high frequencies at top)
            freq_idx = int((1 - y / height) * (data_height - 1))
            
            for x in range(width):
                # Map x to time index
                time_idx = int(x / width * (data_width - 1))
                
                # Get value and color
                value = normalized[freq_idx, time_idx]
                color_idx = int(value * (len(colormap) - 1))
                color = colormap[color_idx]
                
                # Set pixel
                pixels[x, y] = color
                
        del pixels  # Release surface lock
    
    def _draw_axes_labels(self, screen, x: int, y: int, width: int, height: int):
        """Draw frequency and time axes labels"""
        if "tiny" not in self._font_cache:
            return
            
        font = self._font_cache["tiny"]
        
        # Frequency labels (vertical axis)
        if self.frequency_bins is not None:
            max_freq = self.frequency_bins[-1]
            freq_labels = [0, max_freq / 4, max_freq / 2, 3 * max_freq / 4, max_freq]
            
            for i, freq in enumerate(freq_labels):
                y_pos = y + height - (i / 4 * height)
                label = f"{freq/1000:.1f}k" if freq >= 1000 else f"{freq:.0f}"
                text = font.render(label, True, (150, 150, 150))
                screen.blit(text, (x - text.get_width() - 5, y_pos - text.get_height() // 2))
                
        # Time labels (horizontal axis)
        time_labels = ["Now", "-1s", "-2s", "-3s", "-4s"]
        for i, label in enumerate(time_labels):
            if i == 0:
                x_pos = x + width - 30
            else:
                x_pos = x + width - (i / 4 * width)
            text = font.render(label, True, (150, 150, 150))
            screen.blit(text, (x_pos - text.get_width() // 2, y + height + 5))
    
    def _generate_viridis_colormap(self):
        """Generate viridis-like colormap"""
        colors = []
        for i in range(256):
            t = i / 255.0
            r = int(68 + (253 - 68) * t)
            g = int(1 + (231 - 1) * t)
            b = int(84 + (37 - 84) * t)
            colors.append((r, g, b))
        return colors
    
    def _generate_plasma_colormap(self):
        """Generate plasma-like colormap"""
        colors = []
        for i in range(256):
            t = i / 255.0
            r = int(13 + (240 - 13) * t)
            g = int(8 + (33 - 8) * t)
            b = int(135 + (33 - 135) * t)
            colors.append((r, g, b))
        return colors
    
    def _generate_inferno_colormap(self):
        """Generate inferno-like colormap"""
        colors = []
        for i in range(256):
            t = i / 255.0
            r = int(0 + (252 - 0) * t)
            g = int(0 + (255 - 0) * t * 0.8)
            b = int(4 + (37 - 4) * (1 - t))
            colors.append((r, g, b))
        return colors
    
    def _generate_magma_colormap(self):
        """Generate magma-like colormap"""
        colors = []
        for i in range(256):
            t = i / 255.0
            r = int(0 + (252 - 0) * t)
            g = int(0 + (253 - 0) * t * 0.6)
            b = int(4 + (191 - 4) * t)
            colors.append((r, g, b))
        return colors
    
    def on_config_change(self):
        """Handle configuration changes"""
        # Update settings
        self.history_size = self._config.get("history_size", 200)
        self.color_map = self._config.get("color_map", "viridis")
        self.db_range = self._config.get("db_range", 80.0)
        self.interpolation = self._config.get("interpolation", True)
        
        # Resize history if needed
        if len(self.spectrogram_data) > self.history_size:
            new_data = deque(maxlen=self.history_size)
            for _ in range(self.history_size):
                new_data.append(self.spectrogram_data.popleft())
            self.spectrogram_data = new_data
            
        self.needs_redraw = True
    
    def handle_event(self, event) -> bool:
        """Handle mouse/keyboard events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                # Cycle through color maps
                maps = list(self.color_maps.keys())
                current_idx = maps.index(self.color_map)
                self.color_map = maps[(current_idx + 1) % len(maps)]
                self._config["color_map"] = self.color_map
                self.needs_redraw = True
                return True
                
        return False