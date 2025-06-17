"""
Display Interface for OMEGA-4 Spectrum Analyzer
Phase 2: Extract display layer with clear interface boundary
Enhanced version with robust error handling and proper initialization
"""

import pygame
import numpy as np
import math
import logging
import time
from typing import Dict, Any, Tuple, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

# Import waterfall 3D visualization
from .spectrum_waterfall_3d import SpectrumWaterfall3D

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BACKGROUND_COLOR = (8, 10, 15)
DEFAULT_GRID_COLOR = (40, 45, 55, 80)
DEFAULT_SCALE_COLOR = (120, 120, 140)
DEFAULT_TEXT_COLOR = (220, 220, 230)

class DisplayMode(Enum):
    """Display mode enumeration"""
    STANDARD = "standard"
    COMPACT = "compact"
    PROFESSIONAL = "professional"

@dataclass
class DisplayMetrics:
    """Display metrics and calculated values"""
    width: int
    height: int
    ui_scale: float
    bars: int
    bar_width: float
    header_height: int = 280
    left_border_width: int = 100
    meter_panel_width: int = 280
    vu_meter_width: int = 190
    bass_zoom_height: int = 200

    def __post_init__(self):
        """Calculate derived metrics"""
        if self.bar_width <= 0:
            self.bar_width = max(1.0, self.width / max(1, self.bars))

@dataclass
class FontSet:
    """Collection of fonts for different UI elements"""
    large: Optional[pygame.font.Font] = None
    medium: Optional[pygame.font.Font] = None
    small: Optional[pygame.font.Font] = None
    tiny: Optional[pygame.font.Font] = None
    grid: Optional[pygame.font.Font] = None
    mono: Optional[pygame.font.Font] = None

    def is_complete(self) -> bool:
        """Check if all essential fonts are loaded"""
        return all([self.large, self.medium, self.small, self.tiny, self.grid])

    def get_fallback_font(self, size: int = 20) -> pygame.font.Font:
        """Get fallback font if specific font not available"""
        return pygame.font.Font(None, size)

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
        # Validate inputs
        if not screen:
            raise ValueError("Screen surface cannot be None")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        if bars <= 0:
            raise ValueError("Number of bars must be positive")
            
        self.screen = screen
        self.metrics = DisplayMetrics(
            width=width,
            height=height,
            ui_scale=self._calculate_ui_scale(width),
            bars=bars,
            bar_width=width / bars
        )
        
        # Display state
        self.show_help = False
        self.display_mode = DisplayMode.PROFESSIONAL
        
        # Initialize fonts
        self.fonts = FontSet()
        self._setup_fonts()
        
        # Initialize display components
        self._setup_display()
        
        # Initialize 3D waterfall visualization
        self.waterfall_3d = SpectrumWaterfall3D(num_bars=bars, use_gpu=True)
        
        logger.info(f"SpectrumDisplay initialized: {width}x{height}, {bars} bars")
        
    def _calculate_ui_scale(self, width: int, base_width: int = 2000) -> float:
        """Calculate UI scaling factor based on window width"""
        scale = width / base_width
        return max(0.6, min(1.4, scale))  # Clamp between 60% and 140%
        
    def _setup_fonts(self):
        """Initialize fonts with proper sizing and fallbacks"""
        try:
            scale = self.metrics.ui_scale
            
            # Create fonts with scaling
            self.fonts.large = pygame.font.Font(None, max(24, int(36 * scale)))
            self.fonts.medium = pygame.font.Font(None, max(20, int(28 * scale)))
            self.fonts.small = pygame.font.Font(None, max(18, int(24 * scale)))
            self.fonts.tiny = pygame.font.Font(None, max(16, int(20 * scale)))
            self.fonts.grid = pygame.font.Font(None, max(14, int(18 * scale)))
            self.fonts.mono = pygame.font.Font(None, max(16, int(20 * scale)))
            
            logger.info(f"Fonts initialized with scale factor: {scale:.2f}")
            
        except Exception as e:
            logger.error(f"Font initialization failed: {e}")
            # Create minimal fallback fonts
            self.fonts.large = self.fonts.get_fallback_font(24)
            self.fonts.medium = self.fonts.get_fallback_font(20)
            self.fonts.small = self.fonts.get_fallback_font(18)
            self.fonts.tiny = self.fonts.get_fallback_font(16)
            self.fonts.grid = self.fonts.get_fallback_font(14)
            self.fonts.mono = self.fonts.get_fallback_font(16)
        
    def _setup_display(self):
        """Initialize display components (colors, etc.)"""
        try:
            # Generate color gradient for spectrum bars
            self.colors = self._generate_professional_colors()
            
            # Pre-calculate common colors
            self.background_color = DEFAULT_BACKGROUND_COLOR
            self.grid_color = DEFAULT_GRID_COLOR
            self.scale_color = DEFAULT_SCALE_COLOR
            self.text_color = DEFAULT_TEXT_COLOR
            
            logger.info("Display components initialized successfully")
            
        except Exception as e:
            logger.error(f"Display setup failed: {e}")
            # Fallback colors
            self.colors = [(100, 150, 255)] * self.metrics.bars
            self.background_color = (20, 20, 30)
            
    def render_frame(self, spectrum_data: np.ndarray, analysis_results: Dict[str, Any],
                    vis_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Main entry point for rendering a frame.
        
        Args:
            spectrum_data: Array of spectrum bar heights (0.0 to 1.0)
            analysis_results: Dictionary containing analyzer results
            vis_params: Visualization parameters (positions, sizes, etc.)
            
        Returns:
            bool: True if rendering successful, False otherwise
        """
        try:
            # Validate inputs
            if spectrum_data is None or len(spectrum_data) == 0:
                logger.warning("Empty spectrum data received")
                self.clear_screen()
                return False
                
            # Clear the screen first
            self.clear_screen()
            
            # Draw spectrum bars if we have the parameters
            if vis_params:
                return self.draw_spectrum_bars(spectrum_data, vis_params)
            else:
                logger.warning("No visualization parameters provided")
                return False
                
        except Exception as e:
            logger.error(f"Frame rendering failed: {e}")
            return False
        
    def clear_screen(self):
        """Clear the screen with background color"""
        try:
            self.screen.fill(self.background_color)
        except Exception as e:
            logger.error(f"Screen clear failed: {e}")
        
    def update_dimensions(self, width: int, height: int):
        """Update display dimensions (for window resize)"""
        if width <= 0 or height <= 0:
            logger.warning(f"Invalid dimensions: {width}x{height}")
            return
            
        old_scale = self.metrics.ui_scale
        
        # Update metrics
        self.metrics.width = width
        self.metrics.height = height
        self.metrics.ui_scale = self._calculate_ui_scale(width)
        self.metrics.bar_width = width / max(1, self.metrics.bars)
        
        # Regenerate fonts if scale changed significantly
        if abs(self.metrics.ui_scale - old_scale) > 0.1:
            self._setup_fonts()
            logger.info(f"Fonts updated for new scale: {self.metrics.ui_scale:.2f}")
        
    def resize(self, width: int, height: int):
        """
        Resize the display (alias for update_dimensions).
        Called when window is resized.
        """
        self.update_dimensions(width, height)
        # Regenerate colors for consistency
        self.colors = self._generate_professional_colors()
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """
        Set fonts for the display.
        
        Args:
            fonts: Dictionary with font names as keys:
                   'large', 'medium', 'small', 'tiny', 'grid', 'mono'
        """
        try:
            self.fonts.large = fonts.get('large', self.fonts.large)
            self.fonts.medium = fonts.get('medium', self.fonts.medium)
            self.fonts.small = fonts.get('small', self.fonts.small)
            self.fonts.tiny = fonts.get('tiny', self.fonts.tiny)
            self.fonts.grid = fonts.get('grid', self.fonts.grid)
            self.fonts.mono = fonts.get('mono', self.fonts.mono)
            
            logger.info("Custom fonts applied")
            
        except Exception as e:
            logger.error(f"Font setting failed: {e}")
        
    def draw_spectrum_bars(self, spectrum_data: np.ndarray, vis_params: Dict[str, Any]) -> bool:
        """
        Draw spectrum bars with 3D waterfall background and enhanced error handling.
        
        Args:
            spectrum_data: Bar heights (0.0 to 1.0)
            vis_params: Dictionary with visualization parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate spectrum data
            if spectrum_data is None or len(spectrum_data) == 0:
                logger.warning("Invalid spectrum data")
                return False
                
            # Clamp spectrum data to valid range
            spectrum_data = np.clip(spectrum_data, 0.0, 1.0)
            
            # Update and render 3D waterfall background first
            current_time = time.time()
            self.waterfall_3d.update_spectrum_slice(spectrum_data, current_time)
            self.waterfall_3d.render_waterfall_layers(self.screen, vis_params, spectrum_data)
            
            # Extract parameters with robust defaults
            vis_start_x = max(0, vis_params.get('vis_start_x', 0))
            vis_width = max(100, vis_params.get('vis_width', self.metrics.width))
            center_y = vis_params.get('center_y', self.metrics.height // 2)
            max_bar_height = max(50, vis_params.get('max_bar_height', 200))
            
            # Calculate boundaries
            spectrum_top = vis_params.get('spectrum_top', 
                                        vis_params.get('vis_start_y', 100))
            spectrum_bottom = vis_params.get('spectrum_bottom', 
                                           spectrum_top + vis_params.get('vis_height', 400))
            
            # Ensure center_y is within bounds
            center_y = max(spectrum_top + 10, min(spectrum_bottom - 10, center_y))
            
            # Calculate bar dimensions
            num_bars = min(len(spectrum_data), len(self.colors))
            bar_width = vis_width / num_bars if num_bars > 0 else 1
            
            # Draw main spectrum bars with transparency to show waterfall
            self._draw_transparent_spectrum_bars(
                spectrum_data, num_bars, bar_width, vis_start_x, center_y, 
                max_bar_height, spectrum_top, spectrum_bottom
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Spectrum bar drawing failed: {e}")
            return False
    
    def _draw_transparent_spectrum_bars(self, spectrum_data: np.ndarray, num_bars: int, 
                                       bar_width: float, vis_start_x: int, center_y: int,
                                       max_bar_height: int, spectrum_top: int, spectrum_bottom: int):
        """Draw main spectrum bars with transparency to allow waterfall visibility"""
        try:
            # Main spectrum transparency (0.85 = 85% opaque)
            main_alpha = 0.85
            
            # Create surface for transparent spectrum bars
            spectrum_surface = pygame.Surface((self.metrics.width, self.metrics.height), pygame.SRCALPHA)
            
            # Draw bars on the transparent surface
            for i in range(num_bars):
                if spectrum_data[i] > 0.001:
                    height = int(spectrum_data[i] * max_bar_height)
                    x = vis_start_x + int(i * bar_width)
                    width = max(1, int(np.ceil(bar_width)))
                    
                    # Get color with fallback
                    base_color = self.colors[i] if i < len(self.colors) else (100, 150, 255)
                    color_with_alpha = (*base_color, int(main_alpha * 255))
                    
                    # Draw upper bar
                    upper_y = max(spectrum_top, center_y - height)
                    upper_height = center_y - upper_y
                    if upper_height > 0:
                        upper_rect = pygame.Rect(x, upper_y, width, upper_height)
                        if self._rect_in_bounds(upper_rect):
                            pygame.draw.rect(spectrum_surface, color_with_alpha, upper_rect)
                    
                    # Draw lower bar (darker)
                    lower_height = min(height, spectrum_bottom - center_y)
                    if lower_height > 0:
                        lower_color = tuple(max(0, int(c * 0.75)) for c in base_color)
                        lower_color_with_alpha = (*lower_color, int(main_alpha * 255))
                        lower_rect = pygame.Rect(x, center_y, width, lower_height)
                        if self._rect_in_bounds(lower_rect):
                            pygame.draw.rect(spectrum_surface, lower_color_with_alpha, lower_rect)
            
            # Blit the transparent spectrum to the main screen
            self.screen.blit(spectrum_surface, (0, 0))
            
        except Exception as e:
            logger.error(f"Transparent spectrum bar drawing failed: {e}")
            # Fallback to opaque bars
            self._draw_opaque_spectrum_bars(spectrum_data, num_bars, bar_width, vis_start_x, 
                                          center_y, max_bar_height, spectrum_top, spectrum_bottom)
    
    def _draw_opaque_spectrum_bars(self, spectrum_data: np.ndarray, num_bars: int, 
                                  bar_width: float, vis_start_x: int, center_y: int,
                                  max_bar_height: int, spectrum_top: int, spectrum_bottom: int):
        """Fallback method to draw opaque spectrum bars (original implementation)"""
        for i in range(num_bars):
            if spectrum_data[i] > 0.001:
                height = int(spectrum_data[i] * max_bar_height)
                x = vis_start_x + int(i * bar_width)
                width = max(1, int(np.ceil(bar_width)))
                
                # Get color with fallback
                color = self.colors[i] if i < len(self.colors) else (100, 150, 255)
                
                # Draw upper bar
                upper_y = max(spectrum_top, center_y - height)
                upper_height = center_y - upper_y
                if upper_height > 0:
                    upper_rect = pygame.Rect(x, upper_y, width, upper_height)
                    if self._rect_in_bounds(upper_rect):
                        pygame.draw.rect(self.screen, color, upper_rect)
                
                # Draw lower bar
                lower_height = min(height, spectrum_bottom - center_y)
                if lower_height > 0:
                    lower_color = tuple(max(0, int(c * 0.75)) for c in color)
                    lower_rect = pygame.Rect(x, center_y, width, lower_height)
                    if self._rect_in_bounds(lower_rect):
                        pygame.draw.rect(self.screen, lower_color, lower_rect)

    def _rect_in_bounds(self, rect: pygame.Rect) -> bool:
        """Check if rectangle is within screen bounds"""
        return (rect.x >= 0 and rect.y >= 0 and 
                rect.right <= self.metrics.width and 
                rect.bottom <= self.metrics.height)
                               
    def _generate_professional_colors(self) -> List[Tuple[int, int, int]]:
        """Generate professional color gradient for spectrum bars"""
        colors = []
        
        try:
            for i in range(self.metrics.bars):
                hue = i / max(1, self.metrics.bars)

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

                # Clamp values to valid range
                color = (
                    max(0, min(255, r)), 
                    max(0, min(255, g)), 
                    max(0, min(255, b))
                )
                colors.append(color)

        except Exception as e:
            logger.error(f"Color generation failed: {e}")
            # Fallback to simple blue gradient
            colors = [(0, int(100 + i * 155 / max(1, self.metrics.bars)), 255) 
                     for i in range(self.metrics.bars)]

        return colors

    def _safe_render_text(self, text: str, font: Optional[pygame.font.Font], 
                         color: Tuple[int, int, int], 
                         fallback_size: int = 20) -> pygame.Surface:
        """Safely render text with fallback font if needed"""
        try:
            if font is None:
                font = self.fonts.get_fallback_font(fallback_size)
            return font.render(str(text), True, color)
        except Exception as e:
            logger.warning(f"Text rendering failed for '{text}': {e}")
            # Emergency fallback
            fallback_font = pygame.font.Font(None, fallback_size)
            return fallback_font.render("Error", True, color)

    def _safe_blit_text(self, text: str, font: Optional[pygame.font.Font], 
                       color: Tuple[int, int, int], position: Tuple[int, int],
                       fallback_size: int = 20) -> pygame.Rect:
        """Safely blit text to screen with error handling"""
        try:
            text_surface = self._safe_render_text(text, font, color, fallback_size)
            rect = self.screen.blit(text_surface, position)
            return rect
        except Exception as e:
            logger.warning(f"Text blitting failed: {e}")
            return pygame.Rect(position[0], position[1], 0, 0)
        
    def draw_grid_and_labels(self, vis_params: Dict[str, Any], ui_scale: float = 1.0):
        """
        Draw the dB grid and labels with improved error handling.
        
        Args:
            vis_params: Visualization parameters including positions
            ui_scale: UI scaling factor for fonts
        """
        try:
            # Extract parameters with validation
            spectrum_left = max(0, vis_params.get('spectrum_left', 0))
            spectrum_right = min(self.metrics.width, 
                               vis_params.get('spectrum_right', self.metrics.width))
            spectrum_top = max(0, vis_params.get('spectrum_top', 100))
            spectrum_bottom = min(self.metrics.height, 
                                vis_params.get('spectrum_bottom', self.metrics.height - 100))
            center_y = vis_params.get('center_y', (spectrum_top + spectrum_bottom) // 2)
            
            # Validate parameters
            if spectrum_right <= spectrum_left or spectrum_bottom <= spectrum_top:
                logger.warning("Invalid spectrum dimensions for grid")
                return
            
            # Amplitude grid levels
            db_levels = ([0, -20, -40, -60] if self.metrics.height < 900 
                        else [0, -10, -20, -30, -40, -50, -60])
            
            spectrum_height = spectrum_bottom - spectrum_top
            
            # Draw vertical frequency grid lines (musical octaves)
            self._draw_frequency_grid_lines(spectrum_left, spectrum_right, 
                                          spectrum_top, spectrum_bottom)
            
            # Draw center line (0dB) prominently
            pygame.draw.line(self.screen, (60, 70, 90), 
                           (spectrum_left, center_y), (spectrum_right, center_y), 2)
            
            # Draw and label 0dB
            self._draw_db_label("0", spectrum_left - 5, center_y, ui_scale)
            
            # Draw other dB levels
            for db in db_levels[1:]:
                normalized_pos = abs(db) / 60.0
                
                # Calculate positions
                upper_y = int(center_y - (spectrum_height / 2) * normalized_pos)
                lower_y = int(center_y + (spectrum_height / 2) * normalized_pos)
                
                # Draw grid lines with bounds checking
                if spectrum_top <= upper_y <= center_y:
                    pygame.draw.line(self.screen, self.grid_color,
                                   (spectrum_left, upper_y), (spectrum_right, upper_y), 1)
                    self._draw_db_label(str(db), spectrum_left - 5, upper_y, ui_scale)
                
                if center_y <= lower_y <= spectrum_bottom:
                    pygame.draw.line(self.screen, self.grid_color,
                                   (spectrum_left, lower_y), (spectrum_right, lower_y), 1)
                    self._draw_db_label(str(db), spectrum_left - 5, lower_y, ui_scale)
                    
        except Exception as e:
            logger.error(f"Grid drawing failed: {e}")

    def _draw_frequency_grid_lines(self, left: int, right: int, top: int, bottom: int):
        """Draw vertical frequency grid lines"""
        try:
            key_frequencies = [60, 250, 500, 1000, 2000, 4000, 8000, 16000]
            
            for freq in key_frequencies:
                if freq >= 20:
                    # Logarithmic position calculation
                    log_pos = ((math.log10(freq) - math.log10(20)) / 
                              (math.log10(20000) - math.log10(20)))
                    x_pos = int(left + log_pos * (right - left))
                    
                    if left <= x_pos <= right:
                        pygame.draw.line(self.screen, (30, 35, 45),
                                       (x_pos, top), (x_pos, bottom), 1)
        except Exception as e:
            logger.warning(f"Frequency grid lines drawing failed: {e}")

    def _draw_db_label(self, text: str, x: int, y: int, ui_scale: float):
        """Draw a dB label with background"""
        try:
            label_rect = pygame.Rect(x - 25, y - 10, 20, 20)
            if label_rect.right >= 5:  # Ensure label is visible
                # Background
                bg_padding = int(2 * ui_scale)
                pygame.draw.rect(self.screen, (15, 20, 30), 
                               label_rect.inflate(bg_padding * 2, bg_padding))
                
                # Text
                self._safe_blit_text(f"{text:>3}", self.fonts.grid, 
                                   (150, 160, 170), (label_rect.x, label_rect.y))
        except Exception as e:
            logger.warning(f"DB label drawing failed: {e}")

    def draw_frequency_scale(self, vis_params: Dict[str, Any], 
                           freq_to_x_func: Optional[Callable[[float], int]] = None,
                           ui_scale: float = 1.0):
        """
        Draw frequency scale labels below spectrum.
        
        Args:
            vis_params: Visualization parameters
            freq_to_x_func: Optional function to map frequency to x position
            ui_scale: UI scaling factor
        """
        try:
            scale_y = vis_params.get('scale_y', 
                                   vis_params.get('vis_start_y', 100) + 
                                   vis_params.get('vis_height', 400) + 10)
            spectrum_left = vis_params.get('spectrum_left', 0)
            spectrum_right = vis_params.get('spectrum_right', self.metrics.width)
            
            # Draw frequency markers
            frequencies = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            
            for freq in frequencies:
                if freq_to_x_func:
                    x_pos = freq_to_x_func(freq)
                else:
                    # Default logarithmic mapping
                    log_pos = ((math.log10(freq) - math.log10(20)) / 
                              (math.log10(20000) - math.log10(20)))
                    x_pos = int(spectrum_left + log_pos * (spectrum_right - spectrum_left))
                
                if spectrum_left <= x_pos <= spectrum_right:
                    # Draw tick mark
                    pygame.draw.line(self.screen, self.scale_color,
                                   (x_pos, scale_y), (x_pos, scale_y + 5), 1)
                    
                    # Draw label
                    if freq < 1000:
                        label = f"{freq}"
                    else:
                        label = f"{freq // 1000}k"
                    
                    self._safe_blit_text(label, self.fonts.grid, self.scale_color,
                                       (x_pos - 15, scale_y + 8))
                    
        except Exception as e:
            logger.error(f"Frequency scale drawing failed: {e}")

    def get_display_info(self) -> Dict[str, Any]:
        """Get current display information for debugging"""
        return {
            "metrics": {
                "width": self.metrics.width,
                "height": self.metrics.height,
                "bars": self.metrics.bars,
                "ui_scale": self.metrics.ui_scale,
                "bar_width": self.metrics.bar_width
            },
            "fonts_loaded": self.fonts.is_complete(),
            "display_mode": self.display_mode.value,
            "color_count": len(self.colors),
            "waterfall_3d": self.waterfall_3d.get_status()
        }
    
    # 3D Waterfall Control Methods
    def toggle_waterfall_3d(self) -> bool:
        """Toggle 3D waterfall visualization on/off"""
        return self.waterfall_3d.toggle_waterfall()
    
    def adjust_waterfall_depth(self, delta: float):
        """Adjust waterfall depth spacing"""
        self.waterfall_3d.adjust_depth(delta)
    
    def adjust_waterfall_speed(self, delta: float):
        """Adjust waterfall slice update interval"""
        self.waterfall_3d.adjust_slice_interval(delta)
    
    def get_waterfall_status(self) -> Dict[str, Any]:
        """Get waterfall 3D status"""
        return self.waterfall_3d.get_status()