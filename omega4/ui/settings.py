"""
Settings UI Panel for OMEGA-4 Audio Analyzer
Phase 7: In-app configuration interface
"""

import pygame
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from omega4.config.schema import Configuration, ColorScheme, WindowMode
from omega4.config.manager import ConfigurationManager
from omega4.config.presets import PresetManager


class SettingsTab(Enum):
    """Settings panel tabs"""
    AUDIO = "Audio"
    DISPLAY = "Display"
    ANALYSIS = "Analysis"
    PANELS = "Panels"
    PRESETS = "Presets"
    KEYBINDS = "Key Bindings"


@dataclass
class UIElement:
    """Base UI element"""
    x: int
    y: int
    width: int
    height: int
    label: str
    value: Any
    
    def get_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
        
    def contains_point(self, x: int, y: int) -> bool:
        return self.get_rect().collidepoint(x, y)


class SettingsPanel:
    """Settings configuration panel"""
    
    def __init__(self, config_manager: ConfigurationManager, 
                 preset_manager: PresetManager,
                 width: int = 800, height: int = 600):
        self.config_manager = config_manager
        self.preset_manager = preset_manager
        self.width = width
        self.height = height
        
        # UI state
        self.visible = False
        self.current_tab = SettingsTab.AUDIO
        self.scroll_offset = 0
        self.selected_element = None
        
        # Fonts (will be set by main app)
        self.fonts = {}
        
        # UI elements
        self.tabs: List[UIElement] = []
        self.elements: Dict[SettingsTab, List[UIElement]] = {
            tab: [] for tab in SettingsTab
        }
        
        # Callbacks
        self.on_apply: Optional[Callable] = None
        self.on_close: Optional[Callable] = None
        
        # Initialize UI
        self._init_ui()
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set font cache"""
        self.fonts = fonts
        
    def _init_ui(self):
        """Initialize UI elements"""
        # Create tabs
        tab_width = self.width // len(SettingsTab)
        for i, tab in enumerate(SettingsTab):
            self.tabs.append(UIElement(
                x=i * tab_width,
                y=0,
                width=tab_width,
                height=40,
                label=tab.value,
                value=tab
            ))
            
        # Create elements for each tab
        self._create_audio_elements()
        self._create_display_elements()
        self._create_analysis_elements()
        self._create_panels_elements()
        self._create_presets_elements()
        self._create_keybinds_elements()
        
    def _create_audio_elements(self):
        """Create audio settings elements"""
        y_offset = 60
        spacing = 50
        
        config = self.config_manager.config.audio if self.config_manager.config else None
        if not config:
            return
            
        # Sample Rate
        self.elements[SettingsTab.AUDIO].append(
            self._create_dropdown(
                20, y_offset, 300, 40,
                "Sample Rate",
                config.sample_rate,
                [44100, 48000, 96000, 192000]
            )
        )
        y_offset += spacing
        
        # Chunk Size
        self.elements[SettingsTab.AUDIO].append(
            self._create_dropdown(
                20, y_offset, 300, 40,
                "Chunk Size",
                config.chunk_size,
                [256, 512, 1024, 2048]
            )
        )
        y_offset += spacing
        
        # Input Gain
        self.elements[SettingsTab.AUDIO].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Input Gain",
                config.input_gain,
                0.1, 10.0
            )
        )
        y_offset += spacing
        
        # Auto Gain
        self.elements[SettingsTab.AUDIO].append(
            self._create_toggle(
                20, y_offset, 300, 40,
                "Auto Gain",
                config.auto_gain
            )
        )
        y_offset += spacing
        
        # Target LUFS
        self.elements[SettingsTab.AUDIO].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Target LUFS",
                config.target_lufs,
                -40.0, 0.0
            )
        )
        
    def _create_display_elements(self):
        """Create display settings elements"""
        y_offset = 60
        spacing = 50
        
        config = self.config_manager.config.display if self.config_manager.config else None
        if not config:
            return
            
        # Window Mode
        self.elements[SettingsTab.DISPLAY].append(
            self._create_dropdown(
                20, y_offset, 300, 40,
                "Window Mode",
                config.window_mode.value,
                [mode.value for mode in WindowMode]
            )
        )
        y_offset += spacing
        
        # Color Scheme
        self.elements[SettingsTab.DISPLAY].append(
            self._create_dropdown(
                20, y_offset, 300, 40,
                "Color Scheme",
                config.color_scheme.value,
                [scheme.value for scheme in ColorScheme]
            )
        )
        y_offset += spacing
        
        # Target FPS
        self.elements[SettingsTab.DISPLAY].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Target FPS",
                config.target_fps,
                30, 144
            )
        )
        y_offset += spacing
        
        # Show FPS
        self.elements[SettingsTab.DISPLAY].append(
            self._create_toggle(
                20, y_offset, 300, 40,
                "Show FPS",
                config.show_fps
            )
        )
        y_offset += spacing
        
        # Grid
        self.elements[SettingsTab.DISPLAY].append(
            self._create_toggle(
                20, y_offset, 300, 40,
                "Show Grid",
                config.grid_enabled
            )
        )
        
    def _create_analysis_elements(self):
        """Create analysis settings elements"""
        y_offset = 60
        spacing = 50
        
        config = self.config_manager.config.analysis if self.config_manager.config else None
        if not config:
            return
            
        # FFT Size
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_dropdown(
                20, y_offset, 300, 40,
                "FFT Size",
                config.fft_size,
                [512, 1024, 2048, 4096, 8192]
            )
        )
        y_offset += spacing
        
        # Number of Bands
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Frequency Bands",
                config.num_bands,
                64, 2048
            )
        )
        y_offset += spacing
        
        # Smoothing
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Smoothing",
                config.smoothing_factor,
                0.0, 1.0
            )
        )
        y_offset += spacing
        
        # Feature toggles
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_toggle(
                20, y_offset, 300, 40,
                "Voice Detection",
                config.voice_detection
            )
        )
        y_offset += spacing
        
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_toggle(
                20, y_offset, 300, 40,
                "Drum Detection",
                config.drum_detection
            )
        )
        y_offset += spacing
        
        # Drum Sensitivity
        self.elements[SettingsTab.ANALYSIS].append(
            self._create_slider(
                20, y_offset, 300, 40,
                "Drum Sensitivity",
                config.drum_sensitivity,
                0.1, 5.0
            )
        )
        
    def _create_panels_elements(self):
        """Create panel visibility elements"""
        y_offset = 60
        spacing = 40
        
        layout = self.config_manager.config.layout if self.config_manager.config else None
        if not layout:
            return
            
        # Panel toggles
        panel_names = [
            "professional_meters",
            "vu_meters",
            "bass_zoom",
            "harmonic_analysis",
            "pitch_detection",
            "chromagram",
            "genre_classification",
            "spectrogram",
            "waterfall"
        ]
        
        for panel_name in panel_names:
            panel_config = layout.get_panel_config(panel_name)
            display_name = panel_name.replace("_", " ").title()
            
            self.elements[SettingsTab.PANELS].append(
                self._create_toggle(
                    20, y_offset, 300, 30,
                    display_name,
                    panel_config.visible
                )
            )
            y_offset += spacing
            
    def _create_presets_elements(self):
        """Create preset management elements"""
        y_offset = 60
        
        # Preset list
        presets = self.preset_manager.list_presets()
        
        for i, preset_meta in enumerate(presets[:10]):  # Show first 10
            self.elements[SettingsTab.PRESETS].append(
                UIElement(
                    x=20,
                    y=y_offset + i * 40,
                    width=400,
                    height=35,
                    label=preset_meta.name,
                    value=preset_meta
                )
            )
            
    def _create_keybinds_elements(self):
        """Create key binding elements"""
        # TODO: Implement key binding configuration
        pass
        
    def _create_slider(self, x: int, y: int, width: int, height: int,
                      label: str, value: float, min_val: float, max_val: float) -> UIElement:
        """Create a slider element"""
        element = UIElement(x, y, width, height, label, value)
        element.min_val = min_val
        element.max_val = max_val
        element.element_type = "slider"
        return element
        
    def _create_toggle(self, x: int, y: int, width: int, height: int,
                      label: str, value: bool) -> UIElement:
        """Create a toggle element"""
        element = UIElement(x, y, width, height, label, value)
        element.element_type = "toggle"
        return element
        
    def _create_dropdown(self, x: int, y: int, width: int, height: int,
                        label: str, value: Any, options: List[Any]) -> UIElement:
        """Create a dropdown element"""
        element = UIElement(x, y, width, height, label, value)
        element.options = options
        element.element_type = "dropdown"
        element.expanded = False
        return element
        
    def show(self):
        """Show settings panel"""
        self.visible = True
        
    def hide(self):
        """Hide settings panel"""
        self.visible = False
        
    def toggle(self):
        """Toggle settings panel visibility"""
        self.visible = not self.visible
        
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle input events
        
        Returns:
            True if event was handled
        """
        if not self.visible:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            
            # Check tabs
            for tab_element in self.tabs:
                if tab_element.contains_point(x, y):
                    self.current_tab = tab_element.value
                    self.scroll_offset = 0
                    return True
                    
            # Check current tab elements
            for element in self.elements[self.current_tab]:
                if element.contains_point(x, y - self.scroll_offset):
                    self._handle_element_click(element)
                    return True
                    
        elif event.type == pygame.MOUSEWHEEL:
            # Scroll
            self.scroll_offset -= event.y * 20
            self.scroll_offset = max(0, self.scroll_offset)
            return True
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.hide()
                if self.on_close:
                    self.on_close()
                return True
                
        return False
        
    def _handle_element_click(self, element: UIElement):
        """Handle clicking on an element"""
        element_type = getattr(element, 'element_type', None)
        
        if element_type == "toggle":
            element.value = not element.value
            self._apply_change(element)
            
        elif element_type == "dropdown":
            element.expanded = not getattr(element, 'expanded', False)
            
        elif element_type == "slider":
            # TODO: Implement slider dragging
            pass
            
        elif self.current_tab == SettingsTab.PRESETS:
            # Load preset
            if hasattr(element.value, 'name'):
                preset = self.preset_manager.load_preset(element.value.name)
                if preset and self.config_manager:
                    self.config_manager.config = preset.configuration
                    if self.on_apply:
                        self.on_apply()
                        
    def _apply_change(self, element: UIElement):
        """Apply configuration change"""
        # TODO: Map element changes back to configuration
        if self.on_apply:
            self.on_apply()
            
    def draw(self, screen: pygame.Surface):
        """Draw settings panel"""
        if not self.visible:
            return
            
        # Create surface
        surface = pygame.Surface((self.width, self.height))
        surface.fill((30, 30, 30))
        
        # Draw border
        pygame.draw.rect(surface, (100, 100, 100), surface.get_rect(), 2)
        
        # Draw tabs
        for tab_element in self.tabs:
            color = (60, 60, 60) if tab_element.value == self.current_tab else (40, 40, 40)
            pygame.draw.rect(surface, color, tab_element.get_rect())
            pygame.draw.rect(surface, (100, 100, 100), tab_element.get_rect(), 1)
            
            # Draw tab label
            if "medium" in self.fonts:
                font = self.fonts["medium"]
                text = font.render(tab_element.label, True, (255, 255, 255))
                text_rect = text.get_rect(center=tab_element.get_rect().center)
                surface.blit(text, text_rect)
                
        # Draw current tab content
        self._draw_tab_content(surface)
        
        # Draw to screen (centered)
        x = (screen.get_width() - self.width) // 2
        y = (screen.get_height() - self.height) // 2
        screen.blit(surface, (x, y))
        
    def _draw_tab_content(self, surface: pygame.Surface):
        """Draw content for current tab"""
        # Create clipping area
        content_rect = pygame.Rect(0, 50, self.width, self.height - 50)
        
        for element in self.elements[self.current_tab]:
            # Adjust for scroll
            draw_y = element.y - self.scroll_offset
            
            if draw_y < 50 or draw_y > self.height - 50:
                continue
                
            element_type = getattr(element, 'element_type', None)
            
            if element_type == "slider":
                self._draw_slider(surface, element, draw_y)
            elif element_type == "toggle":
                self._draw_toggle(surface, element, draw_y)
            elif element_type == "dropdown":
                self._draw_dropdown(surface, element, draw_y)
            else:
                self._draw_preset_item(surface, element, draw_y)
                
    def _draw_slider(self, surface: pygame.Surface, element: UIElement, y: int):
        """Draw slider element"""
        # Label
        if "small" in self.fonts:
            font = self.fonts["small"]
            text = font.render(element.label, True, (200, 200, 200))
            surface.blit(text, (element.x, y))
            
        # Slider track
        track_y = y + 25
        track_rect = pygame.Rect(element.x, track_y, element.width, 4)
        pygame.draw.rect(surface, (60, 60, 60), track_rect)
        
        # Slider handle
        normalized = (element.value - element.min_val) / (element.max_val - element.min_val)
        handle_x = element.x + int(normalized * element.width)
        handle_rect = pygame.Rect(handle_x - 5, track_y - 5, 10, 14)
        pygame.draw.rect(surface, (100, 150, 255), handle_rect)
        
        # Value text
        value_text = f"{element.value:.2f}"
        if "tiny" in self.fonts:
            font = self.fonts["tiny"]
            text = font.render(value_text, True, (150, 150, 150))
            surface.blit(text, (element.x + element.width + 10, y + 10))
            
    def _draw_toggle(self, surface: pygame.Surface, element: UIElement, y: int):
        """Draw toggle element"""
        # Label
        if "small" in self.fonts:
            font = self.fonts["small"]
            text = font.render(element.label, True, (200, 200, 200))
            surface.blit(text, (element.x, y + 5))
            
        # Toggle switch
        switch_x = element.x + element.width - 60
        switch_rect = pygame.Rect(switch_x, y + 5, 50, 25)
        
        # Background
        bg_color = (50, 150, 50) if element.value else (80, 80, 80)
        pygame.draw.rect(surface, bg_color, switch_rect, border_radius=12)
        
        # Handle
        handle_x = switch_x + (30 if element.value else 5)
        handle_rect = pygame.Rect(handle_x, y + 8, 18, 18)
        pygame.draw.circle(surface, (255, 255, 255), handle_rect.center, 9)
        
    def _draw_dropdown(self, surface: pygame.Surface, element: UIElement, y: int):
        """Draw dropdown element"""
        # Label
        if "small" in self.fonts:
            font = self.fonts["small"]
            text = font.render(element.label, True, (200, 200, 200))
            surface.blit(text, (element.x, y))
            
        # Dropdown box
        box_y = y + 20
        box_rect = pygame.Rect(element.x, box_y, element.width, 25)
        pygame.draw.rect(surface, (50, 50, 50), box_rect)
        pygame.draw.rect(surface, (100, 100, 100), box_rect, 1)
        
        # Current value
        if "small" in self.fonts:
            font = self.fonts["small"]
            text = font.render(str(element.value), True, (255, 255, 255))
            surface.blit(text, (element.x + 5, box_y + 3))
            
    def _draw_preset_item(self, surface: pygame.Surface, element: UIElement, y: int):
        """Draw preset list item"""
        # Background
        item_rect = pygame.Rect(element.x, y, element.width, element.height)
        pygame.draw.rect(surface, (45, 45, 45), item_rect)
        pygame.draw.rect(surface, (70, 70, 70), item_rect, 1)
        
        # Preset name
        if "medium" in self.fonts and hasattr(element.value, 'name'):
            font = self.fonts["medium"]
            text = font.render(element.value.name, True, (255, 255, 255))
            surface.blit(text, (element.x + 10, y + 5))
            
        # Description
        if "tiny" in self.fonts and hasattr(element.value, 'description'):
            font = self.fonts["tiny"]
            text = font.render(element.value.description, True, (150, 150, 150))
            surface.blit(text, (element.x + 10, y + 22))