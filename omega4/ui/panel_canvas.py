"""
Technical Panel Canvas System - Dynamic panel layout management
"""

import pygame
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class PanelOrientation(Enum):
    """Panel orientation preferences"""
    LANDSCAPE = "landscape"  # Wider than tall
    PORTRAIT = "portrait"    # Taller than wide
    SQUARE = "square"       # Roughly equal dimensions


@dataclass
class PanelDimensions:
    """Panel dimension specifications"""
    min_width: int
    optimal_width: int
    min_height: int
    optimal_height: int
    fixed_height: Optional[int] = None  # If panel has fixed height
    orientation: PanelOrientation = PanelOrientation.SQUARE
    
    @property
    def height(self) -> int:
        """Get the panel height (fixed or optimal)"""
        return self.fixed_height if self.fixed_height else self.optimal_height
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)"""
        return self.optimal_width / self.height


class PanelInfo:
    """Information about a panel including its instance and layout properties"""
    
    def __init__(self, panel_id: str, panel_instance: Any, dimensions: PanelDimensions):
        self.id = panel_id
        self.panel = panel_instance
        self.dimensions = dimensions
        self.is_visible = False
        self.current_rect: Optional[pygame.Rect] = None
        self.target_rect: Optional[pygame.Rect] = None  # For animations
        self.animation_progress = 0.0
    
    def get_preferred_size(self, available_width: int) -> Tuple[int, int]:
        """Get preferred size based on available width"""
        # Use optimal width unless it exceeds available space
        width = min(self.dimensions.optimal_width, available_width)
        width = max(width, self.dimensions.min_width)  # Ensure minimum
        
        height = self.dimensions.height
        
        return width, height


class Row:
    """Represents a row of panels in the canvas"""
    
    def __init__(self, y: int, max_width: int, padding: int = 10):
        self.y = y
        self.max_width = max_width
        self.padding = padding
        self.panels: List[PanelInfo] = []
        self.height = 0
        self.used_width = 0
    
    def can_fit(self, panel_info: PanelInfo) -> bool:
        """Check if panel can fit in this row"""
        required_width = panel_info.dimensions.min_width + self.padding
        if self.panels:
            required_width += self.padding  # Add padding between panels
        
        return self.used_width + required_width <= self.max_width
    
    def add_panel(self, panel_info: PanelInfo) -> pygame.Rect:
        """Add panel to row and return its position"""
        x = self.used_width
        if self.panels:
            x += self.padding  # Add padding from previous panel
        
        width, height = panel_info.get_preferred_size(
            self.max_width - x - self.padding
        )
        
        # Update row height to accommodate tallest panel
        self.height = max(self.height, height)
        
        # Create rect for panel
        rect = pygame.Rect(x, self.y, width, height)
        panel_info.current_rect = rect
        
        # Update used width
        self.used_width = x + width
        self.panels.append(panel_info)
        
        return rect
    
    def remove_panel(self, panel_id: str) -> bool:
        """Remove panel from row"""
        for i, panel in enumerate(self.panels):
            if panel.id == panel_id:
                self.panels.pop(i)
                self.reflow()
                return True
        return False
    
    def reflow(self):
        """Recalculate panel positions after removal"""
        self.used_width = 0
        self.height = 0
        
        for panel in self.panels:
            x = self.used_width
            if self.used_width > 0:
                x += self.padding
            
            width, height = panel.get_preferred_size(
                self.max_width - x - self.padding
            )
            
            panel.current_rect = pygame.Rect(x, self.y, width, height)
            self.used_width = x + width
            self.height = max(self.height, height)


class PanelLayoutEngine:
    """Manages the layout of panels using a row-based bin packing algorithm"""
    
    def __init__(self, max_width: int, padding: int = 10):
        self.max_width = max_width
        self.padding = padding
        self.rows: List[Row] = []
    
    def add_panel(self, panel_info: PanelInfo, start_y: int = 0) -> pygame.Rect:
        """Add panel to layout and return its position"""
        # Try to fit in existing rows first
        for row in self.rows:
            if row.can_fit(panel_info):
                return row.add_panel(panel_info)
        
        # Create new row
        y = start_y
        if self.rows:
            last_row = self.rows[-1]
            y = last_row.y + last_row.height + self.padding
        
        new_row = Row(y, self.max_width, self.padding)
        self.rows.append(new_row)
        return new_row.add_panel(panel_info)
    
    def remove_panel(self, panel_id: str) -> bool:
        """Remove panel and reflow layout"""
        for row in self.rows:
            if row.remove_panel(panel_id):
                # Remove empty rows
                self.rows = [r for r in self.rows if r.panels]
                # Reflow all rows
                self.reflow_rows()
                return True
        return False
    
    def reflow_rows(self):
        """Recalculate row positions after changes"""
        y = 0
        for row in self.rows:
            row.y = y
            # Update panel rects with new row position
            for panel in row.panels:
                if panel.current_rect:
                    panel.current_rect.y = y
            y += row.height + self.padding
    
    def get_total_height(self) -> int:
        """Get total height of all rows"""
        if not self.rows:
            return 0
        
        last_row = self.rows[-1]
        return last_row.y + last_row.height
    
    def clear(self):
        """Clear all panels"""
        self.rows.clear()


class TechnicalPanelCanvas:
    """Main canvas for managing technical panels"""
    
    # Panel dimension registry
    PANEL_DIMENSIONS = {
        'transient_detection': PanelDimensions(350, 450, 280, 350, orientation=PanelOrientation.PORTRAIT),  # Normalized to genre classification
        'voice_detection': PanelDimensions(400, 450, 180, 180, fixed_height=180, orientation=PanelOrientation.LANDSCAPE),
        'phase_correlation': PanelDimensions(480, 600, 400, 400, fixed_height=400, orientation=PanelOrientation.LANDSCAPE),  # Increased width by 20%
        'harmonic_analysis': PanelDimensions(300, 400, 250, 300, orientation=PanelOrientation.PORTRAIT),
        'professional_meters': PanelDimensions(350, 450, 280, 350, orientation=PanelOrientation.PORTRAIT),
        'bass_zoom': PanelDimensions(400, 1650, 150, 250, orientation=PanelOrientation.LANDSCAPE),  # Prefers full width
        'chromagram': PanelDimensions(330, 440, 380, 450, orientation=PanelOrientation.PORTRAIT),  # Further increased height for better layout
        'genre_classification': PanelDimensions(350, 450, 280, 350, orientation=PanelOrientation.PORTRAIT),
        'pitch_detection': PanelDimensions(350, 450, 280, 350, orientation=PanelOrientation.PORTRAIT),  # Normalized to genre classification
        'vu_meters': PanelDimensions(200, 300, 300, 400, orientation=PanelOrientation.PORTRAIT),
        'integrated_music': PanelDimensions(360, 480, 400, 500, orientation=PanelOrientation.LANDSCAPE),  # Reduced width by 40%
        'room_analysis': PanelDimensions(300, 400, 200, 250, orientation=PanelOrientation.SQUARE),
        'beat_detection': PanelDimensions(350, 450, 280, 350, orientation=PanelOrientation.PORTRAIT),  # Beat detection and BPM
        'spectrogram_waterfall': PanelDimensions(500, 800, 300, 400, orientation=PanelOrientation.LANDSCAPE),  # Spectrogram waterfall display
        'frequency_band_tracker': PanelDimensions(400, 600, 250, 350, orientation=PanelOrientation.LANDSCAPE),  # Frequency band energy tracker
    }
    
    def __init__(self, x: int, y: int, max_width: int, padding: int = 10):
        self.x = x
        self.y = y
        self.max_width = max_width
        self.padding = padding
        
        # Panel registry
        self.panels: Dict[str, PanelInfo] = {}
        
        # Layout engine
        self.layout_engine = PanelLayoutEngine(max_width, padding)
        
        # Visual properties
        self.bg_color = (10, 12, 15)
        self.border_color = (40, 45, 50)
        self.panel_bg_color = (15, 18, 22)
        self.panel_border_color = (50, 55, 60)
        
        # Animation settings
        self.animation_speed = 0.15  # 0-1 per frame
    
    def register_panel(self, panel_id: str, panel_instance: Any):
        """Register a panel with the canvas"""
        if panel_id in self.PANEL_DIMENSIONS:
            dimensions = self.PANEL_DIMENSIONS[panel_id]
            panel_info = PanelInfo(panel_id, panel_instance, dimensions)
            self.panels[panel_id] = panel_info
    
    def toggle_panel(self, panel_id: str) -> bool:
        """Toggle panel visibility and return new state"""
        if panel_id not in self.panels:
            return False
        
        panel_info = self.panels[panel_id]
        
        if panel_info.is_visible:
            # Hide panel
            self.layout_engine.remove_panel(panel_id)
            panel_info.is_visible = False
            panel_info.current_rect = None
        else:
            # Show panel
            rect = self.layout_engine.add_panel(panel_info, 0)  # Start at 0 within canvas
            panel_info.is_visible = True
            panel_info.animation_progress = 0.0
        
        return panel_info.is_visible
    
    def update(self, dt: float):
        """Update canvas animations"""
        for panel_info in self.panels.values():
            if panel_info.is_visible and panel_info.animation_progress < 1.0:
                panel_info.animation_progress = min(
                    1.0, 
                    panel_info.animation_progress + self.animation_speed
                )
    
    def draw(self, screen: pygame.Surface):
        """Draw the canvas and all visible panels"""
        total_height = self.get_total_height()
        
        if total_height > 0:
            # Draw canvas background
            canvas_rect = pygame.Rect(self.x, self.y, self.max_width, total_height)
            pygame.draw.rect(screen, self.bg_color, canvas_rect)
            pygame.draw.rect(screen, self.border_color, canvas_rect, 1)
            
            # Draw visible panels
            for panel_info in self.panels.values():
                if panel_info.is_visible and panel_info.current_rect:
                    # Apply animation
                    alpha = panel_info.animation_progress
                    
                    # Draw panel background with border
                    panel_rect = panel_info.current_rect.copy()
                    panel_rect.x += self.x
                    panel_rect.y += self.y  # Also offset y by canvas position
                    
                    # Panel background
                    pygame.draw.rect(screen, self.panel_bg_color, panel_rect)
                    pygame.draw.rect(screen, self.panel_border_color, panel_rect, 1)
                    
                    # Draw panel content with fade-in
                    if alpha > 0.1:  # Start drawing content after initial fade
                        # Create subsurface for clipping
                        try:
                            # Draw the panel based on its type
                            if panel_info.id == 'phase_correlation':
                                # Phase correlation now accepts height parameter
                                panel_info.panel.draw(
                                    screen, 
                                    panel_rect.x,
                                    panel_rect.y, 
                                    panel_rect.width,
                                    panel_rect.height
                                )
                            elif panel_info.id == 'transient_detection':
                                # Transient detection now accepts height parameter
                                panel_info.panel.draw(
                                    screen, 
                                    panel_rect.x,
                                    panel_rect.y, 
                                    panel_rect.width,
                                    panel_rect.height
                                )
                            elif panel_info.id == 'voice_detection':
                                # Voice detection uses simpler signature
                                panel_info.panel.draw(
                                    screen, 
                                    panel_rect.x,
                                    panel_rect.y, 
                                    panel_rect.width
                                )
                            else:
                                # Legacy panels need height and ui_scale
                                ui_scale = panel_rect.width / 400.0  # Base scale on width
                                panel_info.panel.draw(
                                    screen, 
                                    panel_rect.x,
                                    panel_rect.y, 
                                    panel_rect.width,
                                    panel_rect.height,
                                    ui_scale
                                )
                        except Exception as e:
                            print(f"Error drawing panel {panel_info.id}: {e}")
    
    def get_total_height(self) -> int:
        """Get total height of canvas"""
        if not any(p.is_visible for p in self.panels.values()):
            return 0
        
        return self.layout_engine.get_total_height() + self.padding * 2
    
    def get_visible_panels(self) -> List[str]:
        """Get list of visible panel IDs"""
        return [pid for pid, pinfo in self.panels.items() if pinfo.is_visible]
    
    def arrange_panels_optimally(self):
        """Rearrange all visible panels for optimal layout"""
        # Get visible panels sorted by size preference
        visible_panels = [
            (pid, pinfo) for pid, pinfo in self.panels.items() 
            if pinfo.is_visible
        ]
        
        # Sort by orientation and size
        # Landscape panels first, then by width preference
        visible_panels.sort(
            key=lambda x: (
                x[1].dimensions.orientation != PanelOrientation.LANDSCAPE,
                -x[1].dimensions.optimal_width
            )
        )
        
        # Clear and re-add all panels
        self.layout_engine.clear()
        
        for panel_id, panel_info in visible_panels:
            rect = self.layout_engine.add_panel(panel_info, self.y)
            if rect:
                rect.x += self.x
                rect.y = self.y + (rect.y - self.y)