"""
OMEGA-4 User Interface Components
Phase 7: UI components for configuration
"""

from .settings import SettingsPanel, SettingsTab
from .panel_canvas import (
    TechnicalPanelCanvas, 
    PanelInfo, 
    PanelDimensions,
    PanelOrientation
)

__all__ = [
    'SettingsPanel',
    'SettingsTab',
    'TechnicalPanelCanvas',
    'PanelInfo',
    'PanelDimensions',
    'PanelOrientation'
]