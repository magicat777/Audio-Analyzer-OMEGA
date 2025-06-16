"""
OMEGA-4 Plugin System
Phase 6: Flexible plugin architecture for extensibility
"""

from .base import (
    Plugin,
    PanelPlugin,
    AnalyzerPlugin,
    EffectPlugin,
    InputPlugin,
    OutputPlugin,
    PluginMetadata,
    PluginType,
    PluginRegistry
)

from .manager import PluginManager, PluginLoadError
from .config import PluginConfig, PluginConfigManager

__all__ = [
    # Base classes
    'Plugin',
    'PanelPlugin',
    'AnalyzerPlugin',
    'EffectPlugin',
    'InputPlugin',
    'OutputPlugin',
    'PluginMetadata',
    'PluginType',
    'PluginRegistry',
    
    # Manager
    'PluginManager',
    'PluginLoadError',
    
    # Configuration
    'PluginConfig',
    'PluginConfigManager'
]