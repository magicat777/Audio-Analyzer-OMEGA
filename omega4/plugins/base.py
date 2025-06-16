"""
Plugin Base Classes for OMEGA-4 Audio Analyzer
Phase 6: Define plugin interfaces and registry
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class PluginType(Enum):
    """Types of plugins supported"""
    PANEL = "panel"
    ANALYZER = "analyzer"
    EFFECT = "effect"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = None
    config_schema: Dict = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class Plugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self):
        self._enabled = True
        self._config = {}
        self._metadata = None
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    def initialize(self, config: Dict = None) -> bool:
        """Initialize the plugin with configuration"""
        self._metadata = self.get_metadata()
        if config:
            self._config = config
        return True
    
    def shutdown(self):
        """Clean up plugin resources"""
        pass
    
    def enable(self):
        """Enable the plugin"""
        self._enabled = True
    
    def disable(self):
        """Disable the plugin"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if plugin is enabled"""
        return self._enabled
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self._config.copy()
    
    def set_config(self, config: Dict):
        """Update configuration"""
        self._config.update(config)
        self.on_config_change()
    
    def on_config_change(self):
        """Called when configuration changes"""
        pass


class PanelPlugin(Plugin):
    """Base class for visualization panel plugins"""
    
    def __init__(self):
        super().__init__()
        self._position = (0, 0)
        self._size = (100, 100)
        self._visible = True
        self._font_cache = {}
    
    @abstractmethod
    def update(self, data: Dict[str, Any]):
        """Update panel with new data
        
        Args:
            data: Dictionary containing:
                - audio_data: Current audio chunk
                - fft_data: FFT magnitude data
                - band_values: Mapped frequency bands
                - voice_info: Voice detection results
                - drum_info: Drum detection results
                - sample_rate: Audio sample rate
                - time_delta: Time since last update
        """
        pass
    
    @abstractmethod
    def draw(self, screen, x: int, y: int, width: int, height: int):
        """Draw the panel on screen
        
        Args:
            screen: Pygame screen surface
            x: Panel x position
            y: Panel y position
            width: Panel width
            height: Panel height
        """
        pass
    
    def set_position(self, x: int, y: int):
        """Set panel position"""
        self._position = (x, y)
    
    def set_size(self, width: int, height: int):
        """Set panel size"""
        self._size = (width, height)
    
    def set_fonts(self, fonts: Dict[str, Any]):
        """Set font cache for rendering"""
        self._font_cache = fonts
    
    def show(self):
        """Show the panel"""
        self._visible = True
    
    def hide(self):
        """Hide the panel"""
        self._visible = False
    
    def is_visible(self) -> bool:
        """Check if panel is visible"""
        return self._visible
    
    def handle_event(self, event) -> bool:
        """Handle user input events
        
        Returns:
            True if event was handled, False otherwise
        """
        return False


class AnalyzerPlugin(Plugin):
    """Base class for audio analyzer plugins"""
    
    def __init__(self):
        super().__init__()
        self._sample_rate = 48000
    
    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate"""
        self._sample_rate = sample_rate
    
    @abstractmethod
    def process(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process audio data
        
        Args:
            audio_data: Audio samples to process
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with analysis results
        """
        pass
    
    def reset(self):
        """Reset analyzer state"""
        pass


class EffectPlugin(Plugin):
    """Base class for audio effect plugins"""
    
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply effect to audio data
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Processed audio samples
        """
        pass


class InputPlugin(Plugin):
    """Base class for audio input plugins"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start audio input"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop audio input"""
        pass
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read audio data
        
        Returns:
            Audio samples or None if no data available
        """
        pass


class OutputPlugin(Plugin):
    """Base class for output plugins (MIDI, OSC, etc.)"""
    
    @abstractmethod
    def send(self, data: Dict[str, Any]):
        """Send data to output"""
        pass


class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
    
    def register(self, plugin: Plugin) -> bool:
        """Register a plugin
        
        Returns:
            True if successful, False if plugin name already exists
        """
        metadata = plugin.get_metadata()
        
        if metadata.name in self._plugins:
            return False
        
        self._plugins[metadata.name] = plugin
        self._plugins_by_type[metadata.plugin_type].append(metadata.name)
        
        return True
    
    def unregister(self, name: str) -> bool:
        """Unregister a plugin
        
        Returns:
            True if successful, False if plugin not found
        """
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        metadata = plugin.get_metadata()
        
        del self._plugins[name]
        self._plugins_by_type[metadata.plugin_type].remove(name)
        
        return True
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name"""
        return self._plugins.get(name)
    
    def get_all(self) -> Dict[str, Plugin]:
        """Get all registered plugins"""
        return self._plugins.copy()
    
    def get_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all plugins of a specific type"""
        names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name] for name in names]
    
    def get_enabled(self) -> List[Plugin]:
        """Get all enabled plugins"""
        return [p for p in self._plugins.values() if p.is_enabled()]
    
    def get_panel_plugins(self) -> List[PanelPlugin]:
        """Get all panel plugins"""
        return [p for p in self.get_by_type(PluginType.PANEL) if isinstance(p, PanelPlugin)]
    
    def get_analyzer_plugins(self) -> List[AnalyzerPlugin]:
        """Get all analyzer plugins"""
        return [p for p in self.get_by_type(PluginType.ANALYZER) if isinstance(p, AnalyzerPlugin)]