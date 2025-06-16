"""
Configuration Schema for OMEGA-4 Audio Analyzer
Phase 7: Define configuration structure and validation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class ColorScheme(Enum):
    """Available color schemes"""
    CLASSIC = "classic"
    DARK = "dark"
    LIGHT = "light"
    NEON = "neon"
    MATRIX = "matrix"
    SUNSET = "sunset"


class WindowMode(Enum):
    """Window display modes"""
    WINDOWED = "windowed"
    FULLSCREEN = "fullscreen"
    BORDERLESS = "borderless"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 48000
    chunk_size: int = 512
    input_device: Optional[str] = None
    input_gain: float = 4.0
    auto_gain: bool = True
    target_lufs: float = -16.0
    noise_floor: float = 0.001
    
    def validate(self) -> List[str]:
        """Validate audio configuration"""
        errors = []
        
        if self.sample_rate not in [44100, 48000, 96000, 192000]:
            errors.append(f"Invalid sample rate: {self.sample_rate}")
            
        if self.chunk_size < 128 or self.chunk_size > 4096:
            errors.append(f"Invalid chunk size: {self.chunk_size}")
            
        if self.input_gain < 0.1 or self.input_gain > 10.0:
            errors.append(f"Invalid input gain: {self.input_gain}")
            
        if self.target_lufs < -40 or self.target_lufs > 0:
            errors.append(f"Invalid target LUFS: {self.target_lufs}")
            
        return errors


@dataclass
class DisplayConfig:
    """Display configuration"""
    width: int = 2000
    height: int = 1080
    window_mode: WindowMode = WindowMode.WINDOWED
    target_fps: int = 60
    vsync: bool = True
    color_scheme: ColorScheme = ColorScheme.CLASSIC
    show_fps: bool = True
    grid_enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate display configuration"""
        errors = []
        
        if self.width < 800 or self.width > 3840:
            errors.append(f"Invalid width: {self.width}")
            
        if self.height < 600 or self.height > 2160:
            errors.append(f"Invalid height: {self.height}")
            
        if self.target_fps < 30 or self.target_fps > 144:
            errors.append(f"Invalid target FPS: {self.target_fps}")
            
        return errors


@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    fft_size: int = 4096
    num_bands: int = 768
    smoothing_factor: float = 0.7
    freq_compensation: bool = True
    psychoacoustic_weighting: bool = True
    adaptive_allocation: bool = False
    
    # Feature toggles
    voice_detection: bool = True
    drum_detection: bool = True
    pitch_detection: bool = True
    genre_classification: bool = True
    
    # Sensitivity settings
    drum_sensitivity: float = 1.0
    voice_threshold: float = 0.5
    pitch_confidence_threshold: float = 0.7
    
    def validate(self) -> List[str]:
        """Validate analysis configuration"""
        errors = []
        
        if self.fft_size not in [512, 1024, 2048, 4096, 8192]:
            errors.append(f"Invalid FFT size: {self.fft_size}")
            
        if self.num_bands < 64 or self.num_bands > 2048:
            errors.append(f"Invalid number of bands: {self.num_bands}")
            
        if self.smoothing_factor < 0 or self.smoothing_factor > 1:
            errors.append(f"Invalid smoothing factor: {self.smoothing_factor}")
            
        if self.drum_sensitivity < 0.1 or self.drum_sensitivity > 5.0:
            errors.append(f"Invalid drum sensitivity: {self.drum_sensitivity}")
            
        return errors


@dataclass
class PanelConfig:
    """Individual panel configuration"""
    enabled: bool = True
    visible: bool = True
    position: Optional[Dict[str, int]] = None
    size: Optional[Dict[str, int]] = None
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayoutConfig:
    """Panel layout configuration"""
    panels: Dict[str, PanelConfig] = field(default_factory=dict)
    layout_name: str = "default"
    
    def get_panel_config(self, panel_name: str) -> PanelConfig:
        """Get configuration for a specific panel"""
        if panel_name not in self.panels:
            self.panels[panel_name] = PanelConfig()
        return self.panels[panel_name]
        
    def set_panel_position(self, panel_name: str, x: int, y: int):
        """Set panel position"""
        config = self.get_panel_config(panel_name)
        config.position = {"x": x, "y": y}
        
    def set_panel_size(self, panel_name: str, width: int, height: int):
        """Set panel size"""
        config = self.get_panel_config(panel_name)
        config.size = {"width": width, "height": height}
        
    def set_panel_visible(self, panel_name: str, visible: bool):
        """Set panel visibility"""
        config = self.get_panel_config(panel_name)
        config.visible = visible


@dataclass
class KeyBindings:
    """Keyboard shortcuts configuration"""
    toggle_fullscreen: str = "F11"
    quit: str = "q"
    toggle_fps: str = "f"
    toggle_grid: str = "g"
    increase_gain: str = "+"
    decrease_gain: str = "-"
    reset_gain: str = "0"
    save_preset: str = "ctrl+s"
    load_preset: str = "ctrl+l"
    next_color_scheme: str = "c"
    toggle_voice_detection: str = "v"
    toggle_drum_detection: str = "d"
    sensitivity_up: str = "]"
    sensitivity_down: str = "["
    
    # Panel toggles
    toggle_professional_meters: str = "1"
    toggle_vu_meters: str = "2"
    toggle_bass_zoom: str = "3"
    toggle_harmonic_analysis: str = "4"
    toggle_pitch_detection: str = "5"
    toggle_chromagram: str = "6"
    toggle_genre_classification: str = "7"
    toggle_spectrogram: str = "8"
    toggle_waterfall: str = "9"


@dataclass
class Configuration:
    """Main configuration structure"""
    version: str = "1.0.0"
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    key_bindings: KeyBindings = field(default_factory=KeyBindings)
    plugins: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate entire configuration"""
        errors = []
        
        # Validate sub-configurations
        errors.extend(self.audio.validate())
        errors.extend(self.display.validate())
        errors.extend(self.analysis.validate())
        
        return errors
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        
        # Convert enums to strings
        if isinstance(self.display.window_mode, WindowMode):
            data["display"]["window_mode"] = self.display.window_mode.value
        if isinstance(self.display.color_scheme, ColorScheme):
            data["display"]["color_scheme"] = self.display.color_scheme.value
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Configuration':
        """Create from dictionary"""
        # Handle nested dataclasses
        if "audio" in data and isinstance(data["audio"], dict):
            data["audio"] = AudioConfig(**data["audio"])
            
        if "display" in data and isinstance(data["display"], dict):
            # Handle enum conversion
            if "window_mode" in data["display"]:
                data["display"]["window_mode"] = WindowMode(data["display"]["window_mode"])
            if "color_scheme" in data["display"]:
                data["display"]["color_scheme"] = ColorScheme(data["display"]["color_scheme"])
            data["display"] = DisplayConfig(**data["display"])
            
        if "analysis" in data and isinstance(data["analysis"], dict):
            data["analysis"] = AnalysisConfig(**data["analysis"])
            
        if "layout" in data and isinstance(data["layout"], dict):
            panels = data["layout"].get("panels", {})
            # Convert panel configs
            panel_configs = {}
            for name, panel_data in panels.items():
                panel_configs[name] = PanelConfig(**panel_data)
            data["layout"]["panels"] = panel_configs
            data["layout"] = LayoutConfig(**data["layout"])
            
        if "key_bindings" in data and isinstance(data["key_bindings"], dict):
            data["key_bindings"] = KeyBindings(**data["key_bindings"])
            
        return cls(**data)


# Default configurations for different use cases
DEFAULT_CONFIGS = {
    "default": Configuration(),
    
    "high_quality": Configuration(
        audio=AudioConfig(sample_rate=96000, chunk_size=1024),
        analysis=AnalysisConfig(fft_size=8192, num_bands=1024)
    ),
    
    "low_latency": Configuration(
        audio=AudioConfig(sample_rate=48000, chunk_size=256),
        analysis=AnalysisConfig(fft_size=2048, num_bands=512)
    ),
    
    "music_production": Configuration(
        audio=AudioConfig(input_gain=2.0, target_lufs=-14.0),
        analysis=AnalysisConfig(
            freq_compensation=True,
            psychoacoustic_weighting=True,
            voice_detection=False,
            genre_classification=False
        )
    ),
    
    "live_performance": Configuration(
        display=DisplayConfig(show_fps=False, grid_enabled=False),
        analysis=AnalysisConfig(
            drum_detection=True,
            drum_sensitivity=1.5,
            voice_detection=True
        )
    )
}