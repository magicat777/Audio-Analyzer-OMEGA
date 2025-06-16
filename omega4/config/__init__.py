"""
OMEGA-4 Configuration System
Phase 7: Comprehensive configuration management
"""

# Import constants from original config.py for backward compatibility
from .config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    BARS_DEFAULT,
    BARS_MAX,
    MAX_FREQ,
    FFT_SIZE_BASE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    TARGET_FPS,
    BACKGROUND_COLOR,
    GRID_COLOR,
    TEXT_COLOR,
    SPECTRUM_COLOR_START,
    SPECTRUM_COLOR_END,
    VOICE_DETECTION_ENABLED,
    HARMONIC_ANALYSIS_ENABLED,
    DRUM_DETECTION_ENABLED,
    USE_THREADING,
    THREAD_POOL_SIZE,
    MAX_PROCESSING_TIME_MS,
    DEFAULT_AUDIO_SOURCE,
    INTERACTIVE_SOURCE_SELECTION,
    get_config
)

from .schema import (
    Configuration,
    AudioConfig,
    DisplayConfig,
    AnalysisConfig,
    LayoutConfig,
    KeyBindings,
    PanelConfig,
    ColorScheme,
    WindowMode,
    DEFAULT_CONFIGS
)

from .manager import ConfigurationManager
from .presets import PresetManager, Preset, PresetMetadata
from .state import StateManager, AppState, WindowState, SessionState

__all__ = [
    # Original constants
    'SAMPLE_RATE',
    'CHUNK_SIZE',
    'BARS_DEFAULT',
    'BARS_MAX',
    'MAX_FREQ',
    'FFT_SIZE_BASE',
    'DEFAULT_WIDTH',
    'DEFAULT_HEIGHT',
    'TARGET_FPS',
    'BACKGROUND_COLOR',
    'GRID_COLOR',
    'TEXT_COLOR',
    'SPECTRUM_COLOR_START',
    'SPECTRUM_COLOR_END',
    'VOICE_DETECTION_ENABLED',
    'HARMONIC_ANALYSIS_ENABLED',
    'DRUM_DETECTION_ENABLED',
    'USE_THREADING',
    'THREAD_POOL_SIZE',
    'MAX_PROCESSING_TIME_MS',
    'DEFAULT_AUDIO_SOURCE',
    'INTERACTIVE_SOURCE_SELECTION',
    'get_config',
    
    # Schema
    'Configuration',
    'AudioConfig',
    'DisplayConfig',
    'AnalysisConfig',
    'LayoutConfig',
    'KeyBindings',
    'PanelConfig',
    'ColorScheme',
    'WindowMode',
    'DEFAULT_CONFIGS',
    
    # Manager
    'ConfigurationManager',
    
    # Presets
    'PresetManager',
    'Preset',
    'PresetMetadata',
    
    # State
    'StateManager',
    'AppState',
    'WindowState',
    'SessionState'
]