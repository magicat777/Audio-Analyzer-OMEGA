"""
Configuration for OMEGA-4 Audio Analyzer
Simple, centralized configuration management
"""

# Audio Processing Configuration
SAMPLE_RATE = 48000
CHUNK_SIZE = 512  # Reduced for lower latency (10.7ms instead of 21.3ms)
BARS_DEFAULT = 512  # Better balance of detail and performance with Mel-scale
BARS_MAX = 1536
MAX_FREQ = 20000
FFT_SIZE_BASE = 2048  # Further reduced for lower latency (42.7ms window)

# Display Configuration
DEFAULT_WIDTH = 1800
DEFAULT_HEIGHT = 1000
TARGET_FPS = 60

# Colors (RGB tuples)
BACKGROUND_COLOR = (10, 10, 20)
GRID_COLOR = (40, 40, 60)
TEXT_COLOR = (200, 200, 220)
SPECTRUM_COLOR_START = (0, 100, 255)  # Blue
SPECTRUM_COLOR_END = (255, 0, 100)    # Red

# Analysis Configuration
VOICE_DETECTION_ENABLED = True
HARMONIC_ANALYSIS_ENABLED = True
DRUM_DETECTION_ENABLED = True

# Performance Settings
USE_THREADING = True
THREAD_POOL_SIZE = 4
MAX_PROCESSING_TIME_MS = 50.0

# Audio Source Settings
DEFAULT_AUDIO_SOURCE = None  # Use system default
INTERACTIVE_SOURCE_SELECTION = False

def get_config():
    """Get configuration as a dictionary for backward compatibility"""
    return {
        'sample_rate': SAMPLE_RATE,
        'chunk_size': CHUNK_SIZE,
        'bars_default': BARS_DEFAULT,
        'bars_max': BARS_MAX,
        'max_freq': MAX_FREQ,
        'fft_size_base': FFT_SIZE_BASE,
        'default_width': DEFAULT_WIDTH,
        'default_height': DEFAULT_HEIGHT,
        'target_fps': TARGET_FPS,
    }