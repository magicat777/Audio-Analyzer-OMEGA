"""
Configuration for Room Mode Analyzer
Provides configurable parameters for room acoustics analysis
"""

from dataclasses import dataclass


@dataclass
class RoomModeConfig:
    """Configuration for room mode analysis"""
    # Frequency range for room mode detection
    min_frequency: float = 30.0
    max_frequency: float = 300.0
    
    # Peak detection sensitivity
    peak_threshold_multiplier: float = 2.0
    minimum_prominence_std: float = 1.0
    
    # Q factor limits
    min_q_factor: float = 0.5
    max_q_factor: float = 100.0
    
    # Room mode severity threshold
    minimum_severity: float = 0.3
    
    # Environmental conditions
    temperature_celsius: float = 20.0
    humidity_percent: float = 50.0
    
    # Performance settings
    enable_caching: bool = True
    cache_max_age_frames: int = 10
    
    # RT60 calculation
    rt60_method: str = "schroeder"  # "schroeder", "edt", "simple"
    min_audio_history: int = 30
    
    # Room dimension hints (optional, in meters)
    room_length_hint: float = 0.0
    room_width_hint: float = 0.0
    room_height_hint: float = 0.0
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.min_frequency >= self.max_frequency:
            raise ValueError("min_frequency must be less than max_frequency")
            
        if self.peak_threshold_multiplier <= 0:
            raise ValueError("peak_threshold_multiplier must be positive")
            
        if self.minimum_prominence_std < 0:
            raise ValueError("minimum_prominence_std must be non-negative")
            
        if self.min_q_factor >= self.max_q_factor:
            raise ValueError("min_q_factor must be less than max_q_factor")
            
        if self.minimum_severity < 0 or self.minimum_severity > 1:
            raise ValueError("minimum_severity must be between 0 and 1")
            
        if self.temperature_celsius < -50 or self.temperature_celsius > 50:
            raise ValueError("temperature_celsius must be between -50 and 50")
            
        if self.humidity_percent < 0 or self.humidity_percent > 100:
            raise ValueError("humidity_percent must be between 0 and 100")
            
        if self.cache_max_age_frames < 0:
            raise ValueError("cache_max_age_frames must be non-negative")
            
        if self.rt60_method not in ["schroeder", "edt", "simple"]:
            raise ValueError("rt60_method must be one of: schroeder, edt, simple")
            
        if self.min_audio_history < 1:
            raise ValueError("min_audio_history must be at least 1")