"""
Pitch Detection Configuration System for OMEGA-4 Audio Analyzer
Provides flexible configuration management for optimizing pitch detection performance
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np


@dataclass
class PitchDetectionConfig:
    """Configuration parameters for pitch detection optimization"""
    
    # Basic parameters
    sample_rate: int = 48000
    min_pitch: float = 50.0  # Hz (roughly G1)
    max_pitch: float = 2000.0  # Hz (roughly B6)
    
    # YIN algorithm parameters
    yin_threshold: float = 0.15
    yin_window_size: int = 1024
    yin_use_vectorization: bool = True
    
    # Cepstral analysis parameters
    cepstral_window_size: int = 2048
    cepstral_peak_threshold: float = 0.3
    cepstral_min_peak_distance: int = 20
    
    # Autocorrelation parameters
    autocorr_min_correlation: float = 0.3
    autocorr_peak_distance: int = 20
    
    # Method weights for combining estimates
    cepstral_weight: float = 0.8
    autocorr_weight: float = 0.9
    yin_weight: float = 1.1
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.15
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    
    # Pre-processing parameters
    enable_preprocessing: bool = True
    highpass_frequency: float = 80.0  # Hz
    highpass_order: int = 4
    enable_compression: bool = True
    compression_threshold: float = 0.1
    compression_ratio: float = 0.5
    
    # Performance optimization
    use_memory_pool: bool = True
    max_history_length: int = 100
    enable_threading: bool = False
    cache_filter_coefficients: bool = True
    
    # Stability and smoothing
    pitch_stability_window: int = 20
    pitch_stability_threshold: float = 0.1
    enable_pitch_smoothing: bool = True
    smoothing_factor: float = 0.2
    
    # Debug and monitoring
    enable_performance_monitoring: bool = False
    enable_debug_logging: bool = False
    log_processing_times: bool = False
    
    # Profile presets
    profile_name: str = "balanced"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
        self.adjust_for_sample_rate()
    
    def validate(self):
        """Validate configuration parameters"""
        # Basic validation
        assert self.sample_rate > 0, "Sample rate must be positive"
        assert self.min_pitch > 0, "Minimum pitch must be positive"
        assert self.max_pitch > self.min_pitch, "Maximum pitch must be greater than minimum"
        
        # YIN validation
        assert 0 < self.yin_threshold < 1, "YIN threshold must be between 0 and 1"
        assert self.yin_window_size > 0, "YIN window size must be positive"
        
        # Weight validation
        assert all(w >= 0 for w in [self.cepstral_weight, self.autocorr_weight, self.yin_weight]), \
            "Method weights must be non-negative"
        
        # Confidence validation
        assert 0 <= self.min_confidence_threshold < self.medium_confidence_threshold < self.high_confidence_threshold <= 1, \
            "Confidence thresholds must be properly ordered between 0 and 1"
        
        # Filter validation
        if self.enable_preprocessing:
            assert self.highpass_frequency > 0, "Highpass frequency must be positive"
            assert self.highpass_order > 0, "Filter order must be positive"
            assert self.highpass_frequency < self.sample_rate / 2, "Highpass frequency must be below Nyquist"
    
    def adjust_for_sample_rate(self):
        """Automatically adjust parameters based on sample rate"""
        # Adjust window sizes based on sample rate
        if self.sample_rate != 48000:
            scale_factor = self.sample_rate / 48000
            self.yin_window_size = int(self.yin_window_size * scale_factor)
            self.cepstral_window_size = int(self.cepstral_window_size * scale_factor)
            
        # Ensure window sizes are reasonable
        self.yin_window_size = max(512, min(4096, self.yin_window_size))
        self.cepstral_window_size = max(1024, min(8192, self.cepstral_window_size))
        
        # Ensure windows are large enough for pitch range
        min_period = int(self.sample_rate / self.max_pitch)
        max_period = int(self.sample_rate / self.min_pitch)
        
        self.yin_window_size = max(self.yin_window_size, max_period * 2)
        self.cepstral_window_size = max(self.cepstral_window_size, max_period * 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_to_file(self, filepath: Path):
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PitchDetectionConfig':
        """Load configuration from JSON file"""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.validate()
        self.adjust_for_sample_rate()
    
    def get_processing_params(self) -> Dict[str, Any]:
        """Get parameters relevant for processing"""
        return {
            'min_pitch': self.min_pitch,
            'max_pitch': self.max_pitch,
            'min_period': int(self.sample_rate / self.max_pitch),
            'max_period': int(self.sample_rate / self.min_pitch),
            'enable_preprocessing': self.enable_preprocessing,
            'highpass_frequency': self.highpass_frequency,
            'highpass_order': self.highpass_order,
            'enable_compression': self.enable_compression,
            'compression_threshold': self.compression_threshold,
            'compression_ratio': self.compression_ratio,
        }
    
    def get_method_weights(self) -> Dict[str, float]:
        """Get normalized method weights"""
        total = self.cepstral_weight + self.autocorr_weight + self.yin_weight
        if total == 0:
            return {'cepstral': 0.33, 'autocorr': 0.33, 'yin': 0.34}
        return {
            'cepstral': self.cepstral_weight / total,
            'autocorr': self.autocorr_weight / total,
            'yin': self.yin_weight / total
        }


# Preset configurations for different use cases
PRESET_CONFIGS = {
    'low_latency': PitchDetectionConfig(
        profile_name='low_latency',
        yin_window_size=512,
        cepstral_window_size=1024,
        enable_preprocessing=False,
        enable_pitch_smoothing=False,
        use_memory_pool=True,
        enable_threading=False,
        yin_use_vectorization=True,
        cache_filter_coefficients=True
    ),
    
    'high_accuracy': PitchDetectionConfig(
        profile_name='high_accuracy',
        yin_window_size=2048,
        cepstral_window_size=4096,
        yin_threshold=0.1,
        enable_preprocessing=True,
        enable_compression=True,
        enable_pitch_smoothing=True,
        smoothing_factor=0.3,
        high_confidence_threshold=0.85,
        min_confidence_threshold=0.2
    ),
    
    'balanced': PitchDetectionConfig(
        profile_name='balanced',
        # Default values are already balanced
    ),
    
    'voice_optimized': PitchDetectionConfig(
        profile_name='voice_optimized',
        min_pitch=80.0,  # Lower limit for adult male voice
        max_pitch=1000.0,  # Upper limit for soprano
        highpass_frequency=70.0,
        enable_compression=True,
        compression_threshold=0.08,
        yin_weight=1.2,  # YIN is good for voice
        cepstral_weight=0.9,
        autocorr_weight=0.7,
        enable_pitch_smoothing=True
    ),
    
    'music_optimized': PitchDetectionConfig(
        profile_name='music_optimized',
        min_pitch=40.0,  # Lower for bass instruments
        max_pitch=4000.0,  # Higher for harmonics
        highpass_frequency=30.0,
        enable_compression=True,
        compression_threshold=0.15,
        cepstral_weight=1.0,
        autocorr_weight=1.0,
        yin_weight=1.0,
        min_confidence_threshold=0.25  # More lenient for complex music
    )
}


def get_preset_config(preset_name: str) -> PitchDetectionConfig:
    """Get a preset configuration by name"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset_name]


def list_presets() -> List[str]:
    """List available preset configurations"""
    return list(PRESET_CONFIGS.keys())


class ConfigurationManager:
    """Manages configuration updates and persistence"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / '.omega4' / 'pitch_detection_config.json'
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_config = self.load_or_create_config()
    
    def load_or_create_config(self) -> PitchDetectionConfig:
        """Load existing config or create default"""
        if self.config_path.exists():
            try:
                return PitchDetectionConfig.load_from_file(self.config_path)
            except Exception as e:
                print(f"Error loading config: {e}. Using default.")
        return PitchDetectionConfig()
    
    def save_current_config(self):
        """Save current configuration to file"""
        self.current_config.save_to_file(self.config_path)
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration"""
        self.current_config = get_preset_config(preset_name)
        self.save_current_config()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update current configuration"""
        self.current_config.update_from_dict(updates)
        self.save_current_config()
    
    def get_config(self) -> PitchDetectionConfig:
        """Get current configuration"""
        return self.current_config