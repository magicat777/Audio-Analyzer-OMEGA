"""
Configuration classes for OMEGA-4 Audio Processing Pipeline
Enhanced with performance optimizations and Linux-specific settings
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PipelineConfig:
    """Configuration for audio processing pipeline"""
    # Core audio settings
    sample_rate: int = 48000
    num_bands: int = 768
    fft_size: int = 4096
    
    # Gain control
    input_gain: float = 4.0
    auto_gain_enabled: bool = True
    target_lufs: float = -16.0
    gain_history_size: int = 300
    
    # Buffering
    ring_buffer_size: int = 16384  # 4096 * 4
    
    # Smoothing
    smoothing_factor: float = 0.7
    
    # Content detector settings
    content_history_size: int = 30
    voice_confidence_threshold: float = 0.7
    mixed_confidence_threshold: float = 0.4
    
    # Frequency mapping settings
    min_frequency: float = 20.0
    max_frequency: float = 20000.0
    transition_frequency: float = 1000.0
    low_freq_band_ratio: float = 0.5
    
    # Numerical stability
    epsilon: float = 1e-6
    
    # Performance thresholds
    max_frame_time_ms: float = 10.0
    
    # Window function settings
    window_function: str = "hann"  # Options: hann, hamming, blackman, kaiser
    overlap_ratio: float = 0.75    # 75% overlap for smooth updates
    
    # Threading configuration for Ubuntu performance
    max_worker_threads: int = 4
    use_multiprocessing: bool = True
    
    # Latency targets for real-time performance
    target_latency_ms: float = 5.0
    buffer_safety_factor: float = 2.0
    
    # Performance monitoring
    enable_performance_logging: bool = False
    log_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR
    stats_update_interval: float = 1.0  # seconds
    
    # Memory management
    max_memory_usage_mb: float = 512.0
    enable_gc_optimization: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.num_bands <= 0:
            raise ValueError("Number of bands must be positive")
        if self.fft_size <= 0 or (self.fft_size & (self.fft_size - 1)) != 0:
            raise ValueError("FFT size must be power of 2")
        if self.min_frequency >= self.max_frequency:
            raise ValueError("Min frequency must be less than max frequency")
        if self.smoothing_factor < 0 or self.smoothing_factor > 1:
            raise ValueError("Smoothing factor must be between 0 and 1")
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
            
        # Nyquist frequency validation
        nyquist = self.sample_rate / 2
        if self.max_frequency > nyquist:
            raise ValueError(f"Max frequency ({self.max_frequency}) cannot exceed Nyquist frequency ({nyquist})")
            
        # Ring buffer must be multiple of FFT size
        if self.ring_buffer_size % self.fft_size != 0:
            recommended_size = ((self.ring_buffer_size // self.fft_size) + 1) * self.fft_size
            raise ValueError(f"Ring buffer size must be multiple of FFT size. Recommended: {recommended_size}")
        
        # Ensure adequate buffering for smooth operation
        min_buffer_frames = self.fft_size * 3  # Minimum 3 frames
        if self.ring_buffer_size < min_buffer_frames:
            raise ValueError(f"Ring buffer too small. Minimum: {min_buffer_frames}")
            
        # Validate window function
        valid_windows = ["hann", "hamming", "blackman", "kaiser"]
        if self.window_function not in valid_windows:
            raise ValueError(f"Invalid window function. Must be one of: {valid_windows}")
            
        # Validate overlap ratio
        if self.overlap_ratio < 0 or self.overlap_ratio > 0.95:
            raise ValueError("Overlap ratio must be between 0 and 0.95")
            
        # Validate threading configuration
        if self.max_worker_threads < 1:
            raise ValueError("Max worker threads must be at least 1")
            
        # Validate memory limit
        if self.max_memory_usage_mb < 64:
            raise ValueError("Max memory usage must be at least 64 MB")


@dataclass
class LinuxAudioConfig:
    """Linux-specific audio configuration for Ubuntu performance optimization"""
    # ALSA/PulseAudio settings
    audio_backend: str = "alsa"  # Options: alsa, pulse, jack
    device_name: Optional[str] = None  # Auto-detect if None
    
    # Real-time scheduling (requires sudo permissions)
    enable_rt_scheduling: bool = False
    rt_priority: int = 80
    
    # Memory locking for consistent performance
    lock_memory: bool = True
    
    # CPU affinity for dedicated cores
    cpu_affinity: Optional[List[int]] = None  # [0, 1] to use cores 0 and 1
    
    # Buffer tuning for low latency
    periods: int = 2  # Number of periods in buffer
    period_size: int = 256  # Frames per period
    
    def __post_init__(self):
        """Validate Linux audio configuration"""
        valid_backends = ["alsa", "pulse", "jack"]
        if self.audio_backend not in valid_backends:
            raise ValueError(f"Invalid audio backend. Must be one of: {valid_backends}")
            
        if self.rt_priority < 0 or self.rt_priority > 99:
            raise ValueError("RT priority must be between 0 and 99")
            
        if self.periods < 2 or self.periods > 16:
            raise ValueError("Periods must be between 2 and 16")
            
        if self.period_size < 32 or self.period_size > 8192:
            raise ValueError("Period size must be between 32 and 8192")
            
        # Validate CPU affinity
        if self.cpu_affinity is not None:
            import os
            max_cpu = os.cpu_count() - 1
            for cpu in self.cpu_affinity:
                if cpu < 0 or cpu > max_cpu:
                    raise ValueError(f"Invalid CPU core {cpu}. Must be between 0 and {max_cpu}")