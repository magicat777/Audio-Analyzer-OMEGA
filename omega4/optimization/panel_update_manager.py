"""
Panel Update Manager - Intelligent update scheduling for performance optimization
"""

import time
from typing import Dict, Set, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class UpdatePriority(Enum):
    """Panel update priority levels"""
    CRITICAL = 1  # Every frame (60 FPS)
    HIGH = 2      # Every 2 frames (30 FPS)
    MEDIUM = 4    # Every 4 frames (15 FPS)
    LOW = 8       # Every 8 frames (7.5 FPS)
    BACKGROUND = 16  # Every 16 frames (3.75 FPS)


@dataclass
class PanelConfig:
    """Configuration for panel update scheduling"""
    name: str
    priority: UpdatePriority
    can_skip_frames: bool = True
    adaptive: bool = True  # Adjust rate based on performance
    min_update_interval: float = 0.016  # Minimum seconds between updates


class PanelUpdateManager:
    """Manages panel update scheduling for optimal performance"""
    
    # Default panel configurations
    DEFAULT_CONFIGS = {
        # Critical panels - visual feedback needs to be immediate
        'spectrum_display': PanelConfig('spectrum_display', UpdatePriority.CRITICAL, can_skip_frames=False),
        'vu_meters': PanelConfig('vu_meters', UpdatePriority.CRITICAL, can_skip_frames=False),
        'beat_detection': PanelConfig('beat_detection', UpdatePriority.CRITICAL, can_skip_frames=False),
        
        # High priority - important real-time feedback
        'professional_meters': PanelConfig('professional_meters', UpdatePriority.HIGH),
        'bass_zoom': PanelConfig('bass_zoom', UpdatePriority.HIGH),
        'transient_detection': PanelConfig('transient_detection', UpdatePriority.HIGH),
        
        # Medium priority - can update less frequently
        'chromagram': PanelConfig('chromagram', UpdatePriority.MEDIUM),
        'pitch_detection': PanelConfig('pitch_detection', UpdatePriority.MEDIUM),
        'voice_detection': PanelConfig('voice_detection', UpdatePriority.MEDIUM),
        'phase_correlation': PanelConfig('phase_correlation', UpdatePriority.MEDIUM),
        
        # Low priority - analysis panels
        'harmonic_analysis': PanelConfig('harmonic_analysis', UpdatePriority.LOW),
        'genre_classification': PanelConfig('genre_classification', UpdatePriority.LOW),
        'integrated_music': PanelConfig('integrated_music', UpdatePriority.LOW),
        'room_analysis': PanelConfig('room_analysis', UpdatePriority.LOW),
        
        # Background priority - heavy computation
        'spectrogram_waterfall': PanelConfig('spectrogram_waterfall', UpdatePriority.BACKGROUND),
        'frequency_band_tracker': PanelConfig('frequency_band_tracker', UpdatePriority.BACKGROUND),
        'performance_profiler': PanelConfig('performance_profiler', UpdatePriority.BACKGROUND),
    }
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # Panel configurations
        self.panel_configs: Dict[str, PanelConfig] = self.DEFAULT_CONFIGS.copy()
        
        # Update tracking
        self.frame_counter = 0
        self.last_update_frame: Dict[str, int] = {}
        self.last_update_time: Dict[str, float] = {}
        self.update_times: Dict[str, float] = {}  # Track actual update duration
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 60
        self.performance_mode = False  # Reduce quality when struggling
        
        # Active panels
        self.active_panels: Set[str] = set()
        
        # Update callbacks
        self.update_callbacks: Dict[str, Callable] = {}
        
    def register_panel(self, panel_id: str, update_callback: Callable, 
                      config: Optional[PanelConfig] = None):
        """Register a panel with its update callback"""
        if config:
            self.panel_configs[panel_id] = config
        elif panel_id not in self.panel_configs:
            # Default to medium priority if not specified
            self.panel_configs[panel_id] = PanelConfig(panel_id, UpdatePriority.MEDIUM)
        
        self.update_callbacks[panel_id] = update_callback
        self.last_update_frame[panel_id] = -999  # Force first update
        self.last_update_time[panel_id] = 0
        
    def activate_panel(self, panel_id: str):
        """Activate a panel for updates"""
        self.active_panels.add(panel_id)
        
    def deactivate_panel(self, panel_id: str):
        """Deactivate a panel from updates"""
        self.active_panels.discard(panel_id)
        
    def should_update_panel(self, panel_id: str) -> bool:
        """Check if a panel should update this frame"""
        if panel_id not in self.active_panels:
            return False
            
        config = self.panel_configs.get(panel_id)
        if not config:
            return True  # Update if no config
            
        # Check frame-based update schedule
        frames_since_update = self.frame_counter - self.last_update_frame.get(panel_id, -999)
        priority_frames = config.priority.value
        
        # Adjust for performance mode
        if self.performance_mode and config.adaptive:
            priority_frames *= 2  # Update half as often in performance mode
            
        if frames_since_update < priority_frames:
            return False
            
        # Check time-based minimum interval
        time_since_update = time.time() - self.last_update_time.get(panel_id, 0)
        if time_since_update < config.min_update_interval:
            return False
            
        return True
        
    def update_panel(self, panel_id: str, *args, **kwargs) -> bool:
        """Update a panel if it's scheduled"""
        if not self.should_update_panel(panel_id):
            return False
            
        callback = self.update_callbacks.get(panel_id)
        if not callback:
            return False
            
        # Track update time
        start_time = time.time()
        
        try:
            callback(*args, **kwargs)
            
            # Record successful update
            self.last_update_frame[panel_id] = self.frame_counter
            self.last_update_time[panel_id] = start_time
            
            # Track update duration
            update_duration = time.time() - start_time
            self.update_times[panel_id] = update_duration
            
            return True
            
        except Exception as e:
            print(f"Error updating panel {panel_id}: {e}")
            return False
            
    def batch_update_panels(self, panel_data: Dict[str, tuple]) -> Dict[str, bool]:
        """Update multiple panels with their respective data"""
        results = {}
        
        # Sort panels by priority for update order
        sorted_panels = sorted(
            panel_data.keys(),
            key=lambda p: self.panel_configs.get(p, PanelConfig(p, UpdatePriority.MEDIUM)).priority.value
        )
        
        for panel_id in sorted_panels:
            if panel_id in panel_data:
                args, kwargs = panel_data[panel_id]
                results[panel_id] = self.update_panel(panel_id, *args, **kwargs)
                
        return results
        
    def end_frame(self, frame_time: float):
        """Called at the end of each frame to update performance metrics"""
        self.frame_counter += 1
        
        # Track frame times
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
            
        # Update performance mode
        if len(self.frame_times) >= 10:
            avg_frame_time = sum(self.frame_times[-10:]) / 10
            
            # Enter performance mode if struggling
            if avg_frame_time > self.target_frame_time * 1.5:  # 50% over target
                if not self.performance_mode:
                    self.performance_mode = True
                    print("Entering performance mode - reducing update rates")
            elif avg_frame_time < self.target_frame_time * 1.2 and self.performance_mode:
                # Exit performance mode when stable
                self.performance_mode = False
                print("Exiting performance mode - normal update rates")
                
    def get_panel_statistics(self) -> Dict[str, Dict]:
        """Get update statistics for all panels"""
        stats = {}
        
        for panel_id in self.active_panels:
            config = self.panel_configs.get(panel_id)
            if not config:
                continue
                
            frames_since_update = self.frame_counter - self.last_update_frame.get(panel_id, 0)
            update_time = self.update_times.get(panel_id, 0)
            
            stats[panel_id] = {
                'priority': config.priority.name,
                'frames_since_update': frames_since_update,
                'last_update_ms': update_time * 1000,
                'target_fps': self.target_fps / config.priority.value,
                'actual_fps': self.target_fps / max(frames_since_update, 1) if frames_since_update > 0 else 0
            }
            
        return stats
        
    def adjust_panel_priority(self, panel_id: str, new_priority: UpdatePriority):
        """Dynamically adjust panel priority"""
        if panel_id in self.panel_configs:
            self.panel_configs[panel_id].priority = new_priority