"""
Adaptive update manager to reduce panel update frequency
"""

class AdaptiveUpdater:
    """Manages update frequencies for different panels to improve performance"""
    
    def __init__(self):
        # Update divisors - higher number = less frequent updates
        self.update_intervals = {
            'spectrum': 1,              # Every frame (60 FPS)
            'meters': 2,                # Every 2 frames (30 FPS)
            'vu_meters': 2,             # Every 2 frames (30 FPS)
            'bass_zoom': 1,             # Every frame (60 FPS)
            'harmonic': 3,              # Every 3 frames (20 FPS)
            'pitch_detection': 4,       # Every 4 frames (15 FPS)
            'chromagram': 3,            # Every 3 frames (20 FPS)
            'genre_classification': 6,  # Every 6 frames (10 FPS)
            'room_analysis': 10,        # Every 10 frames (6 FPS)
            'integrated_music': 4,      # Every 4 frames (15 FPS)
        }
        
        self.frame_count = 0
        self.last_update = {}
        
        # Initialize last update times
        for panel in self.update_intervals:
            self.last_update[panel] = 0
    
    def should_update(self, panel_name: str) -> bool:
        """Check if a panel should be updated this frame"""
        if panel_name not in self.update_intervals:
            return True  # Default to updating if not specified
        
        interval = self.update_intervals[panel_name]
        
        # Check if enough frames have passed
        if self.frame_count - self.last_update[panel_name] >= interval:
            self.last_update[panel_name] = self.frame_count
            return True
        
        return False
    
    def tick(self):
        """Increment frame counter"""
        self.frame_count += 1
    
    def set_interval(self, panel_name: str, interval: int):
        """Dynamically adjust update interval for a panel"""
        if interval < 1:
            interval = 1
        self.update_intervals[panel_name] = interval
    
    def get_effective_fps(self, panel_name: str, base_fps: int = 60) -> float:
        """Get the effective FPS for a panel given the base FPS"""
        if panel_name not in self.update_intervals:
            return base_fps
        
        return base_fps / self.update_intervals[panel_name]