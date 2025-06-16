"""
Performance monitoring for OMEGA-4 Audio Processing Pipeline
"""

import time
import logging
from collections import deque
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics for audio processing"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.frame_times = deque(maxlen=history_size)
        self.buffer_levels = deque(maxlen=history_size)
        self.frame_start: Optional[float] = None
        
        # Performance thresholds
        self.warning_threshold_ms = 10.0
        self.critical_threshold_ms = 20.0
        
        # Statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.warnings_issued = 0
        
    def start_frame(self):
        """Mark the start of a frame processing"""
        self.frame_start = time.perf_counter()
        
    def end_frame(self, buffer_fill_level: float = 0.0):
        """Mark the end of frame processing and record metrics"""
        if self.frame_start is None:
            logger.warning("end_frame called without start_frame")
            return
            
        frame_time = time.perf_counter() - self.frame_start
        self.frame_times.append(frame_time)
        self.buffer_levels.append(buffer_fill_level)
        self.total_frames += 1
        
        # Check performance
        frame_time_ms = frame_time * 1000
        if frame_time_ms > self.critical_threshold_ms:
            logger.critical(f"Frame processing critically slow: {frame_time_ms:.2f}ms")
            self.dropped_frames += 1
        elif frame_time_ms > self.warning_threshold_ms:
            logger.warning(f"Frame processing slow: {frame_time_ms:.2f}ms")
            self.warnings_issued += 1
            
        self.frame_start = None
        
    def get_statistics(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.frame_times:
            return {
                'avg_frame_time_ms': 0.0,
                'max_frame_time_ms': 0.0,
                'min_frame_time_ms': 0.0,
                'std_frame_time_ms': 0.0,
                'dropped_frame_rate': 0.0,
                'warning_rate': 0.0,
                'avg_buffer_level': 0.0
            }
            
        frame_times_ms = [t * 1000 for t in self.frame_times]
        
        return {
            'avg_frame_time_ms': np.mean(frame_times_ms),
            'max_frame_time_ms': np.max(frame_times_ms),
            'min_frame_time_ms': np.min(frame_times_ms),
            'std_frame_time_ms': np.std(frame_times_ms),
            'dropped_frame_rate': self.dropped_frames / max(1, self.total_frames),
            'warning_rate': self.warnings_issued / max(1, self.total_frames),
            'avg_buffer_level': np.mean(self.buffer_levels) if self.buffer_levels else 0.0
        }
        
    def reset(self):
        """Reset all statistics"""
        self.frame_times.clear()
        self.buffer_levels.clear()
        self.total_frames = 0
        self.dropped_frames = 0
        self.warnings_issued = 0
        self.frame_start = None
        
    def should_reduce_quality(self) -> bool:
        """Check if quality should be reduced due to performance issues"""
        if len(self.frame_times) < 10:
            return False
            
        recent_avg = np.mean(list(self.frame_times)[-10:]) * 1000
        return recent_avg > self.warning_threshold_ms