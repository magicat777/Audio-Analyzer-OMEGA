"""
Performance Tracking System for OMEGA-4
Provides decorators and utilities for tracking performance metrics
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
from collections import defaultdict, deque
import threading


class PerformanceTracker:
    """Singleton performance tracking system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # Timing data
        self.timings = defaultdict(lambda: deque(maxlen=100))
        self.call_counts = defaultdict(int)
        
        # Frame tracking
        self.frame_start_time = None
        self.frame_end_time = None
        self.frame_number = 0
        
        # Panel tracking
        self.panel_timings = defaultdict(lambda: deque(maxlen=60))
        self.current_panel = None
        self.panel_start_time = None
    
    def start_frame(self):
        """Mark the start of a new frame"""
        self.frame_start_time = time.perf_counter_ns()
        self.frame_number += 1
    
    def end_frame(self):
        """Mark the end of the current frame"""
        if self.frame_start_time is not None:
            self.frame_end_time = time.perf_counter_ns()
            frame_time = (self.frame_end_time - self.frame_start_time) / 1_000_000  # ms
            self.timings['frame_time'].append(frame_time)
    
    def start_panel(self, panel_name: str):
        """Start timing a panel update"""
        self.current_panel = panel_name
        self.panel_start_time = time.perf_counter_ns()
    
    def end_panel(self):
        """End timing the current panel"""
        if self.current_panel and self.panel_start_time:
            elapsed = (time.perf_counter_ns() - self.panel_start_time) / 1_000_000  # ms
            self.panel_timings[self.current_panel].append(elapsed)
            self.current_panel = None
            self.panel_start_time = None
    
    def record_timing(self, operation: str, duration_ms: float):
        """Record a timing measurement"""
        self.timings[operation].append(duration_ms)
        self.call_counts[operation] += 1
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        if operation:
            if operation not in self.timings:
                return {'error': f'No data for operation: {operation}'}
            
            times = list(self.timings[operation])
            if not times:
                return {'error': 'No timing data available'}
            
            return {
                'operation': operation,
                'count': self.call_counts[operation],
                'last': times[-1] if times else 0,
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'total': sum(times)
            }
        else:
            # Return all stats
            all_stats = {}
            for op in self.timings:
                all_stats[op] = self.get_stats(op)
            return all_stats
    
    def get_panel_stats(self) -> Dict[str, Dict[str, float]]:
        """Get panel-specific statistics"""
        panel_stats = {}
        
        for panel_name, times in self.panel_timings.items():
            if times:
                times_list = list(times)
                panel_stats[panel_name] = {
                    'last': times_list[-1],
                    'avg': sum(times_list) / len(times_list),
                    'min': min(times_list),
                    'max': max(times_list),
                    'samples': len(times_list)
                }
        
        return panel_stats
    
    def reset(self):
        """Reset all timing data"""
        self.timings.clear()
        self.call_counts.clear()
        self.panel_timings.clear()
        self.frame_number = 0


# Global tracker instance
_tracker = PerformanceTracker()


def track_time(operation_name: Optional[str] = None):
    """
    Decorator to track function execution time
    
    Usage:
        @track_time("fft_processing")
        def process_fft(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter_ns()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ns = time.perf_counter_ns() - start_time
                elapsed_ms = elapsed_ns / 1_000_000
                _tracker.record_timing(name, elapsed_ms)
        
        return wrapper
    return decorator


def track_panel(panel_name: str):
    """
    Decorator to track panel update time
    
    Usage:
        @track_panel("spectrum_display")
        def update(self, data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tracker.start_panel(panel_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _tracker.end_panel()
        
        return wrapper
    return decorator


class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter_ns()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed_ns = time.perf_counter_ns() - self.start_time
            elapsed_ms = elapsed_ns / 1_000_000
            _tracker.record_timing(self.operation_name, elapsed_ms)


# Convenience functions
def start_frame():
    """Mark the start of a new frame"""
    _tracker.start_frame()


def end_frame():
    """Mark the end of the current frame"""
    _tracker.end_frame()


def get_stats(operation: Optional[str] = None) -> Dict[str, Any]:
    """Get performance statistics"""
    return _tracker.get_stats(operation)


def get_panel_stats() -> Dict[str, Dict[str, float]]:
    """Get panel-specific statistics"""
    return _tracker.get_panel_stats()


def reset():
    """Reset all timing data"""
    _tracker.reset()


def get_tracker() -> PerformanceTracker:
    """Get the global tracker instance"""
    return _tracker