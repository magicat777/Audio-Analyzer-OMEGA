"""
Parallel Panel Update System for OMEGA-4
Utilizes multiple CPU cores to update panels in parallel
"""

import concurrent.futures
import threading
import time
from typing import Dict, List, Tuple, Any, Callable
from collections import defaultdict
import numpy as np


class ParallelPanelUpdater:
    """Manages parallel updates of visualization panels"""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel updater
        
        Args:
            max_workers: Maximum number of worker threads (None = CPU count)
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.update_times = defaultdict(list)  # Track update times for each panel
        self.panel_groups = {
            'independent': [],  # Panels that don't depend on others
            'sequential': [],   # Panels that must update in order
            'low_priority': []  # Panels that can skip frames if needed
        }
        
        # Panel dependencies
        self.dependencies = {}
        
        # Performance metrics
        self.frame_times = []
        self.max_frame_time = 16.67  # Target 60 FPS
        
    def register_panel(self, panel_name: str, 
                      update_func: Callable,
                      group: str = 'independent',
                      dependencies: List[str] = None):
        """
        Register a panel for parallel updates
        
        Args:
            panel_name: Unique identifier for the panel
            update_func: Function to call for updates
            group: Update group ('independent', 'sequential', 'low_priority')
            dependencies: List of panel names this panel depends on
        """
        self.panel_groups[group].append({
            'name': panel_name,
            'update_func': update_func,
            'last_update': 0,
            'update_interval': 1  # Frames between updates
        })
        
        if dependencies:
            self.dependencies[panel_name] = dependencies
            
    def update_all_panels(self, shared_data: Dict[str, Any]) -> Dict[str, concurrent.futures.Future]:
        """
        Update all panels in parallel where possible
        
        Args:
            shared_data: Data shared between panels (FFT results, audio data, etc.)
            
        Returns:
            Dict of panel_name -> Future for tracking completion
        """
        start_time = time.time()
        futures = {}
        completed_panels = set()
        
        # First, update independent panels in parallel
        independent_futures = []
        for panel_info in self.panel_groups['independent']:
            if self._should_update_panel(panel_info):
                future = self.executor.submit(
                    self._update_panel_safe,
                    panel_info,
                    shared_data,
                    completed_panels
                )
                futures[panel_info['name']] = future
                independent_futures.append(future)
        
        # Wait for independent panels to complete
        concurrent.futures.wait(independent_futures, timeout=self.max_frame_time * 0.5)
        
        # Update panels with dependencies
        for panel_info in self.panel_groups['sequential']:
            if self._should_update_panel(panel_info):
                # Check if dependencies are satisfied
                deps = self.dependencies.get(panel_info['name'], [])
                if all(dep in completed_panels for dep in deps):
                    future = self.executor.submit(
                        self._update_panel_safe,
                        panel_info,
                        shared_data,
                        completed_panels
                    )
                    futures[panel_info['name']] = future
        
        # Update low priority panels if we have time
        elapsed = (time.time() - start_time) * 1000
        if elapsed < self.max_frame_time * 0.8:  # If we have 20% time buffer
            for panel_info in self.panel_groups['low_priority']:
                if self._should_update_panel(panel_info):
                    future = self.executor.submit(
                        self._update_panel_safe,
                        panel_info,
                        shared_data,
                        completed_panels
                    )
                    futures[panel_info['name']] = future
        
        # Record frame time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
            
        return futures
    
    def _should_update_panel(self, panel_info: Dict) -> bool:
        """Check if panel should update this frame"""
        current_frame = int(time.time() * 60)  # Approximate frame counter
        frames_since_update = current_frame - panel_info['last_update']
        
        if frames_since_update >= panel_info['update_interval']:
            panel_info['last_update'] = current_frame
            return True
        return False
    
    def _update_panel_safe(self, panel_info: Dict, shared_data: Dict, 
                          completed_panels: set) -> Tuple[str, float]:
        """Safely update a panel with error handling"""
        panel_name = panel_info['name']
        start_time = time.time()
        
        try:
            # Call the update function
            panel_info['update_func'](shared_data)
            
            # Mark as completed
            completed_panels.add(panel_name)
            
            # Record update time
            update_time = (time.time() - start_time) * 1000
            self.update_times[panel_name].append(update_time)
            if len(self.update_times[panel_name]) > 60:
                self.update_times[panel_name].pop(0)
                
            return panel_name, update_time
            
        except Exception as e:
            print(f"Error updating panel {panel_name}: {e}")
            return panel_name, -1
    
    def adjust_update_frequencies(self):
        """Dynamically adjust panel update frequencies based on performance"""
        if len(self.frame_times) < 10:
            return
            
        avg_frame_time = np.mean(self.frame_times[-10:])
        
        # If we're missing our frame time target
        if avg_frame_time > self.max_frame_time:
            # Reduce update frequency for low priority panels
            for panel_info in self.panel_groups['low_priority']:
                panel_info['update_interval'] = min(10, panel_info['update_interval'] + 1)
                
            # Also reduce some independent panels if needed
            if avg_frame_time > self.max_frame_time * 1.2:
                for panel_info in self.panel_groups['independent']:
                    if panel_info['name'] not in ['spectrum', 'bass_zoom']:  # Keep critical panels fast
                        panel_info['update_interval'] = min(6, panel_info['update_interval'] + 1)
        
        # If we have headroom, increase update frequencies
        elif avg_frame_time < self.max_frame_time * 0.7:
            for group in self.panel_groups.values():
                for panel_info in group:
                    panel_info['update_interval'] = max(1, panel_info['update_interval'] - 1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'max_frame_time': max(self.frame_times) if self.frame_times else 0,
            'panel_update_times': {}
        }
        
        for panel_name, times in self.update_times.items():
            if times:
                stats['panel_update_times'][panel_name] = {
                    'avg': np.mean(times),
                    'max': max(times),
                    'min': min(times)
                }
                
        return stats
    
    def shutdown(self):
        """Shutdown the thread pool"""
        self.executor.shutdown(wait=True)


class OptimizedDataSharing:
    """Manages shared data between panels to avoid redundant calculations"""
    
    def __init__(self):
        self.shared_data = {}
        self.data_lock = threading.RLock()
        self.computed_flags = set()
        
    def set_base_data(self, audio_data: np.ndarray, sample_rate: int):
        """Set base audio data for the frame"""
        with self.data_lock:
            self.shared_data['audio_data'] = audio_data
            self.shared_data['sample_rate'] = sample_rate
            self.computed_flags.clear()  # Reset flags for new frame
            
    def get_or_compute(self, key: str, compute_func: Callable) -> Any:
        """Get data if available, or compute and cache it"""
        with self.data_lock:
            if key in self.computed_flags:
                return self.shared_data.get(key)
            
            # Compute the data
            result = compute_func(self.shared_data)
            self.shared_data[key] = result
            self.computed_flags.add(key)
            
            return result
    
    def get_fft_data(self, fft_size: int, window_type: str = 'hann') -> Dict[str, np.ndarray]:
        """Get FFT data, computing only if necessary"""
        key = f'fft_{fft_size}_{window_type}'
        
        def compute_fft(data):
            from .gpu_accelerated_fft import get_gpu_fft_processor
            
            audio = data['audio_data']
            if len(audio) >= fft_size:
                chunk = audio[-fft_size:]
            else:
                chunk = np.pad(audio, (0, fft_size - len(audio)))
                
            gpu_fft = get_gpu_fft_processor()
            magnitude, complex_fft = gpu_fft.compute_fft(chunk, window_type, return_complex=True)
            freqs = np.fft.rfftfreq(fft_size, 1/data['sample_rate'])
            
            return {
                'magnitude': magnitude,
                'complex': complex_fft,
                'freqs': freqs
            }
        
        return self.get_or_compute(key, compute_fft)
    
    def clear(self):
        """Clear all shared data"""
        with self.data_lock:
            self.shared_data.clear()
            self.computed_flags.clear()