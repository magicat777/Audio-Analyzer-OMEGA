"""
Performance Profiling Panel for OMEGA-4 Audio Analyzer
Real-time monitoring of GPU, CPU, and panel update performance
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import time
import psutil
import threading

# Try to import GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except:
    NVIDIA_AVAILABLE = False


class PerformanceProfiler:
    """Tracks performance metrics for the audio analyzer"""
    
    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.target_frame_time = 1000.0 / target_fps  # ms
        
        # Frame timing
        self.frame_times = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.fps_history = deque(maxlen=300)
        self.last_frame_time = time.perf_counter()
        
        # Panel timing
        self.panel_times = {}  # panel_name -> deque of times
        self.panel_order = []  # Order of panel updates
        
        # System metrics
        self.cpu_usage = deque(maxlen=60)  # 1 second
        self.memory_usage = deque(maxlen=60)
        self.gpu_usage = deque(maxlen=60)
        self.gpu_memory = deque(maxlen=60)
        
        # Performance thresholds
        self.warning_threshold = self.target_frame_time * 0.8  # 80% of budget
        self.critical_threshold = self.target_frame_time  # 100% of budget
        
        # Audio latency metrics
        self.audio_latencies = {
            'input_latency': deque(maxlen=60),      # Audio capture latency
            'processing_latency': deque(maxlen=60),  # FFT processing time
            'output_latency': deque(maxlen=60),      # Total output latency
            'buffer_size': 512,                      # Audio buffer size in samples
            'sample_rate': 48000,                    # Sample rate
            'fft_size': 2048                         # FFT window size
        }
        
        # System monitor thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        # GPU handle
        self.gpu_handle = None
        if NVIDIA_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
    
    def _monitor_system(self):
        """Background thread to monitor system resources"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                mem = psutil.virtual_memory()
                self.memory_usage.append(mem.percent)
                
                # GPU metrics if available
                if NVIDIA_AVAILABLE and self.gpu_handle:
                    try:
                        # GPU utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        self.gpu_usage.append(util.gpu)
                        
                        # GPU memory
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                        self.gpu_memory.append(gpu_mem_percent)
                    except:
                        pass
                
                time.sleep(0.1)  # 10 Hz update rate
                
            except Exception as e:
                print(f"Performance monitor error: {e}")
                time.sleep(1)
    
    def start_frame(self):
        """Mark the start of a new frame"""
        current_time = time.perf_counter()
        
        # Calculate frame time
        if hasattr(self, 'last_frame_time'):
            frame_time = (current_time - self.last_frame_time) * 1000  # ms
            self.frame_times.append(frame_time)
            
            # Calculate FPS
            if frame_time > 0:
                fps = 1000.0 / frame_time
                self.fps_history.append(fps)
        
        self.last_frame_time = current_time
        self.current_frame_start = current_time
        self.panel_order.clear()
    
    def start_panel(self, panel_name: str):
        """Mark the start of a panel update"""
        if panel_name not in self.panel_times:
            self.panel_times[panel_name] = deque(maxlen=60)
        
        return time.perf_counter()
    
    def end_panel(self, panel_name: str, start_time: float):
        """Mark the end of a panel update"""
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        self.panel_times[panel_name].append(elapsed)
        self.panel_order.append((panel_name, elapsed))
    
    def update_audio_latency(self, latency_type: str, value_ms: float):
        """Update audio latency metrics"""
        if latency_type in self.audio_latencies and isinstance(self.audio_latencies[latency_type], deque):
            self.audio_latencies[latency_type].append(value_ms)
    
    def update_audio_config(self, buffer_size: int = None, sample_rate: int = None, fft_size: int = None):
        """Update audio configuration parameters"""
        if buffer_size is not None:
            self.audio_latencies['buffer_size'] = buffer_size
        if sample_rate is not None:
            self.audio_latencies['sample_rate'] = sample_rate
        if fft_size is not None:
            self.audio_latencies['fft_size'] = fft_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = {
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'min_fps': min(self.fps_history) if self.fps_history else 0,
            'max_fps': max(self.fps_history) if self.fps_history else 0,
            
            'current_frame_time': self.frame_times[-1] if self.frame_times else 0,
            'avg_frame_time': np.mean(self.frame_times) if self.frame_times else 0,
            'max_frame_time': max(self.frame_times) if self.frame_times else 0,
            
            'cpu_usage': self.cpu_usage[-1] if self.cpu_usage else 0,
            'memory_usage': self.memory_usage[-1] if self.memory_usage else 0,
            'gpu_usage': self.gpu_usage[-1] if self.gpu_usage else 0,
            'gpu_memory': self.gpu_memory[-1] if self.gpu_memory else 0,
            
            'panel_stats': {},
            
            # Audio latency stats
            'audio_buffer_ms': (self.audio_latencies['buffer_size'] / self.audio_latencies['sample_rate']) * 1000 if self.audio_latencies['sample_rate'] > 0 else 0,
            'fft_window_ms': (self.audio_latencies['fft_size'] / self.audio_latencies['sample_rate']) * 1000 if self.audio_latencies['sample_rate'] > 0 else 0,
            'input_latency': self.audio_latencies['input_latency'][-1] if self.audio_latencies['input_latency'] else 0,
            'processing_latency': self.audio_latencies['processing_latency'][-1] if self.audio_latencies['processing_latency'] else 0,
            'total_latency': 0  # Will calculate below
        }
        
        # Calculate total latency
        stats['total_latency'] = stats['audio_buffer_ms'] + stats['fft_window_ms'] + stats['processing_latency']
        
        # Panel statistics
        for panel_name, times in self.panel_times.items():
            if times:
                stats['panel_stats'][panel_name] = {
                    'current': times[-1],
                    'avg': np.mean(times),
                    'max': max(times),
                    'min': min(times)
                }
        
        return stats
    
    def get_bottlenecks(self) -> List[Tuple[str, float]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check panels exceeding time budget
        time_budget_per_panel = self.target_frame_time / max(len(self.panel_times), 1)
        
        for panel_name, times in self.panel_times.items():
            if times:
                avg_time = np.mean(times)
                if avg_time > time_budget_per_panel:
                    bottlenecks.append((panel_name, avg_time))
        
        # Sort by time descending
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        return bottlenecks
    
    def shutdown(self):
        """Clean up resources"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)


class PerformanceProfilerPanel:
    """Visual performance monitoring panel"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        
        # Display settings
        self.graph_height = 100
        self.graph_padding = 10
        self.text_height = 20
        
        # Colors
        self.bg_color = (30, 30, 40)
        self.grid_color = (50, 50, 60)
        self.text_color = (200, 200, 200)
        self.fps_color = (100, 255, 100)
        self.warning_color = (255, 255, 100)
        self.critical_color = (255, 100, 100)
        
        # Fonts
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
        # Visibility
        self.visible = False
    
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_large = fonts.get('large')
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
    
    def toggle_visibility(self):
        """Toggle panel visibility"""
        self.visible = not self.visible
    
    def update(self):
        """Update performance metrics"""
        # Profiler updates itself via start_frame/end_frame calls
        pass
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw the performance profiler panel"""
        if not self.visible:
            return
        
        # Background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(240)
        overlay.fill(self.bg_color)
        screen.blit(overlay, (x, y))
        
        # Border
        pygame.draw.rect(screen, self.text_color, (x, y, width, height), 2)
        
        current_y = y + 10
        
        # Title
        if self.font_medium:
            title = self.font_medium.render("Performance Monitor", True, self.text_color)
            screen.blit(title, (x + 10, current_y))
            current_y += 30
        
        # Get stats
        stats = self.profiler.get_stats()
        
        # FPS Display
        if self.font_large and self.font_small:
            fps = stats['current_fps']
            fps_color = (
                self.fps_color if fps >= 60 else
                self.warning_color if fps >= 50 else
                self.critical_color
            )
            
            fps_text = self.font_large.render(f"{fps:.1f} FPS", True, fps_color)
            screen.blit(fps_text, (x + 10, current_y))
            
            # Frame time
            frame_time = stats['current_frame_time']
            time_text = self.font_small.render(f"{frame_time:.1f}ms", True, self.text_color)
            screen.blit(time_text, (x + 150, current_y + 10))
            current_y += 40
        
        # FPS Graph
        graph_y = current_y
        self._draw_fps_graph(screen, x + 10, graph_y, width - 20, self.graph_height)
        current_y += self.graph_height + 20
        
        # System Stats
        if self.font_small:
            # CPU and Memory
            cpu_text = f"CPU: {stats['cpu_usage']:.1f}%"
            mem_text = f"RAM: {stats['memory_usage']:.1f}%"
            
            cpu_surf = self.font_small.render(cpu_text, True, self.text_color)
            mem_surf = self.font_small.render(mem_text, True, self.text_color)
            
            screen.blit(cpu_surf, (x + 10, current_y))
            screen.blit(mem_surf, (x + 100, current_y))
            current_y += 20
            
            # GPU if available
            if stats['gpu_usage'] > 0:
                gpu_text = f"GPU: {stats['gpu_usage']:.1f}%"
                gpu_mem_text = f"VRAM: {stats['gpu_memory']:.1f}%"
                
                gpu_surf = self.font_small.render(gpu_text, True, self.text_color)
                gpu_mem_surf = self.font_small.render(gpu_mem_text, True, self.text_color)
                
                screen.blit(gpu_surf, (x + 10, current_y))
                screen.blit(gpu_mem_surf, (x + 100, current_y))
                current_y += 20
        
        # Audio Latency Section
        if self.font_small:
            # Title
            latency_title = self.font_small.render("Audio Latency:", True, self.text_color)
            screen.blit(latency_title, (x + 10, current_y))
            current_y += 20
            
            # Buffer and FFT window
            buffer_text = f"Buffer: {stats['audio_buffer_ms']:.1f}ms"
            fft_text = f"FFT: {stats['fft_window_ms']:.1f}ms"
            
            buffer_surf = self.font_small.render(buffer_text, True, self.text_color)
            fft_surf = self.font_small.render(fft_text, True, self.text_color)
            
            screen.blit(buffer_surf, (x + 20, current_y))
            screen.blit(fft_surf, (x + 120, current_y))
            current_y += 20
            
            # Processing and total latency
            proc_text = f"Processing: {stats['processing_latency']:.1f}ms"
            total_text = f"Total: {stats['total_latency']:.1f}ms"
            
            # Color code total latency
            total_color = (
                self.fps_color if stats['total_latency'] < 50 else
                self.warning_color if stats['total_latency'] < 100 else
                self.critical_color
            )
            
            proc_surf = self.font_small.render(proc_text, True, self.text_color)
            total_surf = self.font_small.render(total_text, True, total_color)
            
            screen.blit(proc_surf, (x + 20, current_y))
            screen.blit(total_surf, (x + 140, current_y))
            current_y += 20
        
        current_y += 10
        
        # Panel Timings
        if self.font_small and stats['panel_stats']:
            # Title
            panel_title = self.font_small.render("Panel Timings (ms):", True, self.text_color)
            screen.blit(panel_title, (x + 10, current_y))
            current_y += 20
            
            # Sort panels by time
            sorted_panels = sorted(
                stats['panel_stats'].items(),
                key=lambda x: x[1]['current'],
                reverse=True
            )
            
            # Show top 5 panels
            for i, (panel_name, panel_stats) in enumerate(sorted_panels[:5]):
                if current_y > y + height - 30:
                    break
                
                time_ms = panel_stats['current']
                time_color = (
                    self.critical_color if time_ms > 2.0 else
                    self.warning_color if time_ms > 1.0 else
                    self.text_color
                )
                
                # Panel name and time
                text = f"  {panel_name}: {time_ms:.2f}"
                text_surf = self.font_small.render(text, True, time_color)
                screen.blit(text_surf, (x + 10, current_y))
                current_y += 18
        
        # Bottleneck Warning
        bottlenecks = self.profiler.get_bottlenecks()
        if bottlenecks and self.font_small:
            current_y += 10
            warning = self.font_small.render("⚠️ Bottlenecks:", True, self.critical_color)
            screen.blit(warning, (x + 10, current_y))
            current_y += 20
            
            for panel, time_ms in bottlenecks[:2]:
                if current_y > y + height - 20:
                    break
                text = f"  {panel}: {time_ms:.1f}ms"
                text_surf = self.font_small.render(text, True, self.warning_color)
                screen.blit(text_surf, (x + 10, current_y))
                current_y += 18
    
    def _draw_fps_graph(self, screen: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw FPS history graph"""
        # Background
        pygame.draw.rect(screen, (20, 20, 30), (x, y, width, height))
        
        # Grid lines
        for i in range(5):
            grid_y = y + (height * i // 4)
            pygame.draw.line(screen, self.grid_color, (x, grid_y), (x + width, grid_y), 1)
        
        # FPS data
        fps_history = list(self.profiler.fps_history)
        if len(fps_history) < 2:
            return
        
        # Scale to graph
        max_fps = 120
        points = []
        
        for i, fps in enumerate(fps_history):
            px = x + (i * width // len(fps_history))
            py = y + height - int((fps / max_fps) * height)
            py = max(y, min(y + height, py))  # Clamp
            points.append((px, py))
        
        # Draw line
        if len(points) >= 2:
            pygame.draw.lines(screen, self.fps_color, False, points, 2)
        
        # Target line (60 FPS)
        target_y = y + height - int((60 / max_fps) * height)
        pygame.draw.line(screen, self.warning_color, 
                        (x, target_y), (x + width, target_y), 1)
        
        # Border
        pygame.draw.rect(screen, self.text_color, (x, y, width, height), 1)
    
    def get_profiler(self) -> PerformanceProfiler:
        """Get the profiler instance for external use"""
        return self.profiler
    
    def shutdown(self):
        """Clean up resources"""
        self.profiler.shutdown()