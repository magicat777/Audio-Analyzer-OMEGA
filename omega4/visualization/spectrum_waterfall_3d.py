"""
3D Spectrum Waterfall Visualization using CuPy GPU acceleration
Creates a live 3D spectrogram behind the main spectrum visualizer
"""

import numpy as np
import pygame
import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from collections import deque
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy detected - GPU acceleration available for waterfall")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available - falling back to CPU for waterfall")

@dataclass
class WaterfallSettings:
    """3D Waterfall visualization settings"""
    enabled: bool = True
    max_slices: int = 25  # Maximum number of historical slices to maintain
    slice_interval: float = 0.1  # Time in seconds between new slices
    depth_scale: float = 0.8  # How much each slice scales (0.8 = 20% reduction)
    depth_spacing: float = 8.0  # Pixel spacing between depth layers
    base_alpha: float = 0.9  # Starting transparency for newest slice
    alpha_decay: float = 0.85  # Alpha multiplication per depth layer
    animation_speed: float = 2.0  # Speed of smooth transitions
    enable_interpolation: bool = True  # Smooth interpolation between slices
    color_shift: float = 0.95  # Color intensity shift per layer (darker = further)

class SpectrumWaterfall3D:
    """
    3D Waterfall spectrum visualization using GPU acceleration.
    Creates depth layers of historical spectrum data receding into the background.
    """
    
    def __init__(self, num_bars: int, use_gpu: bool = True):
        """
        Initialize the 3D waterfall system.
        
        Args:
            num_bars: Number of frequency bars in the spectrum
            use_gpu: Whether to use GPU acceleration (CuPy)
        """
        self.num_bars = num_bars
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.settings = WaterfallSettings()
        
        # Spectrum history storage
        if self.use_gpu:
            self.spectrum_history_gpu = []  # List of CuPy arrays
            self.interpolated_slices_gpu = []  # Smooth interpolated slices
        else:
            self.spectrum_history_cpu = []  # Fallback CPU storage
        
        # Timing control
        self.last_slice_time = 0.0
        self.animation_time = 0.0
        
        # Pre-computed depth parameters for performance
        self._depth_scales = np.array([
            self.settings.depth_scale ** i for i in range(self.settings.max_slices)
        ])
        self._depth_alphas = np.array([
            self.settings.base_alpha * (self.settings.alpha_decay ** i) 
            for i in range(self.settings.max_slices)
        ])
        self._depth_positions = np.array([
            i * self.settings.depth_spacing for i in range(self.settings.max_slices)
        ])
        
        # CUDA kernels for GPU processing
        if self.use_gpu:
            self._setup_cuda_kernels()
        
        # Surface cache for rendered layers
        self._surface_cache = {}
        self._cache_valid = False
        
        logger.info(f"SpectrumWaterfall3D initialized: {num_bars} bars, "
                   f"GPU={'enabled' if self.use_gpu else 'disabled'}")
    
    def _setup_cuda_kernels(self):
        """Setup CUDA kernels for GPU processing"""
        if not self.use_gpu:
            return
            
        try:
            # Kernel for smooth interpolation between spectrum slices
            self.interpolation_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void interpolate_spectrum(float* output, 
                                    const float* slice1, 
                                    const float* slice2,
                                    float factor,
                                    int num_bars) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_bars) {
                    output[idx] = slice1[idx] * (1.0f - factor) + slice2[idx] * factor;
                }
            }
            ''', 'interpolate_spectrum')
            
            # Kernel for applying depth effects (scaling, fading)
            self.depth_effects_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void apply_depth_effects(float* output,
                                   const float* input,
                                   float scale_factor,
                                   float alpha_factor,
                                   float color_shift,
                                   int num_bars) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < num_bars) {
                    output[idx] = input[idx] * scale_factor * alpha_factor * color_shift;
                }
            }
            ''', 'apply_depth_effects')
            
            logger.info("CUDA kernels compiled successfully for waterfall")
            
        except Exception as e:
            logger.error(f"Failed to setup CUDA kernels: {e}")
            self.use_gpu = False
    
    def update_spectrum_slice(self, spectrum_data: np.ndarray, current_time: float):
        """
        Update the waterfall with a new spectrum slice.
        
        Args:
            spectrum_data: Current spectrum data (numpy array)
            current_time: Current timestamp in seconds
        """
        # Check if it's time for a new slice
        if current_time - self.last_slice_time < self.settings.slice_interval:
            if self.settings.enable_interpolation:
                self._update_interpolation(current_time)
            return
        
        # Add new slice to history
        if self.use_gpu:
            self._add_slice_gpu(spectrum_data)
        else:
            self._add_slice_cpu(spectrum_data)
        
        self.last_slice_time = current_time
        self._cache_valid = False  # Invalidate surface cache
        
        # Update animation time for smooth transitions
        self.animation_time = current_time
    
    def _add_slice_gpu(self, spectrum_data: np.ndarray):
        """Add a new spectrum slice using GPU storage"""
        try:
            # Convert to CuPy array if needed
            if isinstance(spectrum_data, np.ndarray):
                gpu_slice = cp.array(spectrum_data, dtype=cp.float32)
            else:
                gpu_slice = spectrum_data.astype(cp.float32)
            
            # Add to history
            self.spectrum_history_gpu.append(gpu_slice)
            
            # Remove oldest slice if we exceed max_slices
            if len(self.spectrum_history_gpu) > self.settings.max_slices:
                old_slice = self.spectrum_history_gpu.pop(0)
                del old_slice  # Explicit memory cleanup
            
        except Exception as e:
            logger.error(f"Failed to add GPU slice: {e}")
            # Fall back to CPU if GPU fails
            self.use_gpu = False
            self._add_slice_cpu(spectrum_data)
    
    def _add_slice_cpu(self, spectrum_data: np.ndarray):
        """Add a new spectrum slice using CPU storage (fallback)"""
        # Ensure we have a numpy array
        if hasattr(spectrum_data, 'get'):  # CuPy array
            cpu_slice = spectrum_data.get()
        else:
            cpu_slice = np.array(spectrum_data, dtype=np.float32)
        
        self.spectrum_history_cpu.append(cpu_slice)
        
        # Remove oldest slice if we exceed max_slices
        if len(self.spectrum_history_cpu) > self.settings.max_slices:
            self.spectrum_history_cpu.pop(0)
    
    def _update_interpolation(self, current_time: float):
        """Update smooth interpolation between slices"""
        if not self.settings.enable_interpolation:
            return
        
        # Calculate interpolation factor based on time since last slice
        time_since_slice = current_time - self.last_slice_time
        interp_factor = min(time_since_slice / self.settings.slice_interval, 1.0)
        
        # Apply easing function for smoother animation
        interp_factor = self._ease_in_out_cubic(interp_factor)
        
        # Update animation state
        self.animation_time = current_time
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Cubic easing function for smooth animations"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def render_waterfall_layers(self, screen: pygame.Surface, vis_params: Dict[str, Any], 
                               main_spectrum_data: np.ndarray) -> bool:
        """
        Render the 3D waterfall layers behind the main spectrum.
        
        Args:
            screen: Pygame surface to render onto
            vis_params: Visualization parameters (positions, dimensions)
            main_spectrum_data: Current spectrum data for the main display
            
        Returns:
            bool: True if rendering successful
        """
        try:
            if not self.settings.enabled:
                return True
            
            # Get spectrum history
            if self.use_gpu and len(self.spectrum_history_gpu) > 0:
                return self._render_layers_gpu(screen, vis_params, main_spectrum_data)
            elif len(self.spectrum_history_cpu) > 0:
                return self._render_layers_cpu(screen, vis_params, main_spectrum_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Waterfall rendering failed: {e}")
            return False
    
    def _render_layers_gpu(self, screen: pygame.Surface, vis_params: Dict[str, Any],
                          main_spectrum_data: np.ndarray) -> bool:
        """Render waterfall layers using GPU acceleration"""
        try:
            # Extract visualization parameters
            vis_start_x = vis_params.get('vis_start_x', 0)
            vis_width = vis_params.get('vis_width', screen.get_width())
            center_y = vis_params.get('center_y', screen.get_height() // 2)
            max_bar_height = vis_params.get('max_bar_height', 200)
            
            # Calculate threading parameters for CUDA kernels
            threads_per_block = 256
            blocks = (self.num_bars + threads_per_block - 1) // threads_per_block
            
            # Render each historical slice from back to front (painter's algorithm)
            for i, gpu_slice in enumerate(reversed(self.spectrum_history_gpu)):
                depth_index = len(self.spectrum_history_gpu) - 1 - i
                
                # Apply depth effects using GPU
                processed_slice = cp.zeros_like(gpu_slice)
                
                self.depth_effects_kernel(
                    (blocks,), (threads_per_block,),
                    (processed_slice, gpu_slice,
                     self._depth_scales[depth_index],
                     self._depth_alphas[depth_index], 
                     self.settings.color_shift ** depth_index,
                     self.num_bars)
                )
                
                # Transfer to CPU for pygame rendering
                cpu_slice = processed_slice.get()
                
                # Render this layer
                self._render_single_layer(screen, cpu_slice, vis_params, depth_index)
            
            return True
            
        except Exception as e:
            logger.error(f"GPU waterfall rendering failed: {e}")
            return False
    
    def _render_layers_cpu(self, screen: pygame.Surface, vis_params: Dict[str, Any],
                          main_spectrum_data: np.ndarray) -> bool:
        """Render waterfall layers using CPU processing (fallback)"""
        try:
            # Render each historical slice from back to front
            for i, cpu_slice in enumerate(reversed(self.spectrum_history_cpu)):
                depth_index = len(self.spectrum_history_cpu) - 1 - i
                
                # Apply depth effects
                scale_factor = self._depth_scales[depth_index]
                alpha_factor = self._depth_alphas[depth_index]
                color_shift = self.settings.color_shift ** depth_index
                
                processed_slice = cpu_slice * scale_factor * alpha_factor * color_shift
                
                # Render this layer
                self._render_single_layer(screen, processed_slice, vis_params, depth_index)
            
            return True
            
        except Exception as e:
            logger.error(f"CPU waterfall rendering failed: {e}")
            return False
    
    def _render_single_layer(self, screen: pygame.Surface, spectrum_slice: np.ndarray,
                           vis_params: Dict[str, Any], depth_index: int):
        """Render a single waterfall layer with 3D perspective from corners"""
        try:
            # Validate inputs
            if depth_index >= len(self._depth_scales) or len(spectrum_slice) == 0:
                return
            
            # Extract main spectrum parameters
            vis_start_x = max(0, vis_params.get('vis_start_x', 0))
            vis_width = max(100, vis_params.get('vis_width', screen.get_width()))
            center_y = max(50, min(screen.get_height() - 50, vis_params.get('center_y', screen.get_height() // 2)))
            max_bar_height = max(10, vis_params.get('max_bar_height', 200))
            
            spectrum_top = vis_params.get('spectrum_top', center_y - max_bar_height)
            spectrum_bottom = vis_params.get('spectrum_bottom', center_y + max_bar_height)
            
            # Calculate depth and scaling
            scale = self._depth_scales[depth_index]
            alpha = self._depth_alphas[depth_index]
            
            if not (0.01 <= scale <= 1.0) or not (0.0 <= alpha <= 1.0):
                return
            
            # Calculate 3D perspective transformation
            # Main spectrum corners define the front plane
            front_corners = {
                'top_left': (vis_start_x, spectrum_top),
                'top_right': (vis_start_x + vis_width, spectrum_top), 
                'bottom_left': (vis_start_x, spectrum_bottom),
                'bottom_right': (vis_start_x + vis_width, spectrum_bottom)
            }
            
            # Calculate perspective depth offset (moving toward center)
            depth_factor = (1.0 - scale) * self.settings.depth_spacing * depth_index
            screen_center_x = screen.get_width() // 2
            screen_center_y = screen.get_height() // 2
            
            # Calculate back plane corners by moving toward screen center
            back_corners = {}
            for corner_name, (x, y) in front_corners.items():
                # Move each corner toward the center based on depth
                dx = screen_center_x - x
                dy = screen_center_y - y
                
                # Apply perspective transformation
                back_x = x + dx * (depth_factor / 100.0)  # Scale depth effect
                back_y = y + dy * (depth_factor / 100.0)
                
                back_corners[corner_name] = (int(back_x), int(back_y))
            
            # Draw 3D connecting lines from front to back corners
            line_color = tuple(max(0, min(255, int(c * 0.3))) for c in (100, 100, 150))
            line_alpha = max(20, int(alpha * 80))
            
            # Create surface for 3D framework
            framework_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            
            # Draw corner-to-corner depth lines
            for corner_name in front_corners:
                front_point = front_corners[corner_name]
                back_point = back_corners[corner_name]
                
                # Draw depth line
                pygame.draw.line(framework_surface, (*line_color, line_alpha), 
                               front_point, back_point, 1)
            
            # Draw back plane outline
            back_points = [
                back_corners['top_left'],
                back_corners['top_right'], 
                back_corners['bottom_right'],
                back_corners['bottom_left'],
                back_corners['top_left']  # Close the rectangle
            ]
            
            if len(back_points) >= 3:
                pygame.draw.lines(framework_surface, (*line_color, line_alpha), 
                                False, back_points, 1)
            
            # Blit framework
            screen.blit(framework_surface, (0, 0))
            
            # Now render the spectrum at the back plane position
            back_vis_start_x = back_corners['top_left'][0]
            back_vis_width = back_corners['top_right'][0] - back_corners['top_left'][0]
            back_spectrum_top = back_corners['top_left'][1]
            back_spectrum_bottom = back_corners['bottom_left'][1]
            back_center_y = (back_spectrum_top + back_spectrum_bottom) // 2
            back_max_height = abs(back_spectrum_bottom - back_spectrum_top) // 2
            
            # Validate back plane dimensions
            if back_vis_width <= 0 or back_max_height <= 0:
                return
            
            # Render spectrum bars at back plane
            num_bars = min(len(spectrum_slice), self.num_bars, 1000)
            if num_bars <= 0:
                return
                
            bar_width = back_vis_width / num_bars
            if bar_width < 0.1:
                return
            
            # Create surface for back plane spectrum
            try:
                back_surface = pygame.Surface((abs(back_vis_width), abs(back_max_height * 2)), pygame.SRCALPHA)
            except (ValueError, MemoryError):
                return
            
            # Render bars on back plane
            for i in range(num_bars):
                try:
                    spectrum_val = float(spectrum_slice[i])
                    if not np.isfinite(spectrum_val) or spectrum_val <= 0.001:
                        continue
                    
                    spectrum_val = max(0.0, min(1.0, spectrum_val))
                    
                    height = max(1, int(spectrum_val * back_max_height))
                    x = max(0, int(i * bar_width))
                    width = max(1, int(bar_width) + 1)
                    
                    # Calculate colors with depth effect
                    base_color = self._get_frequency_color(i, num_bars)
                    color_shift = max(0.2, min(1.0, self.settings.color_shift ** depth_index))
                    
                    depth_color = tuple(max(0, min(255, int(c * color_shift))) for c in base_color)
                    alpha_int = max(30, min(255, int(alpha * 255)))
                    color_with_alpha = (*depth_color, alpha_int)
                    
                    # Draw bars on back surface
                    upper_y = max(0, back_max_height - height)
                    upper_rect = pygame.Rect(x, upper_y, width, height)
                    lower_rect = pygame.Rect(x, back_max_height, width, height)
                    
                    if back_surface.get_rect().contains(upper_rect):
                        pygame.draw.rect(back_surface, color_with_alpha, upper_rect)
                    
                    lower_color = tuple(max(0, min(255, int(c * 0.75))) for c in depth_color)
                    lower_color_with_alpha = (*lower_color, alpha_int)
                    
                    if back_surface.get_rect().contains(lower_rect):
                        pygame.draw.rect(back_surface, lower_color_with_alpha, lower_rect)
                        
                except (ValueError, TypeError, OverflowError, IndexError):
                    continue
            
            # Blit back plane spectrum
            if back_vis_start_x >= 0 and back_center_y - back_max_height >= 0:
                screen.blit(back_surface, (back_vis_start_x, back_center_y - back_max_height))
            
        except Exception as e:
            logger.debug(f"3D layer rendering skipped: {e}")
    
    def _get_frequency_color(self, bar_index: int, total_bars: int) -> Tuple[int, int, int]:
        """Get color for a frequency bar (similar to main spectrum)"""
        if total_bars <= 1:
            return (100, 150, 255)
        
        hue = bar_index / total_bars
        
        # Color gradient: Purple -> Red -> Orange -> Yellow -> Green -> Cyan -> Blue
        if hue < 0.167:  # Purple to Red
            t = hue / 0.167
            r = int(150 + 105 * t)
            g = int(50 * (1 - t))
            b = int(200 * (1 - t))
        elif hue < 0.333:  # Red to Orange to Yellow
            t = (hue - 0.167) / 0.166
            r = 255
            g = int(255 * t)
            b = 0
        elif hue < 0.5:  # Yellow to Green
            t = (hue - 0.333) / 0.167
            r = int(255 * (1 - t))
            g = 255
            b = int(100 * t)
        elif hue < 0.667:  # Green to Cyan
            t = (hue - 0.5) / 0.167
            r = 0
            g = int(255 - 55 * t)
            b = int(100 + 155 * t)
        elif hue < 0.833:  # Cyan to Blue
            t = (hue - 0.667) / 0.166
            r = int(100 * t)
            g = int(200 * (1 - t))
            b = 255
        else:  # Blue to Light Blue
            t = (hue - 0.833) / 0.167
            r = int(100 + 155 * t)
            g = int(200 * t)
            b = 255
        
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    def toggle_waterfall(self) -> bool:
        """Toggle waterfall visualization on/off"""
        self.settings.enabled = not self.settings.enabled
        logger.info(f"Waterfall 3D: {'enabled' if self.settings.enabled else 'disabled'}")
        return self.settings.enabled
    
    def adjust_depth(self, delta: float):
        """Adjust the depth spacing between layers"""
        self.settings.depth_spacing = max(2.0, min(20.0, self.settings.depth_spacing + delta))
        self._depth_positions = np.array([
            i * self.settings.depth_spacing for i in range(self.settings.max_slices)
        ])
        logger.info(f"Waterfall depth spacing: {self.settings.depth_spacing:.1f}px")
    
    def adjust_slice_interval(self, delta: float):
        """Adjust how frequently new slices are added"""
        self.settings.slice_interval = max(0.05, min(1.0, self.settings.slice_interval + delta))
        logger.info(f"Waterfall slice interval: {self.settings.slice_interval:.2f}s")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current waterfall status"""
        if self.use_gpu:
            num_slices = len(self.spectrum_history_gpu)
            gpu_memory = sum(slice.nbytes for slice in self.spectrum_history_gpu) / (1024*1024)
        else:
            num_slices = len(self.spectrum_history_cpu)
            gpu_memory = 0
        
        return {
            "enabled": self.settings.enabled,
            "gpu_acceleration": self.use_gpu,
            "num_slices": num_slices,
            "max_slices": self.settings.max_slices,
            "slice_interval": self.settings.slice_interval,
            "depth_spacing": self.settings.depth_spacing,
            "gpu_memory_mb": gpu_memory,
            "interpolation": self.settings.enable_interpolation
        }