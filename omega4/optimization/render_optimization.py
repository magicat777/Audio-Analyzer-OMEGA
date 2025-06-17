"""
Render Optimization - Surface caching and dirty rectangle management
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import hashlib


@dataclass
class CachedSurface:
    """Cached surface with metadata"""
    surface: pygame.Surface
    hash_key: str
    last_used: float
    render_count: int = 0


class RenderOptimizer:
    """Optimizes rendering through caching and dirty rectangle management"""
    
    def __init__(self, max_cache_size: int = 100):
        self.max_cache_size = max_cache_size
        
        # Surface cache for static elements
        self.surface_cache: Dict[str, CachedSurface] = {}
        
        # Dirty rectangles tracking
        self.dirty_rects: List[pygame.Rect] = []
        self.full_redraw_needed = True
        
        # Panel surface buffers
        self.panel_surfaces: Dict[str, pygame.Surface] = {}
        self.panel_changed: Dict[str, bool] = {}
        
        # Background caching
        self.cached_backgrounds: Dict[str, pygame.Surface] = {}
        
        # Text rendering cache
        self.text_cache: Dict[str, pygame.Surface] = {}
        self.max_text_cache = 500
        
    def get_or_create_panel_surface(self, panel_id: str, width: int, height: int) -> pygame.Surface:
        """Get or create a surface for a panel"""
        key = f"{panel_id}_{width}x{height}"
        
        if key not in self.panel_surfaces:
            # Create surface with per-pixel alpha for better blending
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self.panel_surfaces[key] = surface
            self.panel_changed[panel_id] = True
        
        return self.panel_surfaces[key]
        
    def mark_panel_dirty(self, panel_id: str, rect: pygame.Rect):
        """Mark a panel area as needing redraw"""
        self.panel_changed[panel_id] = True
        self.dirty_rects.append(rect)
        
    def is_panel_dirty(self, panel_id: str) -> bool:
        """Check if panel needs redraw"""
        return self.panel_changed.get(panel_id, True)
        
    def render_text_cached(self, text: str, font: pygame.font.Font, 
                          color: Tuple[int, int, int], 
                          antialias: bool = True) -> pygame.Surface:
        """Render text with caching"""
        # Create cache key
        cache_key = f"{text}_{id(font)}_{color}_{antialias}"
        
        if cache_key in self.text_cache:
            return self.text_cache[cache_key]
            
        # Render text
        surface = font.render(text, antialias, color)
        
        # Add to cache
        self.text_cache[cache_key] = surface
        
        # Manage cache size
        if len(self.text_cache) > self.max_text_cache:
            # Remove oldest entries (simple FIFO for now)
            oldest_keys = list(self.text_cache.keys())[:len(self.text_cache) - self.max_text_cache]
            for key in oldest_keys:
                del self.text_cache[key]
                
        return surface
        
    def create_cached_background(self, key: str, width: int, height: int, 
                               color: Tuple[int, int, int], 
                               alpha: int = 255) -> pygame.Surface:
        """Create or retrieve a cached background surface"""
        cache_key = f"{key}_{width}x{height}_{color}_{alpha}"
        
        if cache_key in self.cached_backgrounds:
            return self.cached_backgrounds[cache_key]
            
        # Create background
        surface = pygame.Surface((width, height))
        surface.set_alpha(alpha)
        surface.fill(color)
        
        self.cached_backgrounds[cache_key] = surface
        return surface
        
    def optimize_panel_group_render(self, screen: pygame.Surface, 
                                   panels: List[Tuple[str, pygame.Rect, callable]],
                                   force_redraw: bool = False) -> List[pygame.Rect]:
        """Optimized rendering of a group of panels"""
        updated_rects = []
        
        for panel_id, rect, draw_func in panels:
            if force_redraw or self.is_panel_dirty(panel_id):
                # Get panel surface
                panel_surface = self.get_or_create_panel_surface(panel_id, rect.width, rect.height)
                
                # Clear surface
                panel_surface.fill((0, 0, 0, 0))
                
                # Draw to panel surface
                draw_func(panel_surface, 0, 0, rect.width, rect.height)
                
                # Blit to screen
                screen.blit(panel_surface, rect)
                updated_rects.append(rect)
                
                # Mark as clean
                self.panel_changed[panel_id] = False
                
        return updated_rects
        
    def batch_render_static_elements(self, elements: List[Tuple[str, callable, Tuple]]) -> Dict[str, pygame.Surface]:
        """Batch render static elements that don't change often"""
        rendered = {}
        
        for element_id, render_func, params in elements:
            # Create hash of parameters
            param_hash = hashlib.md5(str(params).encode()).hexdigest()
            cache_key = f"{element_id}_{param_hash}"
            
            if cache_key in self.surface_cache:
                # Use cached version
                cached = self.surface_cache[cache_key]
                rendered[element_id] = cached.surface
                cached.render_count += 1
            else:
                # Render new
                surface = render_func(*params)
                self.surface_cache[cache_key] = CachedSurface(
                    surface=surface,
                    hash_key=cache_key,
                    last_used=pygame.time.get_ticks(),
                    render_count=1
                )
                rendered[element_id] = surface
                
                # Manage cache size
                if len(self.surface_cache) > self.max_cache_size:
                    # Remove least used
                    sorted_cache = sorted(
                        self.surface_cache.items(),
                        key=lambda x: (x[1].render_count, x[1].last_used)
                    )
                    for key, _ in sorted_cache[:len(self.surface_cache) - self.max_cache_size]:
                        del self.surface_cache[key]
                        
        return rendered
        
    def get_dirty_rects(self) -> List[pygame.Rect]:
        """Get list of dirty rectangles for this frame"""
        return self.dirty_rects.copy()
        
    def clear_dirty_rects(self):
        """Clear dirty rectangles after frame update"""
        self.dirty_rects.clear()
        self.full_redraw_needed = False
        
    def request_full_redraw(self):
        """Request a full screen redraw next frame"""
        self.full_redraw_needed = True
        
    def needs_full_redraw(self) -> bool:
        """Check if full redraw is needed"""
        return self.full_redraw_needed
        
    def optimize_spectrum_render(self, spectrum_data: np.ndarray, 
                               prev_spectrum: Optional[np.ndarray] = None,
                               threshold: float = 0.01) -> Optional[List[int]]:
        """Optimize spectrum rendering by detecting changed bars"""
        if prev_spectrum is None or len(spectrum_data) != len(prev_spectrum):
            return None  # Full redraw needed
            
        # Find bars that changed significantly
        diff = np.abs(spectrum_data - prev_spectrum)
        changed_indices = np.where(diff > threshold)[0]
        
        if len(changed_indices) > len(spectrum_data) * 0.5:
            return None  # Too many changes, full redraw more efficient
            
        return changed_indices.tolist()
        
    def create_spectrum_surface_cached(self, width: int, height: int, 
                                     bar_count: int) -> pygame.Surface:
        """Create cached surface for spectrum rendering"""
        key = f"spectrum_{width}x{height}_{bar_count}"
        
        if key not in self.panel_surfaces:
            # Create surface with per-pixel alpha
            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            self.panel_surfaces[key] = surface
            
        return self.panel_surfaces[key]