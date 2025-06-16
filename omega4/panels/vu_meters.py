"""
VU Meters Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract VU meters as self-contained module
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class VUMetersPanel:
    """Professional VU meters panel with proper ballistics and peak hold"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # VU meter ballistics parameters
        self.vu_integration_time = 300e-3  # 300ms integration time (standard VU)
        self.vu_damping = 0.94  # VU needle damping factor (faster response)
        # Use digital full scale reference instead of analog 0dBu
        # This makes 0 VU = -18 dBFS, which is a common digital reference
        self.vu_reference_level = 0.125  # -18 dBFS reference level
        self.vu_peak_hold_time = 2.0  # Peak hold time in seconds
        
        # VU meter buffers (300ms at 48kHz = 14400 samples)
        buffer_size = int(self.vu_integration_time * sample_rate)
        self.vu_left_buffer = deque(maxlen=buffer_size)
        self.vu_right_buffer = deque(maxlen=buffer_size)
        
        # VU meter display values
        self.vu_left_db = -60.0
        self.vu_right_db = -60.0
        self.vu_left_display = -60.0
        self.vu_right_display = -60.0
        
        # Peak hold for VU meters
        self.vu_left_peak_db = -60.0
        self.vu_right_peak_db = -60.0
        self.vu_left_peak_time = 0.0
        self.vu_right_peak_time = 0.0
        
        # Fonts will be set by main app
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.font_tiny = None
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_large = fonts.get('large')
        self.font_medium = fonts.get('medium')
        self.font_small = fonts.get('small')
        self.font_tiny = fonts.get('tiny')
        
    def update(self, audio_data: np.ndarray, dt: float):
        """Process audio for VU meter display with proper ballistics"""
        if audio_data is None or len(audio_data) == 0:
            return
            
        # For mono input, duplicate to both channels
        left_channel = audio_data
        right_channel = audio_data
        
        # Add samples to VU buffers
        for sample in left_channel:
            self.vu_left_buffer.append(sample)
            self.vu_right_buffer.append(sample)
            
        # Calculate RMS over integration period
        left_rms = np.sqrt(np.mean(np.array(self.vu_left_buffer) ** 2))
        right_rms = np.sqrt(np.mean(np.array(self.vu_right_buffer) ** 2))
        
        # Convert to dB relative to reference
        # First convert to dBFS, then offset to VU scale
        left_dbfs = self._linear_to_db_vu(left_rms) if left_rms > 0 else -60.0
        right_dbfs = self._linear_to_db_vu(right_rms) if right_rms > 0 else -60.0
        
        # Convert dBFS to VU scale (0 VU = -18 dBFS)
        self.vu_left_db = left_dbfs + 18.0
        self.vu_right_db = right_dbfs + 18.0
        
        # Apply damping for smooth needle movement
        self.vu_left_display += (self.vu_left_db - self.vu_left_display) * (1.0 - self.vu_damping)
        self.vu_right_display += (self.vu_right_db - self.vu_right_display) * (1.0 - self.vu_damping)
        
        # Update peaks (use display values instead of raw values for smoother peak tracking)
        if self.vu_left_display > self.vu_left_peak_db:
            self.vu_left_peak_db = self.vu_left_display
            self.vu_left_peak_time = 0.0
        else:
            self.vu_left_peak_time += dt
            if self.vu_left_peak_time > self.vu_peak_hold_time:
                self.vu_left_peak_db = max(self.vu_left_peak_db - 10.0 * dt, -20.0)  # Decay but stop at scale minimum
                
        if self.vu_right_display > self.vu_right_peak_db:
            self.vu_right_peak_db = self.vu_right_display
            self.vu_right_peak_time = 0.0
        else:
            self.vu_right_peak_time += dt
            if self.vu_right_peak_time > self.vu_peak_hold_time:
                self.vu_right_peak_db = max(self.vu_right_peak_db - 10.0 * dt, -20.0)  # Decay but stop at scale minimum
    
    def _linear_to_db_vu(self, linear: float) -> float:
        """Convert linear scale to dB for VU meter"""
        if linear <= 0:
            return -60.0
        return 20.0 * np.log10(linear)
        
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw professional VU meters on the right side"""
        
        # Background panel with slight transparency
        panel_rect = pygame.Rect(x, y, width, height)
        # Create a surface for the background with alpha
        panel_surface = pygame.Surface((width, height))
        panel_surface.set_alpha(230)  # Slight transparency
        panel_surface.fill((20, 25, 35))
        screen.blit(panel_surface, (x, y))
        # Draw border
        pygame.draw.rect(screen, (60, 70, 90), panel_rect, 2)
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("VU METERS", True, (200, 220, 240))
            title_rect = title_text.get_rect(centerx=x + width // 2, top=y + int(10 * ui_scale))
            screen.blit(title_text, title_rect)
        
        # VU meter parameters - reduced height to create space for text below
        meter_y_start = y + int(60 * ui_scale)  # Space for title
        
        # Calculate meter height to leave room for 4 rows of text + padding below
        # 4 rows: L/R labels, current dB values, peak values, reference text
        # Each row approx 20px + 10px padding = 90px total
        text_space_needed = int(90 * ui_scale)
        
        # For tall panels (matching spectrum height), use most of the height
        if height > 400:
            # Use height minus space for title and bottom text area
            meter_height = height - int(60 * ui_scale) - text_space_needed - int(10 * ui_scale)
        else:
            # For smaller panels, use proportional calculation
            meter_height = height - int(60 * ui_scale) - text_space_needed - int(10 * ui_scale)
            
        meter_width = int(40 * ui_scale)
        meter_spacing = int(20 * ui_scale)
        
        # Calculate meter positions
        total_meter_width = 2 * meter_width + meter_spacing
        meters_x_start = x + (width - total_meter_width) // 2
        
        # Scale parameters
        scale_min = -20.0  # dB
        scale_max = 3.0    # dB
        
        # Draw left meter
        left_x = meters_x_start
        self._draw_single_vu_meter(screen, left_x, meter_y_start, meter_width, meter_height,
                                   self.vu_left_display, self.vu_left_peak_db,
                                   scale_min, scale_max, "L", ui_scale)
        
        # Draw right meter
        right_x = meters_x_start + meter_width + meter_spacing
        self._draw_single_vu_meter(screen, right_x, meter_y_start, meter_width, meter_height,
                                   self.vu_right_display, self.vu_right_peak_db,
                                   scale_min, scale_max, "R", ui_scale)
        
        # Draw text below meters with proper spacing
        text_y_start = meter_y_start + meter_height + int(10 * ui_scale)
        
        # Row 1: L/R channel labels
        if self.font_medium:
            left_label_x = meters_x_start + meter_width // 2
            right_label_x = meters_x_start + meter_width + meter_spacing + meter_width // 2
            
            left_label = self.font_medium.render("L", True, (200, 220, 240))
            left_rect = left_label.get_rect(centerx=left_label_x, top=text_y_start)
            screen.blit(left_label, left_rect)
            
            right_label = self.font_medium.render("R", True, (200, 220, 240))
            right_rect = right_label.get_rect(centerx=right_label_x, top=text_y_start)
            screen.blit(right_label, right_rect)
        
        # Row 2: Current dB values
        if self.font_small:
            text_y_start += int(20 * ui_scale)
            
            left_value = self.font_small.render(f"{self.vu_left_display:+5.1f} dB", True, (180, 200, 220))
            left_rect = left_value.get_rect(centerx=left_label_x, top=text_y_start)
            screen.blit(left_value, left_rect)
            
            right_value = self.font_small.render(f"{self.vu_right_display:+5.1f} dB", True, (180, 200, 220))
            right_rect = right_value.get_rect(centerx=right_label_x, top=text_y_start)
            screen.blit(right_value, right_rect)
        
        # Row 3: Peak values
        if self.font_tiny:
            text_y_start += int(20 * ui_scale)
            
            left_peak = self.font_tiny.render(f"Peak: {self.vu_left_peak_db:+5.1f}", True, (150, 170, 190))
            left_rect = left_peak.get_rect(centerx=left_label_x, top=text_y_start)
            screen.blit(left_peak, left_rect)
            
            right_peak = self.font_tiny.render(f"Peak: {self.vu_right_peak_db:+5.1f}", True, (150, 170, 190))
            right_rect = right_peak.get_rect(centerx=right_label_x, top=text_y_start)
            screen.blit(right_peak, right_rect)
        
        # Row 4: Reference info
        if self.font_tiny:
            text_y_start += int(20 * ui_scale)
            ref_text = self.font_tiny.render("0 VU = -18 dBFS", True, (150, 170, 190))
            ref_rect = ref_text.get_rect(centerx=x + width // 2, top=text_y_start)
            screen.blit(ref_text, ref_rect)
    
    def _draw_single_vu_meter(self, screen: pygame.Surface, x: int, y: int, width: int, height: int,
                              value: float, peak: float, scale_min: float, scale_max: float,
                              label: str, ui_scale: float = 1.0):
        """Draw a single VU meter"""
        # Meter background
        meter_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (15, 20, 30), meter_rect)
        pygame.draw.rect(screen, (50, 60, 80), meter_rect, 1)
        
        # Draw scale marks - more detail for tall panels
        if height > 400:
            scale_positions = [
                (-20, "−20", True),
                (-15, "−15", False),
                (-10, "−10", True),
                (-7, "−7", True),
                (-5, "−5", True),
                (-3, "−3", True),
                (-2, "", False),
                (-1, "", False),
                (0, "0", True),
                (1, "+1", False),
                (2, "+2", False),
                (3, "+3", True)
            ]
        else:
            scale_positions = [
                (-20, "−20", True),
                (-10, "−10", True),
                (-7, "−7", True),
                (-5, "−5", True),
                (-3, "−3", True),
                (0, "0", True),
                (3, "+3", True)
            ]
        
        for db, db_label, is_major in scale_positions:
            # Calculate Y position
            normalized = (db - scale_min) / (scale_max - scale_min)
            mark_y = int(y + height * (1.0 - normalized))
            
            # Draw tick mark
            if is_major:
                tick_length = int(8 * ui_scale)
                tick_color = (180, 180, 200)
                # Draw label - show all labels for tall panels
                show_label = (height > 400 or db in [-20, -10, 0, 3])
                if show_label and db_label and self.font_tiny:
                    label_surf = self.font_tiny.render(db_label, True, tick_color)
                    label_rect = label_surf.get_rect(right=x - int(5 * ui_scale), centery=mark_y)
                    if label_rect.top >= y and label_rect.bottom <= y + height:
                        screen.blit(label_surf, label_rect)
            else:
                tick_length = int(4 * ui_scale)
                tick_color = (100, 100, 120)
                # Draw minor labels for tall panels
                if height > 400 and db_label and self.font_tiny:
                    label_surf = self.font_tiny.render(db_label, True, tick_color)
                    label_rect = label_surf.get_rect(right=x - int(5 * ui_scale), centery=mark_y)
                    if label_rect.top >= y and label_rect.bottom <= y + height:
                        screen.blit(label_surf, label_rect)
                
            pygame.draw.line(screen, tick_color, (x, mark_y), (x + tick_length, mark_y), 1)
        
        # Calculate needle position
        value_normalized = (value - scale_min) / (scale_max - scale_min)
        value_normalized = max(0.0, min(1.0, value_normalized))
        value_y = int(y + height * (1.0 - value_normalized))
        
        # Calculate peak position
        peak_normalized = (peak - scale_min) / (scale_max - scale_min)
        peak_normalized = max(0.0, min(1.0, peak_normalized))
        peak_y = int(y + height * (1.0 - peak_normalized))
        
        # Draw the meter bar with gradient effect
        bar_x = x + int(10 * ui_scale)
        bar_width = width - int(20 * ui_scale)
        
        # Fill up to current value with color coding
        if value_y < y + height:
            fill_height = y + height - value_y
            fill_rect = pygame.Rect(bar_x, value_y, bar_width, fill_height)
            
            # Color based on level
            if value > 0:
                # Red zone
                color = (200, 50, 40)
            elif value > -3:
                # Yellow zone
                color = (200, 180, 40)
            else:
                # Green zone
                color = (40, 180, 60)
                
            pygame.draw.rect(screen, color, fill_rect)
            
            # Add gradient effect
            for i in range(3):
                grad_color = tuple(int(c * (1 - i * 0.2)) for c in color)
                grad_rect = pygame.Rect(bar_x + i, value_y + i, bar_width - 2*i, fill_height - 2*i)
                pygame.draw.rect(screen, grad_color, grad_rect, 1)
        
        # Draw needle pointer
        needle_color = (255, 255, 255)
        pygame.draw.line(screen, needle_color, (x, value_y), (x + width, value_y), 2)
        
        # Draw peak indicator
        if peak > scale_min:
            peak_color = (255, 150, 150)
            # Draw a thicker line for better visibility
            pygame.draw.line(screen, peak_color, (bar_x - 2, peak_y), (bar_x + bar_width + 2, peak_y), 2)
            # Add small triangular markers on sides
            pygame.draw.polygon(screen, peak_color, 
                              [(bar_x - 5, peak_y), (bar_x - 2, peak_y - 2), (bar_x - 2, peak_y + 2)])
            pygame.draw.polygon(screen, peak_color, 
                              [(bar_x + bar_width + 5, peak_y), (bar_x + bar_width + 2, peak_y - 2), 
                               (bar_x + bar_width + 2, peak_y + 2)])
        
        # Channel label is now drawn in the main draw method
    
    def get_results(self) -> Dict[str, any]:
        """Get current VU meter values"""
        return {
            'left_db': self.vu_left_display,
            'right_db': self.vu_right_display,
            'left_peak': self.vu_left_peak_db,
            'right_peak': self.vu_right_peak_db
        }