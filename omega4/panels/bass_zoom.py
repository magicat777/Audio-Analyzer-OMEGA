"""
Bass Zoom Window Panel for OMEGA-4 Audio Analyzer
Phase 3: Extract bass zoom window as self-contained module
"""

import pygame
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
from queue import Queue, Empty
import threading


class BassZoomPanel:
    """Bass zoom window panel for detailed bass frequency analysis (20-200 Hz)"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Bass detail configuration
        self.bass_detail_bars = 64
        self.bass_bar_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_timestamps = np.zeros(self.bass_detail_bars, dtype=np.float64)
        
        # Bass frequency ranges and bin mapping (initialized in setup_bass_mapping)
        self.bass_freq_ranges = []
        self.bass_bin_mapping = []
        
        # Async bass processing for reduced latency
        self.bass_processing_queue = Queue(maxsize=5)
        self.latest_bass_result = None
        self.bass_thread_running = True
        self.bass_processing_thread = threading.Thread(target=self._bass_processor_worker)
        self.bass_processing_thread.daemon = True
        self.bass_processing_thread.start()
        
        # Fonts will be set by main app
        self.font_small = None
        self.font_tiny = None
        
        # Initialize bass mapping
        self.setup_bass_mapping()
        
    def set_fonts(self, fonts: Dict[str, pygame.font.Font]):
        """Set fonts for rendering"""
        self.font_small = fonts.get('small')
        self.font_tiny = fonts.get('tiny')
        
    def setup_bass_mapping(self):
        """Set up bass frequency ranges and FFT bin mapping"""
        bass_fft_size = 8192
        bass_freqs = np.fft.rfftfreq(bass_fft_size, 1 / self.sample_rate)
        
        # Find bins in bass range (20-200 Hz)
        valid_bins = []
        for i, freq in enumerate(bass_freqs):
            if 20 <= freq <= 200:
                valid_bins.append(i)
        
        if len(valid_bins) == 0:
            # Fallback if no valid bins
            self.bass_freq_ranges = [(20, 200)]
            self.bass_bin_mapping = [[]]
            self.bass_detail_bars = 1
            return
        
        # Create frequency ranges based on actual FFT bins for complete coverage
        self.bass_freq_ranges = []
        self.bass_bin_mapping = []
        
        # Group bins to create smooth visualization
        bins_per_bar = max(1, len(valid_bins) // 31)  # Target ~31 bars for good visual density
        
        for i in range(0, len(valid_bins), bins_per_bar):
            end_idx = min(i + bins_per_bar, len(valid_bins))
            bin_group = valid_bins[i:end_idx]
            
            if len(bin_group) > 0:
                f_start = bass_freqs[bin_group[0]]
                f_end = bass_freqs[bin_group[-1]]
                
                # Ensure ranges don't overlap and cover gaps
                if len(self.bass_freq_ranges) > 0:
                    f_start = max(f_start, self.bass_freq_ranges[-1][1])
                
                self.bass_freq_ranges.append((f_start, f_end))
                self.bass_bin_mapping.append(bin_group)
        
        # Update bar count based on actual ranges
        self.bass_detail_bars = len(self.bass_freq_ranges)
        
        # Resize arrays to match actual bar count
        self.bass_bar_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_values = np.zeros(self.bass_detail_bars, dtype=np.float32)
        self.bass_peak_timestamps = np.zeros(self.bass_detail_bars, dtype=np.float64)
    
    def _bass_processor_worker(self):
        """Worker thread for async bass processing"""
        while self.bass_thread_running:
            try:
                audio_data = self.bass_processing_queue.get(timeout=0.1)
                if audio_data is None:
                    break
                    
                # Process bass detail in background thread
                result = self._process_bass_detail_internal(audio_data)
                self.latest_bass_result = result
                
            except Empty:
                continue
            except Exception as e:
                import traceback
                print(f"Bass processing error: {type(e).__name__}: {e}")
                traceback.print_exc()
    
    def update(self, audio_data: np.ndarray, drum_info: Dict = None):
        """Update bass analysis with new audio data"""
        # Submit for async processing
        try:
            # Non-blocking put - skip if queue is full
            self.bass_processing_queue.put_nowait(audio_data.copy())
        except:
            pass  # Skip this frame if queue is full
        
        # Apply any completed results
        self.apply_bass_results()
        
        # Store drum info for kick detection enhancement
        self.drum_info = drum_info if drum_info is not None else {}
    
    def apply_bass_results(self):
        """Apply the latest bass processing results if available"""
        if self.latest_bass_result is not None:
            self.bass_bar_values = self.latest_bass_result['bar_values']
            self.bass_peak_values = self.latest_bass_result['peak_values']
            self.bass_peak_timestamps = self.latest_bass_result['peak_timestamps']
            self.latest_bass_result = None
    
    def _process_bass_detail_internal(self, audio_data: np.ndarray):
        """Internal bass processing logic (runs on separate thread)"""
        bass_fft_size = 8192
        
        # Apply window
        window = np.hanning(min(len(audio_data), bass_fft_size))
        if len(audio_data) < bass_fft_size:
            padded_audio = np.zeros(bass_fft_size)
            padded_audio[:len(audio_data)] = audio_data * window
            windowed_audio = padded_audio
        else:
            windowed_audio = audio_data[:bass_fft_size] * window
        
        # Compute FFT
        bass_fft = np.fft.rfft(windowed_audio)
        bass_magnitude = np.abs(bass_fft)
        
        # Create copies for thread safety
        bar_values = self.bass_bar_values.copy()
        peak_values = self.bass_peak_values.copy()
        peak_timestamps = self.bass_peak_timestamps.copy()
        
        # Calculate dynamic range for proper scaling
        bass_max = 0.0
        raw_values = {}
        
        # First pass: collect raw values
        for i, bin_indices in enumerate(self.bass_bin_mapping):
            if i >= len(bar_values):
                break  # Prevent index out of bounds
            if len(bin_indices) > 0:
                bar_value = np.mean(bass_magnitude[bin_indices])
                if i >= len(self.bass_freq_ranges):
                    continue  # Skip if freq_ranges doesn't have this index
                center_freq = (self.bass_freq_ranges[i][0] + self.bass_freq_ranges[i][1]) / 2
                
                # Apply frequency compensation
                if center_freq < 60:
                    bar_value *= 0.3
                elif center_freq < 100:
                    bar_value *= 0.6
                elif center_freq < 150:
                    bar_value *= 1.0
                else:
                    bar_value *= 0.8
                
                raw_values[i] = bar_value
                bass_max = max(bass_max, bar_value)
        
        # Dynamic scaling
        if bass_max > 0:
            scale_factor = 0.85 / bass_max
            log_scale = np.log10(max(1.0, bass_max * 10)) / 2.0
            scale_factor *= log_scale
        else:
            scale_factor = 1.0
        
        # Second pass: apply scaling and update bars
        current_time = time.time()
        for i, bin_indices in enumerate(self.bass_bin_mapping):
            if i >= len(bar_values):
                break  # Prevent index out of bounds
            if len(bin_indices) > 0 and i in raw_values:
                scaled_value = raw_values[i] * scale_factor
                
                # Apply compression curve
                if scaled_value > 0.7:
                    compressed_value = 0.7 + (scaled_value - 0.7) * 0.3
                else:
                    compressed_value = scaled_value
                
                # Smooth the bass values
                if compressed_value > bar_values[i]:
                    bar_values[i] = bar_values[i] * 0.1 + compressed_value * 0.9
                else:
                    bar_values[i] = bar_values[i] * 0.6 + compressed_value * 0.4
                
                # Clamp to 0-1 range
                bar_values[i] = max(0.0, min(1.0, bar_values[i]))
                
                # Update peak hold
                if bar_values[i] > peak_values[i]:
                    peak_values[i] = bar_values[i]
                    peak_timestamps[i] = current_time
                elif current_time - peak_timestamps[i] > 3.0:
                    peak_values[i] *= 0.95
        
        return {
            'bar_values': bar_values,
            'peak_values': peak_values,
            'peak_timestamps': peak_timestamps
        }
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw detailed bass frequency zoom window (20-200 Hz) with peak hold"""
        # Background
        zoom_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, (15, 25, 40), zoom_rect)
        pygame.draw.rect(screen, (60, 80, 120), zoom_rect, 2)

        # Title
        if self.font_small:
            title = self.font_small.render("BASS DETAIL (20-200 Hz)", True, (200, 220, 255))
            screen.blit(title, (x + int(5 * ui_scale), y + int(5 * ui_scale)))

        # Reserve space for scale at bottom (35px for scale + labels)
        scale_height = int(35 * ui_scale)
        visualization_height = height - int(40 * ui_scale) - scale_height  # Title space + scale space

        # Use the enhanced bass detail data
        if hasattr(self, 'bass_bar_values') and len(self.bass_bar_values) > 0:
            # Simple approach: make bars fill entire width with no gaps
            total_width = width - int(20 * ui_scale)
            bar_width = total_width / self.bass_detail_bars  # Exact division
            max_height = visualization_height
            current_time = time.time()
            bar_bottom = y + height - scale_height - int(5 * ui_scale)  # Leave space for scale

            for j in range(self.bass_detail_bars):
                # Clamp amplitude to prevent overflow
                amplitude = min(self.bass_bar_values[j], 1.0)
                if j < len(self.bass_freq_ranges):
                    freq_range = self.bass_freq_ranges[j]
                else:
                    continue
                
                # Calculate exact bar position to eliminate gaps
                bar_height = min(int(amplitude * max_height), max_height)
                bar_x = x + int(10 * ui_scale) + j * bar_width
                bar_y = max(y + int(25 * ui_scale), bar_bottom - bar_height)
                
                # Smooth gradient color based on frequency position in bass range
                freq_center = (freq_range[0] + freq_range[1]) / 2
                
                # Calculate position in 20-200Hz range (0.0 to 1.0)
                bass_min = 20.0
                bass_max = 200.0
                freq_position = (freq_center - bass_min) / (bass_max - bass_min)
                freq_position = max(0.0, min(1.0, freq_position))  # Clamp to 0-1
                
                # Create smooth gradient: Purple -> Red -> Orange -> Yellow -> Green
                if freq_position < 0.2:  # 20-56Hz: Purple to Magenta
                    t = freq_position / 0.2
                    color = (
                        int(150 + 105 * t),  # 150->255 (purple to magenta)
                        int(50 + 50 * t),    # 50->100
                        int(255 - 55 * t)    # 255->200
                    )
                elif freq_position < 0.4:  # 56-92Hz: Magenta to Red  
                    t = (freq_position - 0.2) / 0.2
                    color = (
                        255,                  # Stay at 255 (red)
                        int(100 - 100 * t),  # 100->0 (lose green)
                        int(200 - 200 * t)   # 200->0 (lose blue)
                    )
                elif freq_position < 0.6:  # 92-128Hz: Red to Orange
                    t = (freq_position - 0.4) / 0.2
                    color = (
                        255,                 # Stay at 255 (red)
                        int(150 * t),       # 0->150 (add orange)
                        0                   # Stay at 0
                    )
                elif freq_position < 0.8:  # 128-164Hz: Orange to Yellow
                    t = (freq_position - 0.6) / 0.2
                    color = (
                        255,                 # Stay at 255
                        int(150 + 105 * t), # 150->255 (orange to yellow)
                        0                   # Stay at 0
                    )
                else:  # 164-200Hz: Yellow to Green
                    t = (freq_position - 0.8) / 0.2
                    color = (
                        int(255 - 55 * t),  # 255->200 (lose some red)
                        255,                # Stay at 255 (green)
                        int(100 * t)       # 0->100 (add some blue for lime)
                    )

                # Enhance brightness for kick detection
                if (
                    hasattr(self, "drum_info") and self.drum_info
                    and isinstance(self.drum_info.get("kick"), (int, float))
                    and self.drum_info.get("kick", 0) > 0.1
                    and freq_center <= 120
                ):
                    # Increase brightness by 30% for kick detection
                    color = tuple(min(255, int(c * 1.3)) for c in color)

                # Draw bar that exactly fills its space (round up width to ensure no gaps)
                actual_bar_width = int(bar_width) + 1 if j < self.bass_detail_bars - 1 else int(bar_width)
                pygame.draw.rect(
                    screen, color, (int(bar_x), bar_y, actual_bar_width, bar_height)
                )

                # Draw peak hold line
                if j < len(self.bass_peak_values) and self.bass_peak_values[j] > 0.05:
                    # Clamp peak value to prevent overflow
                    clamped_peak = min(self.bass_peak_values[j], 1.0)
                    peak_height = min(int(clamped_peak * max_height), max_height)
                    peak_y = max(y + int(25 * ui_scale), bar_bottom - peak_height)
                    pygame.draw.line(
                        screen,
                        (255, 255, 255, 180),
                        (int(bar_x), peak_y),
                        (int(bar_x + actual_bar_width - 1), peak_y),
                        1
                    )

            # Draw frequency scale line (now properly positioned)
            scale_y = y + height - scale_height
            pygame.draw.line(screen, (100, 100, 120), 
                           (x + int(10 * ui_scale), scale_y), 
                           (x + width - int(10 * ui_scale), scale_y), 2)
            
            # Draw background for scale area
            scale_bg_rect = pygame.Rect(x + int(5 * ui_scale), scale_y - 2, 
                                      width - int(10 * ui_scale), scale_height - 3)
            pygame.draw.rect(screen, (10, 15, 25), scale_bg_rect)
            
            # Frequency labels with tick marks
            if self.font_tiny:
                key_freqs = [20, 30, 40, 60, 80, 100, 150, 200]
                for target_freq in key_freqs:
                    # Find position for this frequency on logarithmic scale
                    log_pos = (np.log10(target_freq) - np.log10(20)) / (np.log10(200) - np.log10(20))
                    if 0 <= log_pos <= 1:
                        label_x = x + int(10 * ui_scale) + log_pos * (width - int(20 * ui_scale))
                        
                        # Draw tick mark
                        pygame.draw.line(screen, (150, 150, 160),
                                           (int(label_x), scale_y - 2),
                                           (int(label_x), scale_y + int(6 * ui_scale)), 2)
                        
                        # Draw label
                        freq_text = f"{target_freq}"
                        freq_surf = self.font_tiny.render(freq_text, True, (200, 220, 240))
                        text_rect = freq_surf.get_rect(centerx=int(label_x), top=scale_y + int(8 * ui_scale))
                        screen.blit(freq_surf, text_rect)
                
                # Add "Hz" label at the end
                hz_surf = self.font_tiny.render("Hz", True, (200, 220, 240))
                screen.blit(hz_surf, (x + width - int(30 * ui_scale), scale_y + int(8 * ui_scale)))
    
    def get_results(self) -> Dict[str, any]:
        """Get current bass analysis results"""
        return {
            'bar_values': self.bass_bar_values.copy(),
            'peak_values': self.bass_peak_values.copy(),
            'freq_ranges': self.bass_freq_ranges.copy(),
            'detail_bars': self.bass_detail_bars
        }
    
    def shutdown(self):
        """Clean shutdown of background thread"""
        self.bass_thread_running = False
        try:
            self.bass_processing_queue.put_nowait(None)  # Signal thread to exit
        except:
            pass
        if self.bass_processing_thread.is_alive():
            self.bass_processing_thread.join(timeout=1.0)