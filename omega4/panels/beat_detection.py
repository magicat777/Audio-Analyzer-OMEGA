"""
Beat Detection and BPM Analysis Panel for OMEGA-4 Audio Analyzer
Real-time beat tracking, BPM calculation, and rhythmic pattern analysis
Designed for DJs and music producers
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import time


class BeatDetector:
    """Real-time beat detection using onset detection and tempo tracking"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Beat detection parameters
        self.onset_threshold = 0.3
        self.min_bpm = 60
        self.max_bpm = 200
        self.bpm_smoothing = 0.8  # Higher = more stable BPM
        
        # Onset detection
        self.prev_spectrum = None
        self.onset_strength_history = deque(maxlen=100)  # ~1.6 seconds at 60fps
        self.peak_pick_threshold = 0.5
        self.peak_pick_delta = 0.1
        
        # Beat tracking
        self.beat_times = deque(maxlen=32)  # Store last 32 beats
        self.current_bpm = 120.0
        self.bpm_confidence = 0.0
        self.beat_phase = 0.0  # Phase within current beat cycle
        
        # Tempo analysis
        self.tempo_candidates = {}  # BPM -> confidence mapping
        self.tempo_history = deque(maxlen=50)
        
        # Rhythmic pattern analysis
        self.downbeat_confidence = 0.0
        self.time_signature = (4, 4)  # Default 4/4
        self.rhythmic_complexity = 0.0
        
        # Energy analysis for beat strength
        self.energy_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000)
        }
        self.energy_history = {band: deque(maxlen=20) for band in self.energy_bands}
        
        # Beat strength and quality metrics
        self.beat_strength = 0.0
        self.beat_regularity = 0.0
        self.syncopation_level = 0.0
        
    def compute_onset_strength(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Compute spectral onset strength using spectral flux"""
        if self.prev_spectrum is None:
            self.prev_spectrum = spectrum.copy()
            return 0.0
        
        # Spectral flux (positive differences only)
        flux = np.sum(np.maximum(0, spectrum - self.prev_spectrum))
        
        # Normalize by spectral energy
        spectral_energy = np.sum(spectrum)
        if spectral_energy > 0:
            flux = flux / spectral_energy
        
        self.prev_spectrum = spectrum.copy()
        return flux
    
    def analyze_energy_bands(self, spectrum: np.ndarray, freqs: np.ndarray):
        """Analyze energy in different frequency bands for beat detection"""
        for band_name, (low_freq, high_freq) in self.energy_bands.items():
            # Find frequency indices
            low_idx = np.argmax(freqs >= low_freq)
            high_idx = np.argmax(freqs >= high_freq)
            
            if high_idx == 0:  # If high_freq is beyond spectrum
                high_idx = len(spectrum)
            
            # Calculate band energy
            band_energy = np.sum(spectrum[low_idx:high_idx])
            self.energy_history[band_name].append(band_energy)
    
    def detect_beats(self, spectrum: np.ndarray, freqs: np.ndarray, 
                    audio_chunk: np.ndarray) -> Tuple[bool, float]:
        """Detect beats using onset detection and energy analysis"""
        current_time = time.time()
        
        # Analyze energy bands
        self.analyze_energy_bands(spectrum, freqs)
        
        # Compute onset strength
        onset_strength = self.compute_onset_strength(spectrum, freqs)
        self.onset_strength_history.append(onset_strength)
        
        # Beat detection using adaptive threshold
        is_beat = False
        beat_strength = 0.0
        
        if len(self.onset_strength_history) > 10:
            # Adaptive threshold based on recent history
            recent_onsets = list(self.onset_strength_history)[-10:]
            mean_onset = np.mean(recent_onsets)
            std_onset = np.std(recent_onsets)
            
            threshold = mean_onset + (self.onset_threshold * std_onset)
            
            # Check for beat
            if (onset_strength > threshold and 
                onset_strength > self.peak_pick_threshold):
                
                # Additional validation using energy bands
                bass_energy = np.mean(list(self.energy_history['bass'])[-3:]) if self.energy_history['bass'] else 0
                kick_strength = self._estimate_kick_strength(spectrum, freqs)
                
                # Combined beat strength
                beat_strength = (onset_strength * 0.6 + 
                               bass_energy * 0.3 + 
                               kick_strength * 0.1)
                
                # Minimum time between beats (prevents double triggers)
                if not self.beat_times or (current_time - self.beat_times[-1]) > 0.2:
                    is_beat = True
                    self.beat_times.append(current_time)
                    self.beat_strength = beat_strength
        
        return is_beat, beat_strength
    
    def _estimate_kick_strength(self, spectrum: np.ndarray, freqs: np.ndarray) -> float:
        """Estimate kick drum strength in typical kick frequency range"""
        # Kick drums typically have energy around 60-120 Hz
        kick_low = np.argmax(freqs >= 50)
        kick_high = np.argmax(freqs >= 120)
        
        if kick_high == 0:
            kick_high = len(spectrum)
        
        kick_energy = np.sum(spectrum[kick_low:kick_high])
        total_energy = np.sum(spectrum)
        
        return kick_energy / total_energy if total_energy > 0 else 0.0
    
    def calculate_bpm(self) -> Tuple[float, float]:
        """Calculate BPM from beat intervals with confidence measure"""
        if len(self.beat_times) < 4:
            return self.current_bpm, 0.0
        
        # Calculate intervals between beats
        intervals = []
        beat_list = list(self.beat_times)
        for i in range(1, len(beat_list)):
            interval = beat_list[i] - beat_list[i-1]
            if 0.3 <= interval <= 1.0:  # Valid beat intervals (60-200 BPM)
                intervals.append(interval)
        
        if not intervals:
            return self.current_bpm, 0.0
        
        # Convert intervals to BPM
        bpms = [60.0 / interval for interval in intervals]
        
        # Find most consistent BPM
        bpm_candidates = {}
        tolerance = 3.0  # BPM tolerance for grouping
        
        for bpm in bpms:
            # Round to nearest 0.5 BPM for clustering
            rounded_bpm = round(bpm * 2) / 2
            
            if rounded_bpm not in bpm_candidates:
                bpm_candidates[rounded_bpm] = []
            bpm_candidates[rounded_bpm].append(bpm)
        
        # Find most frequent BPM cluster
        best_bpm = self.current_bpm
        best_confidence = 0.0
        
        for candidate_bpm, bpm_list in bpm_candidates.items():
            if len(bpm_list) >= 2:  # Need at least 2 consistent beats
                confidence = len(bpm_list) / len(bpms)
                variance = np.var(bpm_list)
                
                # Penalize high variance
                confidence *= max(0.1, 1.0 - variance / 100.0)
                
                if confidence > best_confidence:
                    best_bpm = np.mean(bpm_list)
                    best_confidence = confidence
        
        # Smooth BPM changes
        if best_confidence > 0.3:
            self.current_bpm = (self.bpm_smoothing * self.current_bpm + 
                              (1 - self.bpm_smoothing) * best_bpm)
            self.bpm_confidence = best_confidence
        
        self.tempo_history.append((self.current_bpm, self.bpm_confidence))
        
        return self.current_bpm, self.bpm_confidence
    
    def analyze_rhythmic_patterns(self) -> Dict[str, float]:
        """Analyze rhythmic complexity and patterns"""
        if len(self.beat_times) < 8:
            return {
                'regularity': 0.0,
                'complexity': 0.0,
                'syncopation': 0.0,
                'groove_strength': 0.0
            }
        
        # Calculate beat regularity
        intervals = []
        beat_list = list(self.beat_times)[-8:]  # Last 8 beats
        
        for i in range(1, len(beat_list)):
            intervals.append(beat_list[i] - beat_list[i-1])
        
        if intervals:
            mean_interval = np.mean(intervals)
            interval_variance = np.var(intervals)
            regularity = max(0.0, 1.0 - (interval_variance / (mean_interval ** 2)))
            self.beat_regularity = regularity
        
        # Rhythmic complexity based on onset pattern
        if len(self.onset_strength_history) > 20:
            onset_array = np.array(list(self.onset_strength_history)[-20:])
            
            # Calculate spectral complexity of onset pattern
            onset_fft = np.abs(np.fft.fft(onset_array))
            spectral_centroid = np.sum(onset_fft * np.arange(len(onset_fft))) / np.sum(onset_fft)
            
            self.rhythmic_complexity = min(1.0, spectral_centroid / len(onset_fft))
        
        # Syncopation detection (off-beat emphasis)
        if len(self.beat_times) >= 4:
            self.syncopation_level = self._calculate_syncopation()
        
        # Overall groove strength
        groove_strength = (self.beat_regularity * 0.4 + 
                          self.rhythmic_complexity * 0.3 + 
                          self.syncopation_level * 0.3)
        
        return {
            'regularity': self.beat_regularity,
            'complexity': self.rhythmic_complexity,
            'syncopation': self.syncopation_level,
            'groove_strength': groove_strength
        }
    
    def _calculate_syncopation(self) -> float:
        """Calculate syncopation level based off-beat emphasis"""
        # Simplified syncopation detection
        if len(self.onset_strength_history) < 16:
            return 0.0
        
        # Look for strong onsets in off-beat positions
        onset_pattern = np.array(list(self.onset_strength_history)[-16:])
        
        # Expected beat positions (every 4th sample for 4/4 time)
        beat_positions = [0, 4, 8, 12]
        off_beat_positions = [2, 6, 10, 14]
        
        beat_strength = np.mean([onset_pattern[pos] for pos in beat_positions if pos < len(onset_pattern)])
        off_beat_strength = np.mean([onset_pattern[pos] for pos in off_beat_positions if pos < len(onset_pattern)])
        
        if beat_strength > 0:
            syncopation = off_beat_strength / beat_strength
            return min(1.0, syncopation)
        
        return 0.0
    
    def get_beat_phase(self) -> float:
        """Get current phase within beat cycle (0.0 to 1.0)"""
        if len(self.beat_times) < 2 or self.current_bpm <= 0:
            return 0.0
        
        current_time = time.time()
        last_beat = self.beat_times[-1]
        beat_duration = 60.0 / self.current_bpm
        
        time_since_beat = current_time - last_beat
        self.beat_phase = (time_since_beat / beat_duration) % 1.0
        
        return self.beat_phase


class BeatDetectionPanel:
    """OMEGA-4 Beat Detection and BPM Analysis Panel"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.detector = BeatDetector(sample_rate)
        
        # Display state
        self.beat_info = {
            'current_bpm': 120.0,
            'bpm_confidence': 0.0,
            'is_beat': False,
            'beat_strength': 0.0,
            'beat_phase': 0.0,
            'rhythmic_patterns': {},
            'last_beat_time': 0.0
        }
        
        # Visual elements
        self.beat_flash_time = 0
        self.beat_flash_duration = 15  # frames
        self.bpm_history_display = deque(maxlen=100)
        
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
    
    def update(self, fft_data: np.ndarray, audio_chunk: np.ndarray, 
               frequencies: np.ndarray):
        """Update beat detection with new audio data"""
        if fft_data is not None and len(fft_data) > 0 and audio_chunk is not None:
            # Check for silence
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            if rms < 0.001:
                self.beat_info = {
                    'current_bpm': 120.0,
                    'bpm_confidence': 0.0,
                    'is_beat': False,
                    'beat_strength': 0.0,
                    'beat_phase': 0.0,
                    'rhythmic_patterns': {},
                    'last_beat_time': 0.0
                }
                return
            
            # Detect beats
            is_beat, beat_strength = self.detector.detect_beats(
                fft_data, frequencies, audio_chunk
            )
            
            # Calculate BPM
            current_bpm, bpm_confidence = self.detector.calculate_bpm()
            
            # Analyze rhythmic patterns
            rhythmic_patterns = self.detector.analyze_rhythmic_patterns()
            
            # Get beat phase
            beat_phase = self.detector.get_beat_phase()
            
            # Update display info
            self.beat_info = {
                'current_bpm': current_bpm,
                'bpm_confidence': bpm_confidence,
                'is_beat': is_beat,
                'beat_strength': beat_strength,
                'beat_phase': beat_phase,
                'rhythmic_patterns': rhythmic_patterns,
                'last_beat_time': self.detector.beat_times[-1] if self.detector.beat_times else 0.0
            }
            
            # Update visual elements
            if is_beat:
                self.beat_flash_time = self.beat_flash_duration
            elif self.beat_flash_time > 0:
                self.beat_flash_time -= 1
            
            # Store BPM for history display
            self.bpm_history_display.append(current_bpm)
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw beat detection panel"""
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Background with beat-sensitive color
        bg_color = (30, 20, 40) if not self.beat_info['is_beat'] else (60, 40, 80)
        
        # Border with beat flash
        border_color = (100, 70, 130)
        if self.beat_flash_time > 0:
            flash_intensity = self.beat_flash_time / self.beat_flash_duration
            border_color = (
                int(100 + 155 * flash_intensity),
                int(70 + 185 * flash_intensity),
                int(130 + 125 * flash_intensity)
            )
        
        # Draw background
        draw_panel_background(screen, x, y, width, height,
                            bg_color=bg_color, border_color=border_color)
        
        # Draw centered header
        if self.font_medium:
            title_color = (255, 200, 255) if self.beat_flash_time > 0 else (200, 150, 200)
            y_offset = draw_panel_header(screen, "Beat Detection & BPM", self.font_medium,
                                       x, y, width, bg_color=bg_color,
                                       border_color=border_color,
                                       text_color=title_color)
        else:
            y_offset = y + 35
        
        y_offset += int(10 * ui_scale)  # Small gap after header
        
        # Current BPM
        if self.font_large:
            bpm = self.beat_info['current_bpm']
            confidence = self.beat_info['bpm_confidence']
            
            # Color based on confidence
            bpm_color = (
                (255, 100, 100) if confidence < 0.3 else
                (255, 255, 100) if confidence < 0.6 else
                (100, 255, 100)
            )
            
            bpm_text = f"{bpm:.1f} BPM"
            bpm_surf = self.font_large.render(bpm_text, True, bpm_color)
            screen.blit(bpm_surf, (x + int(15 * ui_scale), y_offset))
            y_offset += int(40 * ui_scale)
        
        # BPM Confidence
        if self.font_small:
            conf_text = f"Confidence: {self.beat_info['bpm_confidence']:.0%}"
            conf_surf = self.font_small.render(conf_text, True, (180, 180, 200))
            screen.blit(conf_surf, (x + int(15 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        # Beat phase indicator (circular)
        self._draw_beat_phase_indicator(screen, x + width - int(80 * ui_scale), 
                                       y + int(20 * ui_scale), int(60 * ui_scale), ui_scale)
        
        # Rhythmic analysis
        patterns = self.beat_info.get('rhythmic_patterns', {})
        if patterns and self.font_tiny:
            y_offset += int(10 * ui_scale)
            
            # Draw rhythmic metrics
            metrics = [
                ('Regularity', patterns.get('regularity', 0.0)),
                ('Complexity', patterns.get('complexity', 0.0)),
                ('Syncopation', patterns.get('syncopation', 0.0)),
                ('Groove', patterns.get('groove_strength', 0.0))
            ]
            
            for i, (name, value) in enumerate(metrics):
                metric_y = y_offset + i * int(18 * ui_scale)
                
                # Metric name
                name_surf = self.font_tiny.render(f"{name}:", True, (160, 160, 180))
                screen.blit(name_surf, (x + int(15 * ui_scale), metric_y))
                
                # Metric bar
                bar_x = x + int(80 * ui_scale)
                bar_width = int(100 * ui_scale)
                bar_height = int(12 * ui_scale)
                
                # Background
                pygame.draw.rect(screen, (40, 40, 60), 
                               (bar_x, metric_y, bar_width, bar_height))
                
                # Value bar
                value_width = int(bar_width * min(value, 1.0))
                if value_width > 0:
                    color = self._get_metric_color(name, value)
                    pygame.draw.rect(screen, color, 
                                   (bar_x, metric_y, value_width, bar_height))
                
                # Value text
                value_text = f"{value:.1%}"
                value_surf = self.font_tiny.render(value_text, True, (140, 140, 160))
                screen.blit(value_surf, (bar_x + bar_width + int(5 * ui_scale), metric_y))
            
            y_offset += int(80 * ui_scale)
        
        # BPM history graph (mini)
        if len(self.bpm_history_display) > 10:
            self._draw_bpm_history(screen, x + int(15 * ui_scale), y_offset, 
                                 width - int(30 * ui_scale), int(40 * ui_scale), ui_scale)
    
    def _draw_beat_phase_indicator(self, screen: pygame.Surface, center_x: int, center_y: int, 
                                  size: int, ui_scale: float):
        """Draw circular beat phase indicator"""
        radius = size // 2
        phase = self.beat_info['beat_phase']
        
        # Background circle
        pygame.draw.circle(screen, (60, 60, 80), (center_x, center_y), radius, 2)
        
        # Phase indicator
        angle = phase * 2 * math.pi - math.pi / 2  # Start at top
        indicator_x = center_x + int((radius - 5) * math.cos(angle))
        indicator_y = center_y + int((radius - 5) * math.sin(angle))
        
        # Beat indicator dot
        dot_color = (255, 100, 100) if self.beat_flash_time > 0 else (100, 255, 100)
        pygame.draw.circle(screen, dot_color, (indicator_x, indicator_y), 4)
        
        # Center dot
        pygame.draw.circle(screen, (200, 200, 200), (center_x, center_y), 3)
        
        # Phase text
        if self.font_tiny:
            phase_text = f"{phase:.2f}"
            phase_surf = self.font_tiny.render(phase_text, True, (180, 180, 200))
            phase_rect = phase_surf.get_rect(center=(center_x, center_y + radius + 15))
            screen.blit(phase_surf, phase_rect)
    
    def _get_metric_color(self, metric_name: str, value: float) -> Tuple[int, int, int]:
        """Get color for rhythmic metric based on value"""
        if metric_name == 'Regularity':
            # Green for high regularity
            return (int(255 * (1 - value)), int(255 * value), 100)
        elif metric_name == 'Complexity':
            # Blue to yellow for complexity
            return (int(255 * value), int(255 * value), int(255 * (1 - value)))
        elif metric_name == 'Syncopation':
            # Purple for syncopation
            return (int(255 * value), 100, int(255 * value))
        else:  # Groove
            # Orange for groove strength
            return (255, int(200 * value), int(100 * value))
    
    def _draw_bpm_history(self, screen: pygame.Surface, x: int, y: int, 
                         width: int, height: int, ui_scale: float):
        """Draw mini BPM history graph"""
        if len(self.bpm_history_display) < 2:
            return
        
        # Background
        pygame.draw.rect(screen, (20, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 1)
        
        # Get BPM range
        bpm_values = list(self.bpm_history_display)
        min_bpm = max(60, min(bpm_values) - 10)
        max_bpm = min(200, max(bpm_values) + 10)
        bpm_range = max_bpm - min_bpm
        
        if bpm_range > 0:
            # Draw BPM line
            points = []
            for i, bpm in enumerate(bpm_values):
                px = x + int((i / (len(bpm_values) - 1)) * width)
                py = y + height - int(((bpm - min_bpm) / bpm_range) * height)
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(screen, (100, 255, 100), False, points, 2)
        
        # Label
        if self.font_tiny:
            label_text = f"BPM History ({min_bpm:.0f}-{max_bpm:.0f})"
            label_surf = self.font_tiny.render(label_text, True, (140, 140, 160))
            screen.blit(label_surf, (x + 2, y + 2))
    
    def get_results(self) -> Dict[str, Any]:
        """Get current beat detection results"""
        return self.beat_info.copy()