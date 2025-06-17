"""
Transient Detection Panel - Attack detection and dynamics visualization
"""

import numpy as np
import pygame
from collections import deque
from scipy import signal


class TransientDetectionPanel:
    """Panel for transient detection, attack analysis, and dynamics visualization"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.panel_height = 160
        self.is_frozen = False
        
        # Transient detection
        self.transient_detected = False
        self.transient_strength = 0.0
        self.transient_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        
        # Attack characteristics
        self.attack_time = 0.0  # ms
        self.decay_time = 0.0   # ms
        self.transient_type = "None"  # Kick, Snare, HiHat, Perc, Other
        
        # Dynamic range tracking
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.crest_factor = 0.0
        self.dynamic_range = 0.0
        
        # Envelope follower
        self.envelope = 0.0
        self.envelope_history = deque(maxlen=300)
        
        # Transient events
        self.transient_events = deque(maxlen=50)  # Store recent transients
        self.last_transient_time = 0
        
        # Energy bands for transient classification
        self.band_energies = np.zeros(4)  # Sub, Low, Mid, High
        
        # Colors
        self.bg_color = (18, 15, 20)
        self.transient_color = (255, 150, 50)
        self.envelope_color = (100, 200, 255)
        self.peak_color = (255, 100, 100)
        
        # Fonts (will be set by main app)
        self.fonts = None
        
        # Update control
        self.update_counter = 0
        self.update_interval = 1  # Update every frame for responsiveness
        
        # Detection parameters
        self.sensitivity = 1.5  # Lower threshold for better detection
        self.prev_envelope = 0.0
        self.attack_threshold = 0.1  # Minimum attack time for transient
        self.spectral_flux_history = deque(maxlen=10)  # For spectral flux detection
    
    def set_fonts(self, fonts):
        """Set fonts for rendering"""
        self.fonts = fonts
    
    def update(self, audio_data, spectrum=None):
        """Update transient detection analysis"""
        if self.is_frozen or len(audio_data) == 0:
            return
        
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return
        
        # Calculate envelope
        self.envelope = self._calculate_envelope(audio_data)
        self.envelope_history.append(self.envelope)
        
        # Calculate levels
        self.peak_level = np.max(np.abs(audio_data))
        self.rms_level = np.sqrt(np.mean(audio_data**2))
        
        # Crest factor (peak/RMS ratio in dB)
        if self.rms_level > 0:
            self.crest_factor = 20 * np.log10(self.peak_level / self.rms_level)
        
        # Detect transients
        self._detect_transient(audio_data)
        
        # Update transient history
        self.transient_history.append(self.transient_strength)
        
        # Classify transient if detected
        if self.transient_detected:
            self._classify_transient(audio_data, spectrum)
            self._measure_attack_decay(audio_data)
            
            # Record transient event
            import time
            self.transient_events.append({
                'time': time.time(),
                'type': self.transient_type,
                'strength': self.transient_strength,
                'attack': self.attack_time
            })
    
    def _calculate_envelope(self, audio_data):
        """Calculate envelope using Hilbert transform"""
        # Use Hilbert transform for analytic signal
        analytic = signal.hilbert(audio_data)
        envelope = np.abs(analytic)
        
        # Smooth the envelope
        if len(envelope) > 10:
            envelope = signal.savgol_filter(envelope, 
                                          min(11, len(envelope) // 4 * 2 + 1), 3)
        
        return np.mean(envelope)
    
    def _detect_transient(self, audio_data):
        """Detect transients using improved spectral flux method"""
        # Method 1: Energy-based detection with derivative
        current_energy = np.sum(audio_data**2)
        
        # Check for sudden energy increase
        if self.prev_envelope > 0:
            energy_ratio = self.envelope / self.prev_envelope
            
            if energy_ratio > self.sensitivity:
                self.transient_detected = True
                self.transient_strength = min(1.0, (energy_ratio - 1.0) / 2.0)
            else:
                self.transient_detected = False
                self.transient_strength *= 0.85  # Slower decay
        
        # Method 2: Spectral flux detection
        if len(audio_data) >= 512:
            # Compute spectrum
            window = np.hanning(min(512, len(audio_data)))
            windowed = audio_data[:len(window)] * window
            spectrum = np.abs(np.fft.rfft(windowed))
            
            # Calculate spectral flux
            if len(self.spectral_flux_history) > 0:
                prev_spectrum = self.spectral_flux_history[-1]
                if len(spectrum) == len(prev_spectrum):
                    # Positive spectral flux
                    flux = np.sum(np.maximum(0, spectrum - prev_spectrum))
                    avg_flux = np.mean([np.sum(s) for s in self.spectral_flux_history])
                    
                    if avg_flux > 0 and flux > avg_flux * 1.8:  # Threshold for flux
                        self.transient_detected = True
                        self.transient_strength = max(self.transient_strength, 
                                                    min(1.0, flux / (avg_flux * 3)))
            
            self.spectral_flux_history.append(spectrum[:128])  # Store first 128 bins
        
        self.prev_envelope = self.envelope
        
        # Alternative: High-frequency content detection
        if len(audio_data) > 256:
            # High-pass filter to isolate transients
            nyquist = self.sample_rate / 2
            high_cutoff = 5000 / nyquist
            
            if high_cutoff < 1.0:
                b, a = signal.butter(4, high_cutoff, btype='high')
                high_freq = signal.filtfilt(b, a, audio_data)
                
                hf_energy = np.sum(high_freq**2)
                total_energy = np.sum(audio_data**2)
                
                if total_energy > 0:
                    hf_ratio = hf_energy / total_energy
                    if hf_ratio > 0.3:  # Significant HF content
                        self.transient_detected = True
                        self.transient_strength = max(self.transient_strength, hf_ratio)
    
    def _classify_transient(self, audio_data, spectrum):
        """Classify the type of transient"""
        if spectrum is None:
            # Compute spectrum if not provided
            window = np.hanning(len(audio_data))
            windowed = audio_data * window
            spectrum = np.abs(np.fft.rfft(windowed))
        
        # Calculate energy in different frequency bands
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        # Define band boundaries
        band_limits = [0, 100, 250, 2000, self.sample_rate/2]
        
        for i in range(4):
            low_idx = np.searchsorted(freqs, band_limits[i])
            high_idx = np.searchsorted(freqs, band_limits[i+1])
            
            if low_idx < len(spectrum) and high_idx < len(spectrum):
                self.band_energies[i] = np.sum(spectrum[low_idx:high_idx])
        
        # Normalize band energies
        total_energy = np.sum(self.band_energies)
        if total_energy > 0:
            self.band_energies /= total_energy
        
        # Classify based on energy distribution
        if self.band_energies[0] > 0.6:  # Sub-bass dominant
            self.transient_type = "Kick"
        elif self.band_energies[1] > 0.4 and self.band_energies[2] > 0.3:
            self.transient_type = "Snare"
        elif self.band_energies[3] > 0.5:  # High frequency dominant
            self.transient_type = "HiHat"
        elif self.band_energies[2] > 0.4:
            self.transient_type = "Perc"
        else:
            self.transient_type = "Other"
    
    def _measure_attack_decay(self, audio_data):
        """Measure attack and decay times"""
        if len(audio_data) < 100:
            return
        
        # Find peak position
        peak_idx = np.argmax(np.abs(audio_data))
        
        # Measure attack time (10% to 90% of peak)
        if peak_idx > 0:
            peak_val = abs(audio_data[peak_idx])
            
            # Find 10% point before peak
            start_idx = 0
            for i in range(peak_idx-1, -1, -1):
                if abs(audio_data[i]) < 0.1 * peak_val:
                    start_idx = i
                    break
            
            # Attack time in ms
            self.attack_time = (peak_idx - start_idx) * 1000 / self.sample_rate
        
        # Measure decay time (peak to 10%)
        if peak_idx < len(audio_data) - 1:
            peak_val = abs(audio_data[peak_idx])
            
            # Find 10% point after peak
            end_idx = len(audio_data) - 1
            for i in range(peak_idx + 1, len(audio_data)):
                if abs(audio_data[i]) < 0.1 * peak_val:
                    end_idx = i
                    break
            
            # Decay time in ms
            self.decay_time = (end_idx - peak_idx) * 1000 / self.sample_rate
    
    def draw(self, screen, x, y, width, height=None, panel_color=None):
        """Draw the transient detection panel"""
        if not self.fonts:
            return
        
        # Import panel utilities
        from .panel_utils import draw_panel_header, draw_panel_background
        
        # Use provided height or default
        if height is None:
            height = self.panel_height
            
        # Draw background with blue tint
        draw_panel_background(screen, x, y, width, height,
                            bg_color=(25, 30, 40), border_color=(80, 120, 160), alpha=230)
        
        # Draw centered header
        font_medium = self.fonts.get('medium', self.fonts['small'])
        y_offset = draw_panel_header(screen, "Transient Detection", font_medium,
                                   x, y, width, bg_color=(25, 30, 40),
                                   border_color=(80, 120, 160),
                                   text_color=(180, 220, 255),
                                   frozen=self.is_frozen)
        
        y_offset += 10  # Small gap after header
        
        # Transient indicator with detection type
        if self.transient_detected:
            # Large visual indicator
            indicator_size = 50
            indicator_x = x + width - indicator_size - 20
            pygame.draw.circle(screen, self.transient_color, 
                             (indicator_x, y_offset + 20), 
                             int(15 + self.transient_strength * 10))
            
            # Type text
            type_surface = self.fonts.get('large', self.fonts['medium']).render(
                self.transient_type, True, (220, 220, 220)
            )
            screen.blit(type_surface, (x + 20, y_offset))
        else:
            # No transient text
            no_trans_surface = self.fonts['small'].render(
                "No transient", True, (120, 120, 120)
            )
            screen.blit(no_trans_surface, (x + 20, y_offset))
        
        y_offset += 50
        
        # Level meters section with more spacing
        meter_height = 12
        meter_width = width - 40
        meter_spacing = 25
        
        # Peak level meter
        self._draw_level_meter(screen, x + 20, y_offset, meter_width, meter_height,
                             self.peak_level, "Peak Level", self.peak_color)
        y_offset += meter_spacing
        
        # RMS level meter
        self._draw_level_meter(screen, x + 20, y_offset, meter_width, meter_height,
                             self.rms_level, "RMS Level", self.envelope_color)
        y_offset += meter_spacing
        
        # Crest factor and timing info
        info_y = y_offset
        crest_text = f"Crest Factor: {self.crest_factor:.1f} dB"
        crest_surface = self.fonts['small'].render(crest_text, True, (180, 180, 180))
        screen.blit(crest_surface, (x + 20, info_y))
        
        if self.transient_detected:
            timing_text = f"Attack: {self.attack_time:.1f}ms / Decay: {self.decay_time:.1f}ms"
            timing_surface = self.fonts['small'].render(timing_text, True, (180, 180, 180))
            screen.blit(timing_surface, (x + 20, info_y + 20))
        
        y_offset += 50
        
        # Envelope and transient history graph - use more vertical space
        graph_height = min(120, height - (y_offset - y) - 80)
        graph_width = width - 40
        if graph_height > 40:
            self._draw_history_graph(screen, x + 20, y_offset, graph_width, graph_height)
            y_offset += graph_height + 20
        
        # Band energy visualization at bottom
        band_height = min(60, height - (y_offset - y) - 20)
        if band_height > 30:
            self._draw_band_energies(screen, x + 20, y_offset, width - 40, band_height)
    
    def _draw_level_meter(self, screen, x, y, width, height, level, label, color):
        """Draw a level meter"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Level bar
        level_width = int(level * width)
        pygame.draw.rect(screen, color, (x, y, level_width, height))
        
        # Label
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 12))
    
    def _draw_history_graph(self, screen, x, y, width, height):
        """Draw envelope and transient history"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Draw envelope history
        if len(self.envelope_history) > 1:
            points = []
            max_env = max(self.envelope_history) if max(self.envelope_history) > 0 else 1
            
            for i, env in enumerate(self.envelope_history):
                px = x + (i * width // len(self.envelope_history))
                py = y + height - int(env / max_env * height * 0.8)
                points.append((px, py))
            
            pygame.draw.lines(screen, self.envelope_color, False, points, 1)
        
        # Draw transient markers
        for i, strength in enumerate(self.transient_history):
            if strength > 0.1:
                px = x + (i * width // len(self.transient_history))
                marker_height = int(strength * height)
                pygame.draw.line(screen, self.transient_color,
                               (px, y + height - marker_height), 
                               (px, y + height), 2)
        
        # Label
        label = "Envelope & Transients"
        label_surface = self.fonts['tiny'].render(label, True, (150, 150, 150))
        screen.blit(label_surface, (x, y - 12))
    
    def _draw_band_energies(self, screen, x, y, width, height):
        """Draw frequency band energy distribution"""
        # Background
        pygame.draw.rect(screen, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 60), (x, y, width, height), 1)
        
        # Draw bars for each band
        band_names = ["Sub", "Low", "Mid", "High"]
        bar_width = width // 4 - 4
        
        for i, (energy, name) in enumerate(zip(self.band_energies, band_names)):
            bar_x = x + 2 + i * (bar_width + 4)
            bar_height = int(energy * (height - 10))
            bar_y = y + height - bar_height - 5
            
            # Color based on band
            colors = [(150, 50, 50), (150, 150, 50), (50, 150, 50), (50, 50, 150)]
            
            pygame.draw.rect(screen, colors[i], 
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Band label
            label_surface = self.fonts['tiny'].render(name, True, (100, 100, 100))
            label_rect = label_surface.get_rect(centerx=bar_x + bar_width // 2,
                                              bottom=y + height - 2)
            screen.blit(label_surface, label_rect)
        
        # Title
        title_surface = self.fonts['tiny'].render("Freq Bands", True, (150, 150, 150))
        screen.blit(title_surface, (x, y - 12))
    
    def get_status(self):
        """Get current transient detection status"""
        return {
            'detected': self.transient_detected,
            'strength': self.transient_strength,
            'type': self.transient_type,
            'attack_time': self.attack_time,
            'crest_factor': self.crest_factor,
            'recent_events': list(self.transient_events)
        }