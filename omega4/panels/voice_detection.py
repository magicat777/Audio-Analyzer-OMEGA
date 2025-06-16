"""
Voice Detection Panel - Real-time voice activity and formant analysis
"""

import numpy as np
import pygame
from collections import deque
from scipy import signal


class VoiceDetectionPanel:
    """Panel for voice detection, formant tracking, and vocal analysis"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.panel_height = 180
        self.is_frozen = False
        
        # Voice detection state
        self.voice_active = False
        self.voice_confidence = 0.0
        self.voice_history = deque(maxlen=120)  # 2 seconds at 60 FPS
        
        # Formant tracking
        self.formants = []  # F1, F2, F3, F4
        self.formant_history = deque(maxlen=60)
        
        # Pitch tracking for voice
        self.voice_pitch = 0.0
        self.pitch_history = deque(maxlen=120)
        
        # Voice characteristics
        self.voice_type = "Unknown"  # Bass, Baritone, Tenor, Alto, Soprano
        self.vocal_quality = 0.0  # 0-1, breathiness/clarity
        
        # Colors
        self.bg_color = (15, 20, 18)
        self.voice_color = (100, 255, 100)
        self.formant_colors = [
            (255, 100, 100),  # F1 - Red
            (100, 255, 100),  # F2 - Green
            (100, 100, 255),  # F3 - Blue
            (255, 255, 100),  # F4 - Yellow
        ]
        
        # Fonts (will be set by main app)
        self.fonts = None
        
        # Update control
        self.update_counter = 0
        self.update_interval = 2  # Update every 2 frames
    
    def set_fonts(self, fonts):
        """Set fonts for rendering"""
        self.fonts = fonts
    
    def detect_voice(self, audio_data, spectrum):
        """Detect voice presence and characteristics"""
        # Simple voice detection based on spectral characteristics
        # Real implementation would use more sophisticated methods
        
        # Check for energy in voice frequency range (85-255 Hz fundamental)
        voice_freq_range = (85, 255)
        freq_bins = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        voice_start = np.searchsorted(freq_bins, voice_freq_range[0])
        voice_end = np.searchsorted(freq_bins, voice_freq_range[1])
        
        if voice_start < len(spectrum) and voice_end < len(spectrum):
            voice_energy = np.sum(spectrum[voice_start:voice_end])
            total_energy = np.sum(spectrum)
            
            # Voice detection heuristics
            if total_energy > 0:
                voice_ratio = voice_energy / total_energy
                
                # Check for formant structure
                formant_score = self._detect_formant_structure(spectrum, freq_bins)
                
                # Combine metrics
                self.voice_confidence = min(100, (voice_ratio * 200 + formant_score * 100) / 2)
                self.voice_active = self.voice_confidence > 30
            else:
                self.voice_active = False
                self.voice_confidence = 0.0
        
        return self.voice_active, self.voice_confidence
    
    def _detect_formant_structure(self, spectrum, freq_bins):
        """Detect formant frequencies characteristic of voice"""
        # Simplified formant detection
        # Real implementation would use LPC or cepstral analysis
        
        # Find peaks in spectrum
        if len(spectrum) > 10:
            # Smooth spectrum for peak detection
            smoothed = signal.savgol_filter(spectrum, min(11, len(spectrum) // 4 * 2 + 1), 3)
            peaks, properties = signal.find_peaks(smoothed, prominence=np.max(smoothed) * 0.1)
            
            # Look for formant-like peaks in expected ranges
            formant_ranges = [
                (200, 900),    # F1
                (800, 2500),   # F2
                (2000, 3500),  # F3
                (3000, 4500),  # F4
            ]
            
            formants_found = []
            for f_range in formant_ranges:
                for peak in peaks:
                    if peak < len(freq_bins):
                        freq = freq_bins[peak]
                        if f_range[0] <= freq <= f_range[1]:
                            formants_found.append(freq)
                            break
            
            self.formants = formants_found[:4]  # Keep first 4 formants
            
            # Score based on number of formants found
            return len(formants_found) / 4.0
        
        return 0.0
    
    def analyze_voice_type(self):
        """Determine voice type based on pitch and formants"""
        if self.voice_pitch > 0:
            if self.voice_pitch < 110:  # Below A2
                self.voice_type = "Bass"
            elif self.voice_pitch < 165:  # Below E3
                self.voice_type = "Baritone"
            elif self.voice_pitch < 262:  # Below C4
                self.voice_type = "Tenor"
            elif self.voice_pitch < 330:  # Below E4
                self.voice_type = "Alto"
            else:
                self.voice_type = "Soprano"
    
    def update(self, audio_data, spectrum=None):
        """Update voice detection analysis"""
        if self.is_frozen:
            return
        
        self.update_counter += 1
        if self.update_counter % self.update_interval != 0:
            return
        
        # Compute spectrum if not provided
        if spectrum is None and len(audio_data) > 0:
            window = np.hanning(len(audio_data))
            windowed = audio_data * window
            spectrum = np.abs(np.fft.rfft(windowed))
        
        if spectrum is not None and len(spectrum) > 0:
            # Detect voice
            self.detect_voice(audio_data, spectrum)
            
            # Update history
            self.voice_history.append(self.voice_confidence)
            
            # Track formants if voice is active
            if self.voice_active and len(self.formants) > 0:
                self.formant_history.append(self.formants[:])
            
            # Detect pitch if voice is active
            if self.voice_active and len(audio_data) > 0:
                self.detect_pitch(audio_data)
                self.analyze_voice_type()
    
    def detect_pitch(self, audio_data):
        """Detect fundamental frequency of voice"""
        # Simple autocorrelation-based pitch detection
        # Real implementation would use more robust methods
        
        if len(audio_data) < 2048:
            return
        
        # Autocorrelation
        corr = np.correlate(audio_data, audio_data, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero lag
        min_period = int(self.sample_rate / 400)  # Max 400 Hz
        max_period = int(self.sample_rate / 80)   # Min 80 Hz
        
        if max_period < len(corr):
            corr_slice = corr[min_period:max_period]
            if len(corr_slice) > 0:
                peak_idx = np.argmax(corr_slice) + min_period
                
                if corr[peak_idx] > 0.3 * corr[0]:  # Significant peak
                    self.voice_pitch = self.sample_rate / peak_idx
                    self.pitch_history.append(self.voice_pitch)
    
    def draw(self, screen, x, y, width, panel_color=None):
        """Draw the voice detection panel"""
        if not self.fonts:
            return
        
        # Panel background
        if panel_color is None:
            panel_color = self.bg_color
        
        panel_rect = pygame.Rect(x, y, width, self.panel_height)
        pygame.draw.rect(screen, panel_color, panel_rect)
        pygame.draw.rect(screen, (100, 100, 100), panel_rect, 1)
        
        # Title
        title = "Voice Detection"
        if self.is_frozen:
            title += " [FROZEN]"
        title_surface = self.fonts['small'].render(title, True, (255, 255, 255))
        screen.blit(title_surface, (x + 10, y + 5))
        
        # Voice activity indicator
        indicator_x = x + width - 100
        indicator_y = y + 5
        indicator_color = self.voice_color if self.voice_active else (50, 50, 50)
        pygame.draw.circle(screen, indicator_color, (indicator_x, indicator_y + 8), 6)
        
        if self.voice_active:
            conf_text = f"{self.voice_confidence:.0f}%"
            conf_surface = self.fonts['tiny'].render(conf_text, True, (200, 200, 200))
            screen.blit(conf_surface, (indicator_x + 10, indicator_y))
        
        # Voice type and pitch
        info_y = y + 30
        if self.voice_active and self.voice_pitch > 0:
            info_text = f"{self.voice_type} - {self.voice_pitch:.0f} Hz"
            info_surface = self.fonts['small'].render(info_text, True, (200, 200, 200))
            screen.blit(info_surface, (x + 10, info_y))
        
        # Confidence history graph
        graph_y = y + 55
        graph_height = 40
        graph_width = width - 20
        
        # Draw graph background
        graph_rect = pygame.Rect(x + 10, graph_y, graph_width, graph_height)
        pygame.draw.rect(screen, (30, 30, 30), graph_rect)
        pygame.draw.rect(screen, (60, 60, 60), graph_rect, 1)
        
        # Draw confidence history
        if len(self.voice_history) > 1:
            points = []
            for i, conf in enumerate(self.voice_history):
                px = x + 10 + (i * graph_width // len(self.voice_history))
                py = graph_y + graph_height - int(conf * graph_height / 100)
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(screen, self.voice_color, False, points, 2)
        
        # Formant display
        formant_y = y + 105
        if self.voice_active and len(self.formants) > 0:
            formant_text = "Formants:"
            text_surface = self.fonts['tiny'].render(formant_text, True, (150, 150, 150))
            screen.blit(text_surface, (x + 10, formant_y))
            
            # Draw formant frequencies
            formant_x = x + 80
            for i, (formant, color) in enumerate(zip(self.formants, self.formant_colors)):
                f_text = f"F{i+1}: {formant:.0f}Hz"
                f_surface = self.fonts['tiny'].render(f_text, True, color)
                screen.blit(f_surface, (formant_x + i * 100, formant_y))
        
        # Vocal quality indicators
        quality_y = y + 130
        if self.voice_active:
            # Simple quality visualization
            quality_width = int(self.vocal_quality * 200)
            quality_rect = pygame.Rect(x + 10, quality_y, quality_width, 10)
            pygame.draw.rect(screen, (100, 200, 100), quality_rect)
            pygame.draw.rect(screen, (60, 60, 60), 
                           pygame.Rect(x + 10, quality_y, 200, 10), 1)
            
            quality_text = "Vocal Clarity"
            text_surface = self.fonts['tiny'].render(quality_text, True, (150, 150, 150))
            screen.blit(text_surface, (x + 10, quality_y - 15))
    
    def get_status(self):
        """Get current voice detection status"""
        return {
            'active': self.voice_active,
            'confidence': self.voice_confidence,
            'pitch': self.voice_pitch,
            'voice_type': self.voice_type,
            'formants': self.formants.copy() if self.formants else []
        }