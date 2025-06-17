"""
Frequency Band Energy Tracker Panel for OMEGA-4 Audio Analyzer
Configurable frequency bands with RMS energy tracking and visual meters
Professional tool for monitoring specific frequency ranges
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class FrequencyBand:
    """Definition of a frequency band for energy tracking"""
    name: str
    low_freq: float
    high_freq: float
    color: Tuple[int, int, int]
    peak_hold: bool = True
    rms_window: int = 30  # Number of frames for RMS calculation


class FrequencyBandTracker:
    """Real-time frequency band energy tracking with RMS analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Default frequency bands (professional audio standard)
        self.frequency_bands = {
            'sub_bass': FrequencyBand('Sub Bass', 20, 60, (150, 50, 200)),
            'bass': FrequencyBand('Bass', 60, 250, (200, 100, 100)),
            'low_mid': FrequencyBand('Low Mid', 250, 500, (200, 150, 100)),
            'mid': FrequencyBand('Mid', 500, 2000, (150, 200, 150)),
            'high_mid': FrequencyBand('High Mid', 2000, 4000, (100, 150, 200)),
            'presence': FrequencyBand('Presence', 4000, 8000, (100, 200, 250)),
            'brilliance': FrequencyBand('Brilliance', 8000, 20000, (200, 200, 250))
        }
        
        # Energy tracking
        self.band_energies = {}
        self.band_rms = {}
        self.band_peaks = {}
        self.band_history = {}
        
        # Peak hold parameters
        self.peak_hold_duration = 90  # frames (1.5 seconds at 60fps)
        self.peak_decay_rate = 0.95   # Decay multiplier per frame
        
        # Initialize tracking data
        for band_id in self.frequency_bands:
            self.band_energies[band_id] = 0.0
            self.band_rms[band_id] = 0.0
            self.band_peaks[band_id] = {'value': 0.0, 'hold_time': 0}
            self.band_history[band_id] = deque(maxlen=100)  # History for visualization
        
        # Analysis parameters
        self.db_range = 60  # dB range for display
        self.reference_level = 0.0  # Reference level for dB calculation
        self.auto_reference = True  # Automatically adjust reference level
        
        # Statistics
        self.total_energy = 0.0
        self.spectral_centroid = 0.0
        self.spectral_spread = 0.0
        self.spectral_balance = {}  # Balance between bands
        
    def update(self, fft_data: np.ndarray, frequencies: np.ndarray):
        """Update frequency band energy tracking"""
        if fft_data is None or len(fft_data) == 0 or frequencies is None:
            return
        
        # Calculate total energy
        self.total_energy = np.sum(fft_data)
        
        # Calculate spectral centroid
        if self.total_energy > 0:
            self.spectral_centroid = np.sum(frequencies * fft_data) / self.total_energy
        
        # Update each frequency band
        for band_id, band in self.frequency_bands.items():
            energy = self._calculate_band_energy(fft_data, frequencies, band)
            
            # Update current energy
            self.band_energies[band_id] = energy
            
            # Add to history for RMS calculation
            self.band_history[band_id].append(energy)
            
            # Calculate RMS over window
            if len(self.band_history[band_id]) >= band.rms_window:
                recent_energies = list(self.band_history[band_id])[-band.rms_window:]
                self.band_rms[band_id] = np.sqrt(np.mean(np.array(recent_energies) ** 2))
            else:
                self.band_rms[band_id] = energy
            
            # Update peak hold
            if band.peak_hold:
                self._update_peak_hold(band_id, energy)
        
        # Update spectral spread
        self._calculate_spectral_spread(fft_data, frequencies)
        
        # Calculate spectral balance
        self._calculate_spectral_balance()
        
        # Auto-adjust reference level if enabled
        if self.auto_reference:
            self._update_reference_level()
    
    def _calculate_band_energy(self, fft_data: np.ndarray, frequencies: np.ndarray, 
                              band: FrequencyBand) -> float:
        """Calculate energy in a specific frequency band"""
        # Find frequency indices
        low_idx = np.argmax(frequencies >= band.low_freq)
        high_idx = np.argmax(frequencies >= band.high_freq)
        
        if high_idx == 0:  # If high_freq is beyond spectrum
            high_idx = len(fft_data)
        
        # Ensure valid range
        if low_idx >= high_idx or high_idx > len(fft_data):
            return 0.0
        
        # Calculate band energy
        band_spectrum = fft_data[low_idx:high_idx]
        energy = np.sum(band_spectrum)
        
        return energy
    
    def _update_peak_hold(self, band_id: str, current_energy: float):
        """Update peak hold for a frequency band"""
        peak_info = self.band_peaks[band_id]
        
        if current_energy > peak_info['value']:
            # New peak
            peak_info['value'] = current_energy
            peak_info['hold_time'] = self.peak_hold_duration
        else:
            # Decay peak hold
            if peak_info['hold_time'] > 0:
                peak_info['hold_time'] -= 1
            else:
                # Apply decay
                peak_info['value'] *= self.peak_decay_rate
                # Ensure peak doesn't go below current energy
                peak_info['value'] = max(peak_info['value'], current_energy)
    
    def _calculate_spectral_spread(self, fft_data: np.ndarray, frequencies: np.ndarray):
        """Calculate spectral spread (measure of frequency distribution width)"""
        if self.total_energy > 0 and self.spectral_centroid > 0:
            weighted_deviations = ((frequencies - self.spectral_centroid) ** 2) * fft_data
            self.spectral_spread = np.sqrt(np.sum(weighted_deviations) / self.total_energy)
        else:
            self.spectral_spread = 0.0
    
    def _calculate_spectral_balance(self):
        """Calculate balance between different frequency regions"""
        total_rms = sum(self.band_rms.values())
        
        if total_rms > 0:
            for band_id in self.frequency_bands:
                self.spectral_balance[band_id] = self.band_rms[band_id] / total_rms
        else:
            for band_id in self.frequency_bands:
                self.spectral_balance[band_id] = 0.0
    
    def _update_reference_level(self):
        """Update reference level for dB calculation"""
        # Use 95th percentile of recent total energy as reference
        if hasattr(self, 'total_energy_history'):
            if len(self.total_energy_history) >= 50:
                self.reference_level = np.percentile(list(self.total_energy_history), 95)
        else:
            self.total_energy_history = deque(maxlen=100)
        
        self.total_energy_history.append(self.total_energy)
    
    def get_band_db(self, band_id: str, use_rms: bool = True) -> float:
        """Get band energy in dB relative to reference level"""
        if use_rms:
            energy = self.band_rms.get(band_id, 0.0)
        else:
            energy = self.band_energies.get(band_id, 0.0)
        
        if energy > 0 and self.reference_level > 0:
            return 20 * np.log10(energy / self.reference_level)
        else:
            return -60.0  # Minimum dB
    
    def get_band_normalized(self, band_id: str, use_rms: bool = True) -> float:
        """Get band energy normalized to 0-1 range"""
        db_value = self.get_band_db(band_id, use_rms)
        # Map dB range to 0-1
        normalized = (db_value + self.db_range) / self.db_range
        return max(0.0, min(1.0, normalized))
    
    def add_custom_band(self, band_id: str, name: str, low_freq: float, 
                       high_freq: float, color: Tuple[int, int, int]):
        """Add a custom frequency band"""
        self.frequency_bands[band_id] = FrequencyBand(name, low_freq, high_freq, color)
        self.band_energies[band_id] = 0.0
        self.band_rms[band_id] = 0.0
        self.band_peaks[band_id] = {'value': 0.0, 'hold_time': 0}
        self.band_history[band_id] = deque(maxlen=100)
    
    def remove_band(self, band_id: str):
        """Remove a frequency band"""
        if band_id in self.frequency_bands:
            del self.frequency_bands[band_id]
            del self.band_energies[band_id]
            del self.band_rms[band_id]
            del self.band_peaks[band_id]
            del self.band_history[band_id]
    
    def get_dominant_band(self) -> Tuple[str, float]:
        """Get the frequency band with highest RMS energy"""
        if not self.band_rms:
            return 'none', 0.0
        
        dominant_band = max(self.band_rms.items(), key=lambda x: x[1])
        return dominant_band
    
    def get_spectral_tilt(self) -> float:
        """Calculate spectral tilt (high freq vs low freq energy)"""
        # Compare high frequency energy to low frequency energy
        high_freq_bands = ['high_mid', 'presence', 'brilliance']
        low_freq_bands = ['sub_bass', 'bass', 'low_mid']
        
        high_energy = sum(self.band_rms.get(band, 0.0) for band in high_freq_bands)
        low_energy = sum(self.band_rms.get(band, 0.0) for band in low_freq_bands)
        
        if low_energy > 0:
            return high_energy / low_energy
        else:
            return 0.0


class FrequencyBandTrackerPanel:
    """OMEGA-4 Frequency Band Energy Tracker Panel"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.tracker = FrequencyBandTracker(sample_rate)
        
        # Display state
        self.band_info = {
            'bands': {},
            'dominant_band': 'none',
            'spectral_centroid': 0.0,
            'spectral_spread': 0.0,
            'spectral_tilt': 0.0,
            'total_energy_db': -60.0
        }
        
        # Display options
        self.show_peak_holds = True
        self.show_rms_values = True
        self.show_balance_chart = True
        self.meter_style = 'vertical'  # 'vertical' or 'horizontal'
        
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
    
    def update(self, fft_data: np.ndarray, frequencies: np.ndarray):
        """Update frequency band tracking"""
        if fft_data is not None and len(fft_data) > 0:
            self.tracker.update(fft_data, frequencies)
            
            # Update display info
            self.band_info = {
                'bands': {
                    band_id: {
                        'name': band.name,
                        'energy': self.tracker.band_energies[band_id],
                        'rms': self.tracker.band_rms[band_id],
                        'peak': self.tracker.band_peaks[band_id]['value'],
                        'db_value': self.tracker.get_band_db(band_id),
                        'normalized': self.tracker.get_band_normalized(band_id),
                        'balance': self.tracker.spectral_balance.get(band_id, 0.0),
                        'color': band.color
                    }
                    for band_id, band in self.tracker.frequency_bands.items()
                },
                'dominant_band': self.tracker.get_dominant_band()[0],
                'spectral_centroid': self.tracker.spectral_centroid,
                'spectral_spread': self.tracker.spectral_spread,
                'spectral_tilt': self.tracker.get_spectral_tilt(),
                'total_energy_db': 20 * np.log10(max(self.tracker.total_energy, 1e-10))
            }
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw frequency band tracker panel"""
        # Background
        pygame.draw.rect(screen, (15, 15, 20), (x, y, width, height))
        pygame.draw.rect(screen, (80, 80, 100), (x, y, width, height), 2)
        
        y_offset = y + int(10 * ui_scale)
        
        # Title
        if self.font_medium:
            title_text = self.font_medium.render("Frequency Band Energy Tracker", True, (200, 200, 220))
            screen.blit(title_text, (x + int(10 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Draw frequency band meters
        if self.meter_style == 'vertical':
            self._draw_vertical_meters(screen, x + int(10 * ui_scale), y_offset, 
                                     width - int(20 * ui_scale), height - y_offset + y - int(60 * ui_scale), ui_scale)
        else:
            self._draw_horizontal_meters(screen, x + int(10 * ui_scale), y_offset, 
                                       width - int(20 * ui_scale), height - y_offset + y - int(60 * ui_scale), ui_scale)
        
        # Draw spectral analysis info
        self._draw_spectral_info(screen, x + int(10 * ui_scale), 
                               y + height - int(50 * ui_scale), 
                               width - int(20 * ui_scale), int(40 * ui_scale), ui_scale)
    
    def _draw_vertical_meters(self, screen: pygame.Surface, x: int, y: int, 
                            width: int, height: int, ui_scale: float):
        """Draw vertical frequency band meters"""
        bands = self.band_info.get('bands', {})
        if not bands:
            return
        
        num_bands = len(bands)
        meter_width = (width - (num_bands - 1) * int(5 * ui_scale)) // num_bands
        meter_height = height - int(40 * ui_scale)  # Leave space for labels
        
        for i, (band_id, band_data) in enumerate(bands.items()):
            meter_x = x + i * (meter_width + int(5 * ui_scale))
            meter_y = y
            
            # Meter background
            pygame.draw.rect(screen, (30, 30, 40), 
                           (meter_x, meter_y, meter_width, meter_height))
            pygame.draw.rect(screen, (60, 60, 80), 
                           (meter_x, meter_y, meter_width, meter_height), 1)
            
            # RMS level bar
            rms_normalized = band_data.get('normalized', 0.0)
            rms_height = int(meter_height * rms_normalized)
            
            if rms_height > 0:
                pygame.draw.rect(screen, band_data['color'], 
                               (meter_x + 2, meter_y + meter_height - rms_height, 
                                meter_width - 4, rms_height))
            
            # Peak hold line
            if self.show_peak_holds:
                peak_normalized = min(1.0, band_data.get('peak', 0.0) / max(band_data.get('rms', 1.0), 1e-10))
                peak_y = meter_y + meter_height - int(meter_height * peak_normalized)
                
                pygame.draw.line(screen, (255, 255, 100), 
                               (meter_x, peak_y), (meter_x + meter_width, peak_y), 2)
            
            # Band label
            if self.font_tiny:
                label_text = band_data['name'][:4]  # Abbreviated
                label_surf = self.font_tiny.render(label_text, True, (180, 180, 200))
                label_rect = label_surf.get_rect(centerx=meter_x + meter_width // 2, 
                                               y=meter_y + meter_height + 5)
                screen.blit(label_surf, label_rect)
            
            # dB value
            if self.show_rms_values and self.font_tiny:
                db_value = band_data.get('db_value', -60.0)
                db_text = f"{db_value:.0f}"
                db_surf = self.font_tiny.render(db_text, True, (150, 150, 170))
                db_rect = db_surf.get_rect(centerx=meter_x + meter_width // 2, 
                                         y=meter_y + meter_height + 20)
                screen.blit(db_surf, db_rect)
    
    def _draw_horizontal_meters(self, screen: pygame.Surface, x: int, y: int, 
                              width: int, height: int, ui_scale: float):
        """Draw horizontal frequency band meters"""
        bands = self.band_info.get('bands', {})
        if not bands:
            return
        
        num_bands = len(bands)
        meter_height = (height - (num_bands - 1) * int(5 * ui_scale)) // num_bands
        meter_width = width - int(80 * ui_scale)  # Leave space for labels
        
        for i, (band_id, band_data) in enumerate(bands.items()):
            meter_x = x + int(80 * ui_scale)
            meter_y = y + i * (meter_height + int(5 * ui_scale))
            
            # Band label
            if self.font_small:
                label_surf = self.font_small.render(band_data['name'], True, (180, 180, 200))
                screen.blit(label_surf, (x, meter_y + meter_height // 2 - 8))
            
            # Meter background
            pygame.draw.rect(screen, (30, 30, 40), 
                           (meter_x, meter_y, meter_width, meter_height))
            pygame.draw.rect(screen, (60, 60, 80), 
                           (meter_x, meter_y, meter_width, meter_height), 1)
            
            # RMS level bar
            rms_normalized = band_data.get('normalized', 0.0)
            rms_width = int(meter_width * rms_normalized)
            
            if rms_width > 0:
                pygame.draw.rect(screen, band_data['color'], 
                               (meter_x + 2, meter_y + 2, rms_width - 4, meter_height - 4))
            
            # Peak hold line
            if self.show_peak_holds:
                peak_normalized = min(1.0, band_data.get('peak', 0.0) / max(band_data.get('rms', 1.0), 1e-10))
                peak_x = meter_x + int(meter_width * peak_normalized)
                
                pygame.draw.line(screen, (255, 255, 100), 
                               (peak_x, meter_y), (peak_x, meter_y + meter_height), 2)
            
            # dB value
            if self.show_rms_values and self.font_tiny:
                db_value = band_data.get('db_value', -60.0)
                db_text = f"{db_value:.0f}dB"
                db_surf = self.font_tiny.render(db_text, True, (150, 150, 170))
                screen.blit(db_surf, (meter_x + meter_width + 5, meter_y + meter_height // 2 - 6))
    
    def _draw_spectral_info(self, screen: pygame.Surface, x: int, y: int, 
                          width: int, height: int, ui_scale: float):
        """Draw spectral analysis information"""
        if not self.font_tiny:
            return
        
        # Background
        pygame.draw.rect(screen, (20, 20, 30), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 1)
        
        # Spectral info
        centroid = self.band_info.get('spectral_centroid', 0.0)
        spread = self.band_info.get('spectral_spread', 0.0)
        tilt = self.band_info.get('spectral_tilt', 0.0)
        dominant = self.band_info.get('dominant_band', 'none')
        
        info_text = f"Centroid: {centroid:.0f}Hz | Spread: {spread:.0f}Hz | Tilt: {tilt:.2f} | Dominant: {dominant}"
        info_surf = self.font_tiny.render(info_text, True, (180, 180, 200))
        screen.blit(info_surf, (x + 5, y + 5))
        
        # Total energy
        total_db = self.band_info.get('total_energy_db', -60.0)
        total_text = f"Total Energy: {total_db:.1f}dB"
        total_surf = self.font_tiny.render(total_text, True, (200, 180, 180))
        screen.blit(total_surf, (x + 5, y + 20))
    
    def toggle_peak_holds(self):
        """Toggle peak hold display"""
        self.show_peak_holds = not self.show_peak_holds
    
    def toggle_meter_style(self):
        """Toggle between vertical and horizontal meter styles"""
        self.meter_style = 'horizontal' if self.meter_style == 'vertical' else 'vertical'
    
    def reset_peaks(self):
        """Reset all peak holds"""
        for band_id in self.tracker.band_peaks:
            self.tracker.band_peaks[band_id] = {'value': 0.0, 'hold_time': 0}
    
    def get_results(self) -> Dict[str, Any]:
        """Get current frequency band analysis results"""
        return self.band_info.copy()