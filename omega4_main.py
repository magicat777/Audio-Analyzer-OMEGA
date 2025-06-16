#!/usr/bin/env python3
"""
Live Audio Analyzer v5 - Professional Studio-Grade Analysis
Advanced real-time audio analysis with enhanced low-end detail and professional features

Version 5 Features:
- Multi-resolution FFT for enhanced bass detail
- Professional metering (LUFS, K-weighting, True Peak)
- Advanced harmonic analysis and instrument identification
- Phase coherence analysis for stereo imaging
- Transient analysis and room mode detection
- Psychoacoustic bass enhancement
- Studio-grade visualization tools
"""

import numpy as np
import pygame
import sys
import threading
import queue
from queue import Queue, Empty
import time
import os
import subprocess
from collections import deque
from typing import Dict, List, Tuple, Any
import argparse
from scipy import signal as scipy_signal
from math import inf

# Import configuration
from omega4.config import (
    SAMPLE_RATE, CHUNK_SIZE, BARS_DEFAULT, BARS_MAX, 
    MAX_FREQ, FFT_SIZE_BASE, DEFAULT_WIDTH, DEFAULT_HEIGHT,
    TARGET_FPS, BACKGROUND_COLOR, GRID_COLOR, TEXT_COLOR,
    get_config
)

# Import display interface
from omega4.visualization.display_interface import SpectrumDisplay

# Import panels
from omega4.panels.professional_meters import ProfessionalMetersPanel
from omega4.panels.vu_meters import VUMetersPanel
from omega4.panels.bass_zoom import BassZoomPanel
from omega4.panels.harmonic_analysis import HarmonicAnalysisPanel
from omega4.panels.pitch_detection import PitchDetectionPanel
from omega4.panels.pitch_detection_config import get_preset_config
from omega4.panels.chromagram import ChromagramPanel
from omega4.panels.genre_classification import GenreClassificationPanel
from omega4.panels.integrated_music_panel import IntegratedMusicPanel

# Import optimization modules
from omega4.optimization.adaptive_updater import AdaptiveUpdater
from omega4.optimization import get_cached_fft, cache_fft_result, PrecomputedFrequencyMapper

# Import analyzers
from omega4.analyzers import (
    PhaseCoherenceAnalyzer,
    TransientAnalyzer,
    RoomModeAnalyzer,
    EnhancedDrumDetector
)

# Import audio modules
from omega4.audio import (
    PipeWireMonitorCapture,
    AudioCaptureManager,
    MultiResolutionFFT,
    AudioProcessingPipeline,
    ContentTypeDetector,
    VoiceDetectionWrapper
)
from omega4.audio.audio_config import PipelineConfig
from omega4.audio.capture import AudioCaptureConfig


class ProfessionalLiveAudioAnalyzer:
    """Professional Live Audio Analyzer v4.1 OMEGA with enhanced low-end detail and studio features"""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, bars=BARS_DEFAULT, source_name=None):
        # Ensure minimum dimensions
        min_width = 1280
        min_height = 1150  # Absolute minimum for header + spectrum + footer
        
        self.width = max(width, min_width)
        self.height = max(height, min_height)
        self.bars = bars
        
        if width < min_width or height < min_height:
            print(f"Warning: Window size adjusted to minimum {self.width}x{self.height} (requested: {width}x{height})")

        # Initialize core components
        capture_config = AudioCaptureConfig(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
            prefer_focusrite=True,
            enable_stats=True
        )
        self.capture = PipeWireMonitorCapture(source_name, capture_config)
        self.drum_detector = EnhancedDrumDetector(SAMPLE_RATE)
        self.voice_detector = VoiceDetectionWrapper(SAMPLE_RATE)
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(sample_rate=SAMPLE_RATE, num_bands=self.bars)
        self.content_detector = ContentTypeDetector(pipeline_config)
        self.audio_pipeline = AudioProcessingPipeline(pipeline_config)

        # Initialize new v5 components
        self.multi_fft = MultiResolutionFFT(SAMPLE_RATE)
        self.professional_meters_panel = ProfessionalMetersPanel(SAMPLE_RATE)
        self.vu_meters_panel = VUMetersPanel(SAMPLE_RATE)
        self.bass_zoom_panel = BassZoomPanel(SAMPLE_RATE)
        self.harmonic_analysis_panel = HarmonicAnalysisPanel(SAMPLE_RATE)
        # Use balanced configuration for pitch detection (works for both voice and music)
        try:
            pitch_config = get_preset_config('balanced')
            self.pitch_detection_panel = PitchDetectionPanel(SAMPLE_RATE, pitch_config)
            print("Pitch detection panel initialized successfully with 'balanced' config")
        except Exception as e:
            print(f"Error initializing pitch detection panel: {e}")
            # Fallback to default config
            self.pitch_detection_panel = PitchDetectionPanel(SAMPLE_RATE)
            print("Using default pitch detection configuration")
        self.chromagram_panel = ChromagramPanel(SAMPLE_RATE)
        self.genre_classification_panel = GenreClassificationPanel(SAMPLE_RATE)
        self.integrated_music_panel = IntegratedMusicPanel(SAMPLE_RATE)
        self.phase_analyzer = PhaseCoherenceAnalyzer(SAMPLE_RATE)
        self.transient_analyzer = TransientAnalyzer(SAMPLE_RATE)
        self.room_analyzer = RoomModeAnalyzer(SAMPLE_RATE)
        
        # Input gain control for better signal levels
        self.input_gain = 4.0  # Default 12dB boost for typical music sources
        self.auto_gain_enabled = True
        self.gain_history = deque(maxlen=300)  # 5 seconds at 60 FPS
        self.target_lufs = -16.0  # Target for good visualization
        
        # Compensation toggles for debugging
        self.freq_compensation_enabled = True  # Toggle with 'Q'
        self.psychoacoustic_enabled = True     # Toggle with 'W'
        self.normalization_enabled = False     # Toggle with 'E' (disabled by default)
        self.smoothing_enabled = True          # Toggle with 'R'

        # Audio analysis with enhanced resolution
        self.ring_buffer = np.zeros(FFT_SIZE_BASE * 4, dtype=np.float32)  # Larger buffer
        self.buffer_pos = 0

        # Enhanced frequency mapping with 768 bars
        self.freqs = np.fft.rfftfreq(FFT_SIZE_BASE, 1 / SAMPLE_RATE)
        
        # Pre-computed frequency mappings (initialize early)
        self.freq_mapper = PrecomputedFrequencyMapper(SAMPLE_RATE, FFT_SIZE_BASE, self.bars)
        
        # Adaptive content detection
        self.adaptive_allocation_enabled = False  # Disabled for perceptual mapping
        self.current_content_type = 'instrumental'
        self.current_allocation = 0.75
        # Use pre-computed band indices from frequency mapper
        self.band_indices = self.freq_mapper.mapping.band_indices

        # Pygame setup with professional UI
        pygame.init()
        
        # Since all panels are visible by default, ensure window is tall enough
        # This is a temporary fix until we calculate required height properly
        if height < 1900:
            print(f"Window height {height} is too small for all panels. Adjusting to 1900px.")
            height = 1900
            self.height = height
            
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Professional Audio Analyzer v4.1 OMEGA-4")
        self.clock = pygame.time.Clock()
        
        # Initialize display interface
        self.display = SpectrumDisplay(self.screen, width, height, self.bars)

        # Dynamic font sizing system
        self.base_width = 2000  # Reference width for font scaling
        self.update_fonts(width)
        
        # Transient detection
        self.transient_events = deque(maxlen=100)
        self.last_transient_time = 0
        self.last_rms = 0.0
        
        # Pass fonts to display interface
        self.display.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny,
            'grid': self.font_grid,
            'mono': self.font_tiny  # Use tiny font as mono for now
        })
        
        # Pass fonts to panels
        self.professional_meters_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny,
            'meters': self.font_meters
        })
        
        self.vu_meters_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.bass_zoom_panel.set_fonts({
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.harmonic_analysis_panel.set_fonts({
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.pitch_detection_panel.set_fonts({
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.chromagram_panel.set_fonts({
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.genre_classification_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })
        
        self.integrated_music_panel.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        })

        # Professional UI panels - All visible by default
        self.show_meters = True
        self.show_vu_meters = True  # VU meters shown by default
        self.show_bass_zoom = True
        self.show_harmonic = True
        self.show_room_analysis = True
        self.show_pitch_detection = True
        self.show_chromagram = False  # Disabled - using integrated panel instead
        self.show_genre_classification = False  # Disabled - using integrated panel instead
        self.show_integrated_music = True  # Integrated music analysis panel (preferred)
        self.show_debug = False  # Debug snapshots triggered by 'D' key
        self.show_voice_detection = False
        self.show_formant = False
        self.show_advanced_voice = False
        self.show_fps = True
        self.show_stats = True
        self.show_phase = False
        self.show_transient = False
        self.show_grid = True
        
        # Frozen/paused states for panels (when True, panel doesn't update)
        self.frozen_meters = False
        self.frozen_vu_meters = False
        self.frozen_bass_zoom = False
        self.frozen_harmonic = False
        self.frozen_room_analysis = False
        self.frozen_pitch_detection = False
        self.frozen_chromagram = False  # Not used - integrated panel preferred
        self.frozen_genre_classification = False  # Not used - integrated panel preferred
        self.frozen_integrated_music = False
        self.fullscreen = False
        self.test_mode = False  # Auto-test all panels

        # Bass detail enhancement
        self.bass_detail_factor = 1.2  # Additional bass resolution
        self.bass_smoothing = 0.95  # Higher = smoother
        
        # Psychoacoustic processing
        self.equal_loudness_curve = self._create_equal_loudness_curve()
        self.psycho_bass_boost = 1.5  # Extra boost for perceived bass
        
        # Content-adaptive filtering
        self.vocal_suppression = 0.0  # 0-1, reduces 1-4kHz for instrumental focus
        
        # Multi-resolution spectrum history
        self.spectrum_history_multi = []
        self.max_spectrum_history = 100  # Frames to keep
        
        # Harmonic tracking
        self.harmonic_peaks = []
        self.fundamental_freq = 0
        self.harmonics = []
        
        # Phase analysis data
        self.phase_data = np.zeros(FFT_SIZE_BASE // 2 + 1)
        self.phase_history = deque(maxlen=30)
        
        # Transient tracking
        self.transient_events = deque(maxlen=50)
        self.last_transient_time = 0
        
        # Room mode detection
        self.room_modes = []
        self.mode_intensities = {}
        
        # Bass enhancement history
        self.bass_history = deque(maxlen=120)  # 2 seconds at 60 FPS
        self.bass_envelope = 0
        
        # Performance optimization
        self.adaptive_updater = AdaptiveUpdater()
        self.last_fps_update = time.time()
        self.fps_history = deque(maxlen=60)
        
        # Voice detection state
        self.voice_active = False
        self.voice_confidence = 0.0
        self.formant_freqs = []
        
        # A/B toggle for compensation comparison
        self.compensation_ab_mode = False  # False = A (with compensation), True = B (raw)
        
        # Screenshot functionality
        self.screenshot_counter = 0
        
        # Performance monitoring
        self.frame_times = deque(maxlen=120)
        self.process_times = deque(maxlen=120)
        
        # Window sizing control
        self.preset_locked = False  # Lock window size when using number key presets
        
        # Initialize running state
        self.running = True

    def update_fonts(self, width):
        """Update font sizes based on window width"""
        scale = width / self.base_width
        
        # Create fonts with dynamic sizing
        self.font_large = pygame.font.Font(None, int(48 * scale))
        self.font_medium = pygame.font.Font(None, int(36 * scale))
        self.font_small = pygame.font.Font(None, int(24 * scale))
        self.font_tiny = pygame.font.Font(None, int(18 * scale))
        self.font_grid = pygame.font.Font(None, int(16 * scale))
        self.font_meters = pygame.font.Font(None, int(32 * scale))  # For professional meters

    def _create_enhanced_band_mapping(self):
        """Create frequency band indices with musical perceptual mapping"""
        # Use Bark-scale inspired mapping for better musical representation
        bands = []
        n_bars = self.bars
        
        # Define frequency bands with perceptual/musical distribution
        # For 1024 bars, optimize for musical content and human hearing
        # Human hearing is most sensitive 1-5kHz, music fundamentals are 80-1kHz
        # 12% bass (20-250Hz) - important for rhythm, but limited by FFT resolution
        # 18% low-mid (250-500Hz) - warmth and body
        # 30% mid (500-2kHz) - most musical fundamentals
        # 25% high-mid (2k-6kHz) - presence and clarity (peak hearing sensitivity)
        # 15% high (6k-20kHz) - air and brilliance
        
        bass_bars = int(n_bars * 0.12)     # ~123 bars
        lowmid_bars = int(n_bars * 0.18)   # ~184 bars
        mid_bars = int(n_bars * 0.30)      # ~307 bars
        highmid_bars = int(n_bars * 0.25)  # ~256 bars
        high_bars = n_bars - (bass_bars + lowmid_bars + mid_bars + highmid_bars)
        
        # Create frequency points using a modified Mel-scale approach
        # This better matches human perception and provides good separation
        freq_points = []
        
        # Helper function to convert Hz to Mel scale
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Create points in Mel scale, then convert back to Hz
        # This gives more resolution where human hearing is most sensitive
        mel_min = hz_to_mel(20)
        mel_max = hz_to_mel(20000)
        
        # Distribute points evenly in Mel scale
        mel_points = np.linspace(mel_min, mel_max, n_bars + 1)
        freq_points = [mel_to_hz(mel) for mel in mel_points]
        
        # Ensure we stay within bounds
        freq_points[0] = max(20, freq_points[0])
        freq_points[-1] = min(20000, freq_points[-1])
        
        # Map to FFT bins based on actual FFT size
        fft_size = FFT_SIZE_BASE
        freq_bin_width = SAMPLE_RATE / fft_size
        
        # Ensure we have the correct number of frequency points
        if len(freq_points) != n_bars + 1:
            print(f"[WARNING] freq_points length {len(freq_points)} != n_bars+1 {n_bars+1}")
            # Regenerate with correct count
            freq_points = np.logspace(np.log10(20), np.log10(20000), n_bars + 1)
        
        for i in range(n_bars):
            if i >= len(freq_points) - 1:
                break
                
            start_freq = freq_points[i]
            end_freq = freq_points[i + 1]
            
            start_idx = int(start_freq / freq_bin_width)
            end_idx = int(end_freq / freq_bin_width)
            
            # Ensure at least one bin per band
            if end_idx <= start_idx:
                end_idx = start_idx + 1
            
            # Clamp to valid FFT range
            start_idx = max(0, min(start_idx, fft_size // 2))
            end_idx = max(start_idx + 1, min(end_idx, fft_size // 2 + 1))
            
            bands.append((start_idx, end_idx))
        
        # Debug frequency distribution
        if not hasattr(self, '_bands_logged'):
            self._bands_logged = True
            print(f"[BANDS] Total bars: {len(bands)}, FFT size: {fft_size}, Bin width: {freq_bin_width:.1f}Hz")
            print(f"[BANDS] First 10 frequency bands:")
            for i in range(min(10, len(bands))):
                start_idx, end_idx = bands[i]
                start_freq = start_idx * freq_bin_width
                end_freq = end_idx * freq_bin_width
                print(f"  Band {i}: {start_freq:.1f}-{end_freq:.1f} Hz (bins {start_idx}-{end_idx})")
            
            # Also show some high frequency bands
            if len(bands) > 100:
                print(f"[BANDS] High frequency bands:")
                for i in [len(bands)//2, len(bands)*3//4, len(bands)-10, len(bands)-1]:
                    if i < len(bands):
                        start_idx, end_idx = bands[i]
                        start_freq = start_idx * freq_bin_width
                        end_freq = end_idx * freq_bin_width
                        print(f"  Band {i}: {start_freq:.1f}-{end_freq:.1f} Hz")
        
        return bands

    def _create_equal_loudness_curve(self):
        """Create equal loudness compensation curve (ISO 226:2003 80 phon approximation)"""
        freqs = self.freqs[:FFT_SIZE_BASE // 2 + 1]
        
        # Simplified equal loudness contour
        curve = np.ones_like(freqs)
        
        # Boost bass frequencies (enhanced for v5)
        bass_mask = freqs < 200
        curve[bass_mask] = 1 + (200 - freqs[bass_mask]) / 50
        
        # Slight mid reduction
        mid_mask = (freqs > 500) & (freqs < 2000)
        curve[mid_mask] *= 0.85
        
        # Presence boost (reduced)
        presence_mask = (freqs > 2000) & (freqs < 5000)
        curve[presence_mask] *= 1.05
        
        # High frequency rolloff (more aggressive)
        high_mask = freqs > 6000
        curve[high_mask] *= 0.5
        
        # Ultra-high frequency strong rolloff
        ultra_mask = freqs > 10000
        curve[ultra_mask] *= 0.2
        
        return curve

    def capture_audio(self):
        """Audio capture thread with enhanced processing"""
        print(f"Starting audio capture thread...")
        debug_counter = 0
        
        while self.running:
            try:
                # Get audio data with enhanced gain
                audio_data = self.capture.get_audio_data()
                
                if audio_data is not None and len(audio_data) > 0:
                    # Only debug if in debug mode
                    if self.show_debug and debug_counter % 60 == 0:
                        rms = np.sqrt(np.mean(audio_data**2))
                        # This will be part of the comprehensive debug output
                    debug_counter += 1
                    # Convert to float32 and apply input gain
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Apply gain
                    audio_data = audio_data * self.input_gain
                    
                    # Update ring buffer
                    chunk_size = len(audio_data)
                    if self.buffer_pos + chunk_size <= len(self.ring_buffer):
                        self.ring_buffer[self.buffer_pos:self.buffer_pos + chunk_size] = audio_data
                    else:
                        # Wrap around
                        first_part = len(self.ring_buffer) - self.buffer_pos
                        self.ring_buffer[self.buffer_pos:] = audio_data[:first_part]
                        self.ring_buffer[:chunk_size - first_part] = audio_data[first_part:]
                    
                    self.buffer_pos = (self.buffer_pos + chunk_size) % len(self.ring_buffer)
                    
                else:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Audio capture error: {e}")
                time.sleep(0.1)

    def process_multi_resolution_fft(self, audio_data):
        """Process audio with multiple FFT resolutions using new module"""
        try:
            # Check FFT cache first
            cached_result = get_cached_fft(audio_data)
            if cached_result is not None:
                # Extract spectrum from cached result
                if 'spectrum' in cached_result:
                    return cached_result['spectrum']
                    
            # Use the improved MultiResolutionFFT module
            fft_results = self.multi_fft.process_audio_chunk(
                audio_data, 
                apply_weighting=self.psychoacoustic_enabled
            )
            
            # Combine results into single spectrum
            if fft_results:
                spectrum, frequencies = self.multi_fft.combine_results_optimized(
                    fft_results, 
                    target_bins=self.bars
                )
            else:
                # Fallback to simple FFT if multi-resolution fails
                fft_result = np.fft.rfft(audio_data)
                spectrum = np.abs(fft_result)
                
                # Resample to match requested number of bars
                if len(spectrum) > self.bars:
                    indices = np.linspace(0, len(spectrum) - 1, self.bars).astype(int)
                    spectrum = spectrum[indices]
                elif len(spectrum) < self.bars:
                    spectrum = np.interp(np.linspace(0, len(spectrum) - 1, self.bars),
                                       np.arange(len(spectrum)), spectrum)
            
            # Detect transients
            rms = np.sqrt(np.mean(audio_data**2))
            if hasattr(self, 'last_rms'):
                if rms > self.last_rms * 2.0 and time.time() - self.last_transient_time > 0.1:
                    self.transient_events.append({
                        'type': 'onset',
                        'time': time.time(),
                        'magnitude': rms
                    })
                    self.last_transient_time = time.time()
            self.last_rms = rms
            
            # Apply additional psychoacoustic compensation if enabled
            if self.psychoacoustic_enabled and hasattr(self, 'equal_loudness_curve'):
                spectrum *= self.equal_loudness_curve[:len(spectrum)]
                if hasattr(self, 'psycho_bass_boost'):
                    bass_mask = frequencies < 250 if 'frequencies' in locals() else np.arange(len(spectrum)) < len(spectrum) // 8
                    spectrum[bass_mask] *= self.psycho_bass_boost
            
            # Cache the result for future use
            cache_fft_result(audio_data, {'spectrum': spectrum})
            
            return spectrum
            
        except Exception as e:
            print(f"Multi-resolution FFT failed: {e}")
            # Fallback to simple FFT
            fft_result = np.fft.rfft(audio_data)
            spectrum = np.abs(fft_result)
            
            if len(spectrum) > self.bars:
                indices = np.linspace(0, len(spectrum) - 1, self.bars).astype(int)
                spectrum = spectrum[indices]
            elif len(spectrum) < self.bars:
                spectrum = np.interp(np.linspace(0, len(spectrum) - 1, self.bars),
                                   np.arange(len(spectrum)), spectrum)
            
            return spectrum

    def auto_adjust_gain(self, audio_data):
        """Automatically adjust input gain based on signal level"""
        if not self.auto_gain_enabled:
            return
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms > 0:
            # Convert to dB
            current_db = 20 * np.log10(rms)
            
            # Target is around -20dB for good headroom
            target_db = -20
            adjustment = target_db - current_db
            
            # Smooth adjustment
            new_gain = self.input_gain * (10 ** (adjustment / 20))
            new_gain = np.clip(new_gain, 0.1, 10.0)  # Limit gain range
            
            # Apply smoothing
            self.input_gain = self.input_gain * 0.95 + new_gain * 0.05
            
            self.gain_history.append(self.input_gain)

    def update_content_type(self, spectrum, audio_data):
        """Update detected content type using spectral analysis"""
        # Analyze frequency distribution to detect content type
        if len(spectrum) > 0:
            freq_bin_width = SAMPLE_RATE / (2 * len(spectrum))
            
            # Define frequency ranges
            bass_end = int(250 / freq_bin_width)
            vocal_start = int(200 / freq_bin_width)
            vocal_end = int(4000 / freq_bin_width)
            high_start = int(6000 / freq_bin_width)
            
            # Calculate energy in each range
            bass_energy = np.mean(spectrum[:bass_end]) if bass_end < len(spectrum) else 0
            vocal_energy = np.mean(spectrum[vocal_start:vocal_end]) if vocal_end < len(spectrum) else 0
            high_energy = np.mean(spectrum[high_start:]) if high_start < len(spectrum) else 0
            total_energy = np.mean(spectrum)
            
            # Detect content type based on energy distribution
            if total_energy > 0:
                bass_ratio = bass_energy / total_energy
                vocal_ratio = vocal_energy / total_energy
                
                # Use voice detection result if available
                if self.voice_active and self.voice_confidence > 30:
                    self.current_content_type = 'vocal'
                elif bass_ratio > 0.6:
                    self.current_content_type = 'bass_heavy'
                elif vocal_ratio > 0.4 and bass_ratio < 0.4:  # More sensitive vocal detection
                    self.current_content_type = 'vocal'
                elif vocal_ratio > 0.3 and high_energy < vocal_energy * 0.5:  # Vocal without much highs
                    self.current_content_type = 'vocal'
                else:
                    self.current_content_type = 'instrumental'
            else:
                self.current_content_type = 'instrumental'
        
        # Update frequency allocation based on content
        if self.adaptive_allocation_enabled:
            if self.current_content_type == 'bass_heavy':
                target_allocation = 0.8
            elif self.current_content_type == 'vocal':
                target_allocation = 0.5  # Less bass allocation for vocals
            else:
                target_allocation = 0.7
            
            # Smooth transition
            self.current_allocation = (self.current_allocation * 0.95 + 
                                     target_allocation * 0.05)

    def apply_frequency_compensation(self, magnitudes, freqs=None):
        """Apply frequency compensation for better visualization
        
        Modern music frequency content (typical):
        - 20-250Hz: Bass instruments, kick drums (high energy)
        - 250-2000Hz: Most instruments, vocals (moderate energy)
        - 2000-6000Hz: Harmonics, presence (low energy)
        - 6000-10000Hz: Cymbals, air (very low energy)
        - >10000Hz: Mostly noise, little musical content
        """
        if not self.freq_compensation_enabled:
            return magnitudes
        
        compensated = magnitudes.copy()
        
        if freqs is not None:
            # Perceptual compensation optimized for music visualization
            # Goal: Show meaningful content, suppress noise
            
            # Content-aware compensation
            if self.current_content_type == 'vocal':
                # For vocals, significantly reduce bass to prevent masking
                sub_mask = freqs < 60
                compensated[sub_mask] *= 0.15  # Very heavy reduction (was 0.3)
                
                # Bass (60-250Hz) - strong reduction for vocals
                bass_mask = (freqs >= 60) & (freqs < 250)
                compensated[bass_mask] *= 0.2  # Much stronger reduction (was 0.4)
                
                # Low-mid (250-500Hz) - moderate reduction
                low_mid_mask = (freqs >= 250) & (freqs < 500)
                compensated[low_mid_mask] *= 0.6  # More reduction (was 0.8)
                
                # Midrange (500-2000Hz) - boost for vocal clarity
                mid_mask = (freqs >= 500) & (freqs < 2000)
                compensated[mid_mask] *= 1.5  # More boost for vocals (was 1.2)
            else:
                # Standard compensation for instrumental content
                # Sub-bass (below 60Hz) - slight reduction to prevent rumble dominance
                sub_mask = freqs < 60
                compensated[sub_mask] *= 0.8
                
                # Bass (60-250Hz) - preserve natural energy
                bass_mask = (freqs >= 60) & (freqs < 250)
                compensated[bass_mask] *= 1.0  # No change - bass is already visually prominent
                
                # Low-mid (250-500Hz) - slight boost for warmth
                low_mid_mask = (freqs >= 250) & (freqs < 500)
                compensated[low_mid_mask] *= 1.1
                
                # Midrange (500-2000Hz) - slight reduction to balance with bass
                mid_mask = (freqs >= 500) & (freqs < 2000)
                compensated[mid_mask] *= 0.85
            
            # High-mid (2000-6000Hz) - moderate boost for presence and vocals
            high_mid_mask = (freqs >= 2000) & (freqs < 6000)
            compensated[high_mid_mask] *= 1.2
            
            # High frequency (6000-10000Hz) - minimal boost (mostly noise in modern music)
            high_mask = (freqs >= 6000) & (freqs < 10000)
            compensated[high_mask] *= 0.8  # Actually reduce to prevent noise amplification
            
            # Ultra-high frequency (10000-20000Hz) - strong reduction (almost no musical content)
            ultra_high_mask = freqs >= 10000
            compensated[ultra_high_mask] *= 0.3  # Heavily reduce - this is mostly noise
            
            # Apply vocal suppression if needed
            if self.vocal_suppression > 0:
                vocal_mask = (freqs >= 800) & (freqs < 4000)
                compensated[vocal_mask] *= (1.0 - self.vocal_suppression * 0.5)
        
        return compensated

    def process_audio_spectrum(self):
        """Main audio processing with multi-resolution FFT"""
        if self.buffer_pos < FFT_SIZE_BASE:
            return None
        
        # Debug counter for periodic output
        if not hasattr(self, 'spectrum_debug_counter'):
            self.spectrum_debug_counter = 0
        
        # Get audio window
        if self.buffer_pos >= FFT_SIZE_BASE:
            audio_window = self.ring_buffer[self.buffer_pos - FFT_SIZE_BASE:self.buffer_pos].copy()
        else:
            audio_window = np.concatenate([
                self.ring_buffer[self.buffer_pos - FFT_SIZE_BASE:],
                self.ring_buffer[:self.buffer_pos]
            ])
        
        # Debug will be handled by comprehensive debug output
        
        # Auto-adjust gain
        self.auto_adjust_gain(audio_window)
        
        # Apply window function
        window = np.hanning(len(audio_window))
        audio_windowed = audio_window * window
        
        # Multi-resolution FFT processing
        multi_spectrum = self.process_multi_resolution_fft(audio_windowed)
        
        # Debug will be handled by comprehensive debug output
        
        # Update content detection
        self.update_content_type(multi_spectrum, audio_windowed)
        
        # Process spectrum data
        # Compute FFT for phase information
        fft_complex = np.fft.rfft(audio_windowed)
        
        # Use the spectrum directly
        spectrum = multi_spectrum
        
        # Debug will be handled by comprehensive debug output
        self.spectrum_debug_counter += 1
        
        # Apply initial normalization to prevent overflow
        if np.max(spectrum) > 0:
            # Normalize by dividing by a reference value for more stable scaling
            # Use a percentile instead of max to avoid outliers
            reference_value = np.percentile(spectrum, 98)
            if reference_value > 0:
                spectrum = spectrum / reference_value * 0.8  # Scale to reasonable range
            
        # Apply frequency compensation if enabled
        if self.freq_compensation_enabled:
            spectrum = self.apply_frequency_compensation(spectrum, self.freqs[:len(spectrum)])
        
        # Apply final normalization if enabled
        if self.normalization_enabled and np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
            
        # Map to frequency bands
        band_values = []
        peak_values = []
        
        for start_idx, end_idx in self.band_indices:
            if end_idx > len(spectrum):
                break
            # Use mean like the original OMEGA-2 for smoother visualization
            if end_idx > start_idx:
                band_value = np.mean(spectrum[start_idx:end_idx])
            else:
                band_value = spectrum[start_idx] if start_idx < len(spectrum) else 0
            
            # Apply a better scaling for visualization
            if band_value > 0:
                # Use power scaling for more natural visualization
                # Use square root for better dynamic range
                band_value = np.sqrt(band_value)
                
                # Don't apply additional frequency scaling here since
                # we already do it in apply_frequency_compensation
                
                # Final clamp to 0-1
                band_value = max(0, min(1, band_value))
            
            band_values.append(band_value)
            peak_values.append(band_value)  # Simplified - no peak hold for now
            
        band_values = np.array(band_values[:self.bars])
        peak_values = np.array(peak_values[:self.bars])
        
        # Debug will be handled by comprehensive debug output
        
        # Smoothing - more aggressive for cleaner visualization
        if self.smoothing_enabled and hasattr(self, 'prev_band_values'):
            # Frequency-dependent smoothing
            for i in range(len(band_values)):
                freq_hz = (self.band_indices[i][0] if i < len(self.band_indices) else i) * SAMPLE_RATE / FFT_SIZE_BASE
                
                # More smoothing for higher frequencies
                if freq_hz < 250:  # Bass - less smoothing
                    smooth_factor = 0.6
                elif freq_hz < 2000:  # Mids - moderate smoothing
                    smooth_factor = 0.75
                else:  # Highs - more smoothing
                    smooth_factor = 0.85
                
                band_values[i] = self.prev_band_values[i] * smooth_factor + band_values[i] * (1 - smooth_factor)
        
        self.prev_band_values = band_values.copy()
        
        spectrum_data = {
            'spectrum': spectrum,
            'fft_complex': fft_complex,
            'band_values': band_values,
            'peak_values': peak_values
        }
        
        # Create frequency array that matches spectrum length
        freq_array = np.linspace(0, SAMPLE_RATE / 2, len(spectrum_data['spectrum']))
        
        # Drum detection (moved here so it's available for genre classification)
        drum_events = self.drum_detector.process_audio(spectrum_data['spectrum'], spectrum_data['band_values'])
        spectrum_data['drum_events'] = drum_events
        
        # Update meters (only if not frozen and adaptive updater allows)
        dt = 1.0 / TARGET_FPS  # Frame time
        if not self.frozen_meters and self.adaptive_updater.should_update('meters'):
            self.professional_meters_panel.update(audio_windowed)
        if not self.frozen_vu_meters and self.adaptive_updater.should_update('vu_meters'):
            self.vu_meters_panel.update(audio_windowed, dt)
        
        # Update other panels (only if visible and not frozen and adaptive updater allows)
        harmonic_info = {}
        if self.show_harmonic and not self.frozen_harmonic and self.adaptive_updater.should_update('harmonic'):
            self.harmonic_analysis_panel.update(spectrum_data['spectrum'], freq_array)
            # Extract harmonic info from the panel if available
            if hasattr(self.harmonic_analysis_panel, 'harmonic_info'):
                harmonic_info = self.harmonic_analysis_panel.harmonic_info
        
        if self.show_pitch_detection and not self.frozen_pitch_detection and self.adaptive_updater.should_update('pitch_detection'):
            self.pitch_detection_panel.update(audio_windowed)
        # Update chromagram with genre context if available
        current_genre = None
        if hasattr(self, 'genre_classification_panel') and hasattr(self.genre_classification_panel, 'current_genre'):
            current_genre = self.genre_classification_panel.current_genre
            
        if self.show_chromagram and not self.frozen_chromagram and self.adaptive_updater.should_update('chromagram'):
            self.chromagram_panel.update(spectrum_data['spectrum'], audio_windowed, freq_array, current_genre)
        
        # Extract drum info for genre classification
        drum_info = {}
        if 'drum_events' in spectrum_data and spectrum_data['drum_events']:
            for event in spectrum_data['drum_events']:
                if isinstance(event, dict):
                    event_type = event.get('type', '')
                    magnitude = event.get('magnitude', 0.0)
                    if 'kick' in event_type.lower():
                        drum_info['kick'] = {'magnitude': magnitude, 'kick_detected': True}
                    elif 'snare' in event_type.lower():
                        drum_info['snare'] = {'magnitude': magnitude, 'snare_detected': True}
                    elif 'hihat' in event_type.lower():
                        drum_info['hihat'] = {'magnitude': magnitude, 'hihat_detected': True}
        
        # Ensure drum_info has the expected structure even if no drums detected
        if 'kick' not in drum_info:
            drum_info['kick'] = {'magnitude': 0.0, 'kick_detected': False}
        if 'snare' not in drum_info:
            drum_info['snare'] = {'magnitude': 0.0, 'snare_detected': False}
        
        # Update genre classification with harmonic features
        chromagram_data = None
        current_chord = None
        detected_key = None
        
        if hasattr(self, 'chromagram_panel'):
            chromagram_data = getattr(self.chromagram_panel, 'chromagram', None)
            current_chord = getattr(self.chromagram_panel, 'current_chord', None)
            detected_key = getattr(self.chromagram_panel, 'detected_key', None)
        
        if self.show_genre_classification and not self.frozen_genre_classification and self.adaptive_updater.should_update('genre_classification'):
            self.genre_classification_panel.update(
                spectrum_data['spectrum'], audio_windowed, freq_array,
                drum_info, harmonic_info,
                chromagram_data, current_chord, detected_key
            )
        
        # Update integrated music panel if active
        if self.show_integrated_music and not self.frozen_integrated_music and self.adaptive_updater.should_update('integrated_music'):
            self.integrated_music_panel.update(
                spectrum_data['spectrum'], audio_windowed, freq_array,
                drum_info, harmonic_info
            )
        
        # Phase coherence analysis
        phase_coherence = self.phase_analyzer.analyze_mono(spectrum_data['fft_complex'])
        spectrum_data['phase_coherence'] = phase_coherence
        
        # Transient detection
        transient_result = self.transient_analyzer.analyze_transients(audio_windowed)
        if transient_result.get('transients'):
            self.transient_events.extend(transient_result['transients'])
            self.last_transient_time = time.time()
        
        # Room mode analysis (only if not frozen and adaptive updater allows)
        if not self.frozen_room_analysis and self.adaptive_updater.should_update('room_analysis'):
            room_modes = self.room_analyzer.detect_room_modes(spectrum_data['spectrum'], freq_array)
            self.room_modes = room_modes
        
        # Voice detection
        voice_result = self.voice_detector.detect_voice_realtime(audio_windowed)
        self.voice_active = voice_result.get('voice_detected', False)
        self.voice_confidence = voice_result.get('confidence', 0.0)
        
        # Force voice detection for testing if audio level is significant
        if not self.voice_active and np.max(np.abs(audio_windowed)) > 0.01:
            # Check spectrum for vocal frequencies
            vocal_range_start = int(200 * len(spectrum_data['spectrum']) / (SAMPLE_RATE / 2))
            vocal_range_end = int(4000 * len(spectrum_data['spectrum']) / (SAMPLE_RATE / 2))
            if vocal_range_end <= len(spectrum_data['spectrum']):
                vocal_energy = np.mean(spectrum_data['spectrum'][vocal_range_start:vocal_range_end])
                total_energy = np.mean(spectrum_data['spectrum'])
                if total_energy > 0 and vocal_energy / total_energy > 0.3:
                    self.voice_active = True
                    self.voice_confidence = 50.0
        
        
        # Update bass zoom if active
        if self.show_bass_zoom:
            # Extract drum info from drum events
            drum_info = {}
            if drum_events:
                for event in drum_events:
                    if isinstance(event, dict):
                        event_type = event.get('type', '')
                        magnitude = event.get('magnitude', 0.0)
                        if 'kick' in event_type.lower():
                            drum_info['kick'] = magnitude
                        elif 'sub' in event_type.lower():
                            drum_info['sub'] = magnitude
                        elif 'floor' in event_type.lower():
                            drum_info['floor'] = magnitude
                        elif 'bass' in event_type.lower():
                            drum_info['bass'] = magnitude
            
            # If no drum events, analyze bass frequencies directly
            if not drum_info and len(spectrum_data['spectrum']) > 0:
                freq_bin_width = SAMPLE_RATE / (2 * len(spectrum_data['spectrum']))
                
                # Analyze specific bass frequencies
                sub_idx = int(40 / freq_bin_width)
                kick_idx = int(60 / freq_bin_width)
                floor_idx = int(80 / freq_bin_width) 
                bass_idx = int(110 / freq_bin_width)
                
                if sub_idx < len(spectrum_data['spectrum']):
                    drum_info['sub'] = spectrum_data['spectrum'][sub_idx]
                if kick_idx < len(spectrum_data['spectrum']):
                    drum_info['kick'] = spectrum_data['spectrum'][kick_idx]
                if floor_idx < len(spectrum_data['spectrum']):
                    drum_info['floor'] = spectrum_data['spectrum'][floor_idx]
                if bass_idx < len(spectrum_data['spectrum']):
                    drum_info['bass'] = spectrum_data['spectrum'][bass_idx]
            
            # Pass audio data and drum info (only if not frozen and adaptive updater allows)
            if not self.frozen_bass_zoom and self.adaptive_updater.should_update('bass_zoom'):
                self.bass_zoom_panel.update(audio_windowed, drum_info)
        
        # Store history
        self.spectrum_history_multi.append(spectrum_data)
        if len(self.spectrum_history_multi) > self.max_spectrum_history:
            self.spectrum_history_multi.pop(0)
        
        # Store for debug snapshots
        self.last_spectrum_data = spectrum_data
        
        return spectrum_data

    def toggle_compensation_ab(self):
        """Toggle between compensated (A) and raw (B) modes"""
        self.compensation_ab_mode = not self.compensation_ab_mode
        
        if self.compensation_ab_mode:  # B mode - disable all compensation
            self.freq_compensation_enabled = False
            self.psychoacoustic_enabled = False
            self.normalization_enabled = False
            print("Mode B: Raw signal (no compensation)")
        else:  # A mode - enable all compensation
            self.freq_compensation_enabled = True
            self.psychoacoustic_enabled = True
            self.normalization_enabled = True
            print("Mode A: Full compensation enabled")

    def test_all_panels(self):
        """Test all panels by freezing/unfreezing them sequentially"""
        if not hasattr(self, 'panel_test_counter'):
            self.panel_test_counter = 0
        
        # Test sequence every 300 frames (5 seconds)
        if self.panel_test_counter % 300 == 0:
            panel_index = (self.panel_test_counter // 300) % 8
            
            # Unfreeze all panels first
            self.frozen_meters = False
            self.frozen_bass_zoom = False
            self.frozen_harmonic = False
            self.frozen_room_analysis = False
            self.frozen_pitch_detection = False
            self.frozen_chromagram = False
            self.frozen_genre_classification = False
            
            # Freeze all panels except the one being tested
            if panel_index == 0:
                print("\n[TEST] Testing Panel 1/7: Professional Meters (all others frozen)")
                self.frozen_bass_zoom = True
                self.frozen_harmonic = True
                self.frozen_room_analysis = True
                self.frozen_pitch_detection = True
                self.frozen_chromagram = True
                self.frozen_genre_classification = True
            elif panel_index == 1:
                print("\n[TEST] Testing Panel 2/7: Bass Zoom (all others frozen)")
                self.frozen_meters = True
                self.frozen_harmonic = True
                self.frozen_room_analysis = True
                self.frozen_pitch_detection = True
                self.frozen_chromagram = True
                self.frozen_genre_classification = True
            elif panel_index == 2:
                print("\n[TEST] Testing Panel 3/7: Harmonic Analysis (all others frozen)")
                self.frozen_meters = True
                self.frozen_bass_zoom = True
                self.frozen_room_analysis = True
                self.frozen_pitch_detection = True
                self.frozen_chromagram = True
                self.frozen_genre_classification = True
            elif panel_index == 3:
                print("\n[TEST] Testing Panel 4/7: Room Mode Analysis (all others frozen)")
                self.frozen_meters = True
                self.frozen_bass_zoom = True
                self.frozen_harmonic = True
                self.frozen_pitch_detection = True
                self.frozen_chromagram = True
                self.frozen_genre_classification = True
            elif panel_index == 4:
                print("\n[TEST] Testing Panel 5/7: Pitch Detection (all others frozen)")
                self.frozen_meters = True
                self.frozen_bass_zoom = True
                self.frozen_harmonic = True
                self.frozen_room_analysis = True
                self.frozen_chromagram = True
                self.frozen_genre_classification = True
            elif panel_index == 5:
                print("\n[TEST] Testing Panel 6/7: Chromagram (all others frozen)")
                self.frozen_meters = True
                self.frozen_bass_zoom = True
                self.frozen_harmonic = True
                self.frozen_room_analysis = True
                self.frozen_pitch_detection = True
                self.frozen_genre_classification = True
            elif panel_index == 6:
                print("\n[TEST] Testing Panel 7/7: Genre Classification (all others frozen)")
                self.frozen_meters = True
                self.frozen_bass_zoom = True
                self.frozen_harmonic = True
                self.frozen_room_analysis = True
                self.frozen_pitch_detection = True
                self.frozen_chromagram = True
            else:
                print("\n[TEST] All panels active - cycle complete")
        
        self.panel_test_counter += 1
    
    def handle_keyboard(self, event):
        """Handle keyboard shortcuts"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_SPACE:
            self.toggle_compensation_ab()
        elif event.key == pygame.K_m:
            self.frozen_meters = not self.frozen_meters
            print(f"Professional Meters: {'FROZEN' if self.frozen_meters else 'ACTIVE'}")
        elif event.key == pygame.K_h:
            self.frozen_harmonic = not self.frozen_harmonic
            print(f"Harmonic Analysis: {'FROZEN' if self.frozen_harmonic else 'ACTIVE'}")
        elif event.key == pygame.K_z:
            self.frozen_bass_zoom = not self.frozen_bass_zoom
            print(f"\n{'='*50}")
            print(f"Bass Zoom Panel: {'FROZEN' if self.frozen_bass_zoom else 'ACTIVE'}")
            print(f"{'='*50}\n")
        elif event.key == pygame.K_r:
            self.frozen_room_analysis = not self.frozen_room_analysis
            print(f"Room Analysis: {'FROZEN' if self.frozen_room_analysis else 'ACTIVE'}")
        elif event.key == pygame.K_p:
            self.frozen_pitch_detection = not self.frozen_pitch_detection
            print(f"Pitch Detection: {'FROZEN' if self.frozen_pitch_detection else 'ACTIVE'}")
        elif event.key == pygame.K_c:
            # C key disabled - chromagram is part of integrated music panel
            print("Use 'I' to toggle Integrated Music Analysis panel (includes chromagram)")
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key == pygame.K_j:
            # J key disabled - genre classification is part of integrated music panel  
            print("Use 'I' to toggle Integrated Music Analysis panel (includes genre classification)")
        elif event.key == pygame.K_u:
            self.frozen_vu_meters = not self.frozen_vu_meters
            print(f"VU Meters: {'FROZEN' if self.frozen_vu_meters else 'ACTIVE'}")
        elif event.key == pygame.K_v:
            self.show_voice_detection = not self.show_voice_detection
        elif event.key == pygame.K_f:
            self.show_formant = not self.show_formant
        elif event.key == pygame.K_a:
            self.show_advanced_voice = not self.show_advanced_voice
        elif event.key == pygame.K_d:
            # Print debug snapshot on demand
            if hasattr(self, 'last_spectrum_data'):
                self.print_debug_snapshot(self.last_spectrum_data)
                print("Debug snapshot printed to terminal")
            else:
                print("No spectrum data available for debug snapshot")
        elif event.key == pygame.K_k:
            self.adaptive_allocation_enabled = not self.adaptive_allocation_enabled
            print(f"Adaptive frequency allocation: {'ON' if self.adaptive_allocation_enabled else 'OFF'}")
        elif event.key == pygame.K_q:
            self.freq_compensation_enabled = not self.freq_compensation_enabled
            print(f"Frequency compensation: {'ON' if self.freq_compensation_enabled else 'OFF'}")
        elif event.key == pygame.K_w:
            self.psychoacoustic_enabled = not self.psychoacoustic_enabled
            print(f"Psychoacoustic processing: {'ON' if self.psychoacoustic_enabled else 'OFF'}")
        elif event.key == pygame.K_e:
            self.normalization_enabled = not self.normalization_enabled
            print(f"Normalization: {'ON' if self.normalization_enabled else 'OFF'}")
        elif event.key == pygame.K_t:
            self.test_mode = not self.test_mode
            print(f"Test mode: {'ON' if self.test_mode else 'OFF'}")
        elif event.key == pygame.K_i:
            # 'I' key disabled - Integrated Music Analysis panel is always visible
            print("Integrated Music Analysis panel is always visible (no toggle needed)")
        elif event.key == pygame.K_s:
            self.save_screenshot()
        elif event.key == pygame.K_F11:
            self.toggle_fullscreen()
        elif event.key >= pygame.K_1 and event.key <= pygame.K_9:
            # Professional window presets with minimum heights for panels
            # Minimum height: 1150px (no panels), 1420px (bass), 1690px (bass+1 row), 1950px (bass+2 rows)
            presets = [
                (1280, 1150),  # 1 - Minimum (no panels)
                (1400, 1420),  # 2 - With bass zoom
                (1600, 1420),  # 3 - HD+ width with bass
                (1920, 1420),  # 4 - Full HD width with bass
                (1920, 1690),  # 5 - Full HD with 1 row panels
                (2560, 1690),  # 6 - QHD width with 1 row
                (2560, 1950),  # 7 - QHD with 2 rows panels
                (3440, 1950),  # 8 - Ultra-wide with 2 rows
                (3840, 2160)   # 9 - 4K UHD (full panels)
            ]
            preset_idx = event.key - pygame.K_1
            if preset_idx < len(presets):
                # Use the actual preset dimensions
                new_width, new_height = presets[preset_idx]
                # Calculate the actual required height for current panels
                required_height = self.calculate_required_height()
                # Use the larger of preset height or required height
                final_height = max(new_height, required_height)
                self.resize_window(new_width, final_height)
                # Don't auto-adjust after setting a preset
                self.preset_locked = True
                print(f"Window preset {preset_idx + 1}: {new_width}x{final_height} (min required: {required_height}px)")
        elif event.key == pygame.K_UP:
            self.input_gain = min(self.input_gain * 1.2, 10.0)
            print(f"Input gain: {self.input_gain:.1f}x ({20*np.log10(self.input_gain):.1f}dB)")
        elif event.key == pygame.K_DOWN:
            self.input_gain = max(self.input_gain / 1.2, 0.1)
            print(f"Input gain: {self.input_gain:.1f}x ({20*np.log10(self.input_gain):.1f}dB)")

    def resize_window(self, width, height):
        """Resize the window and update UI elements"""
        # Avoid redundant resizes
        if self.width == width and self.height == height:
            print(f"[RESIZE] Skipping redundant resize to {width}x{height}")
            return
            
        print(f"[RESIZE] Resizing from {self.width}x{self.height} to {width}x{height}")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.update_fonts(width)
        
        # Update display interface
        self.display.resize(width, height)
        self.display.set_fonts({
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny,
            'grid': self.font_grid,
            'mono': self.font_tiny  # Use tiny font as mono for now
        })
        
        # Update panel fonts
        font_dict = {
            'large': self.font_large,
            'medium': self.font_medium,
            'small': self.font_small,
            'tiny': self.font_tiny
        }
        
        self.professional_meters_panel.set_fonts({
            **font_dict,
            'meters': self.font_meters
        })
        
        # Update all other panels
        for panel in [self.vu_meters_panel, self.bass_zoom_panel, self.harmonic_analysis_panel,
                      self.pitch_detection_panel, self.chromagram_panel, self.genre_classification_panel,
                      self.integrated_music_panel]:
            panel.set_fonts(font_dict)
        
        print(f"Window resized to {width}x{height}")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # Get monitor size
            info = pygame.display.Info()
            self.resize_window(info.current_w, info.current_h)
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        else:
            self.resize_window(DEFAULT_WIDTH, DEFAULT_HEIGHT)

    def save_screenshot(self):
        """Save a screenshot of the current display"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"omega_analyzer_screenshot_{timestamp}_{self.screenshot_counter:03d}.png"
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved: {filename}")
        self.screenshot_counter += 1

    def print_debug_snapshot(self, spectrum_data):
        """Print a single debug snapshot to terminal"""
        # Print header with timestamp
        print("\n" + "=" * 80)
        print("PROFESSIONAL AUDIO ANALYZER V4 OMEGA-4 - DEBUG SNAPSHOT -", time.strftime("%H:%M:%S"))
        print("=" * 80)
        
        # Performance metrics
        avg_frame_time = np.mean(list(self.frame_times)) if self.frame_times else 0
        avg_process_time = np.mean(list(self.process_times)) if self.process_times else 0
        current_fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        
        print("\n[PERFORMANCE METRICS]")
        print(f"FPS: Current={int(current_fps)}, Average={current_fps:.1f}, Target={TARGET_FPS}")
        print(f"Processing latency: {avg_process_time:.1f}ms")
        print(f"Audio buffer: {CHUNK_SIZE} samples ({CHUNK_SIZE/SAMPLE_RATE*1000:.1f}ms)")
        print(f"FFT window: {FFT_SIZE_BASE} samples ({FFT_SIZE_BASE/SAMPLE_RATE*1000:.1f}ms)")
        print(f"Total latency: {(CHUNK_SIZE + FFT_SIZE_BASE)/SAMPLE_RATE*1000 + avg_process_time:.1f}ms")
        
        # Frequency mapping info
        print("\n[PERCEPTUAL FREQUENCY MAPPING]")
        print("Mode: Mel-scale (Perceptually Uniform)")
        print("Distribution:")
        
        # Count bars in each frequency range
        freq_ranges = [
            (20, 250, "Bass"),
            (250, 500, "Low-mid"),
            (500, 2000, "Mid"),
            (2000, 6000, "High-mid"),
            (6000, 20000, "High")
        ]
        
        total_bars = len(self.band_indices)
        for low, high, name in freq_ranges:
            count = sum(1 for start, end in self.band_indices 
                       if start * SAMPLE_RATE / FFT_SIZE_BASE >= low 
                       and end * SAMPLE_RATE / FFT_SIZE_BASE <= high)
            percentage = count / total_bars * 100 if total_bars > 0 else 0
            print(f"  {name} ({low}-{high}Hz): {percentage:.0f}% of bars ({count} bars)")
        
        print(f"Performance: {'✅ Excellent' if current_fps > 55 else '⚠️ Degraded'}")
        
        # Mini spectrum visualization
        if 'band_values' in spectrum_data:
            self.print_mini_spectrum(spectrum_data['band_values'])
        
        # Frequency distribution analysis
        if 'spectrum' in spectrum_data:
            self.print_frequency_distribution(spectrum_data['spectrum'])
        
        # Professional metering
        if hasattr(self, 'professional_meters_panel'):
            self.print_metering_info()
        
        # Pitch detection
        if hasattr(self.pitch_detection_panel, 'detected_pitch'):
            print(f"\n[OMEGA PITCH DETECTION]")
            pitch = getattr(self.pitch_detection_panel, 'detected_pitch', None)
            if pitch:
                print(f"Pitch: {pitch}")
                confidence = getattr(self.pitch_detection_panel, 'pitch_confidence', 0)
                print(f"Confidence: {confidence:.1f}%")
            else:
                print("No pitch detected")
        
        # Transient analysis
        if hasattr(self, 'transient_events') and self.transient_events:
            print(f"\n[TRANSIENT ANALYSIS]")
            recent_transients = list(self.transient_events)[-5:]
            if recent_transients:
                print(f"Recent transients: {len(recent_transients)}")
                for t in recent_transients[-3:]:
                    if isinstance(t, dict):
                        print(f"  Type: {t.get('type', 'unknown')}, Time: {t.get('time', 0):.3f}s")
        
        # Bass detail (analyze bass frequencies directly from spectrum)
        print(f"\n[BASS DETAIL 20-300Hz]")
        if 'spectrum' in spectrum_data and len(spectrum_data['spectrum']) > 0:
            freq_bin_width = SAMPLE_RATE / (2 * len(spectrum_data['spectrum']))
            
            # Analyze specific bass frequencies
            bass_freqs = [
                (40, "Sub-bass"),
                (60, "Kick"),
                (80, "Floor tom"),
                (110, "Bass line"),
                (160, "Low strings"),
                (200, "Upper bass")
            ]
            
            for freq, name in bass_freqs:
                idx = int(freq / freq_bin_width)
                if idx < len(spectrum_data['spectrum']):
                    # Get magnitude value (already processed/normalized)
                    magnitude = spectrum_data['spectrum'][idx]
                    
                    # Convert to dB with proper reference
                    # Use a reference that makes sense for normalized data
                    if magnitude > 0:
                        # Add small offset to prevent log(1) = 0
                        # and scale to more realistic range
                        db_value = 20 * np.log10(magnitude / 2.0)
                        # Clamp to reasonable range
                        db_value = max(-60, min(-3, db_value))  # Never show 0dB
                        
                        # Create bar visualization
                        bar_length = int((db_value + 60) / 3)  # Scale from -60dB to 0dB
                        bar_length = max(0, min(bar_length, 20))
                        bar = "█" * bar_length
                        print(f"{name:12} ({freq:3d}Hz): {bar:20} {db_value:+6.1f} dB")
                    else:
                        print(f"{name:12} ({freq:3d}Hz): {'':20}  -60.0 dB")
        
        # Voice detection
        print(f"\n[VOICE DETECTION]")
        print(f"Has Voice: {'YES' if self.voice_active else 'NO'}")
        print(f"Confidence: {self.voice_confidence:.1f}%")
        print(f"Content Type: {self.current_content_type.upper()}")
        
        # Dynamic range
        if 'band_values' in spectrum_data:
            band_values = spectrum_data['band_values']
            active_bars = np.count_nonzero(band_values > 0.01)
            dynamic_range = 20 * np.log10(np.max(band_values) / (np.min(band_values[band_values > 0]) + 1e-10)) if np.any(band_values > 0) else 0
            
            print(f"\n[DYNAMIC RANGE]")
            print(f"Dynamic Range: {dynamic_range:.1f} dB")
            print(f"Active bars: {active_bars}/{len(band_values)} ({active_bars/len(band_values)*100:.1f}%)")
        
        print("\n" + "=" * 80)
    
    def print_mini_spectrum(self, band_values):
        """Print a mini ASCII spectrum visualization"""
        print("\n[SPECTRUM VISUALIZATION]")
        
        # Downsample to 80 characters width using averaging instead of sampling
        width = 80
        if len(band_values) > width:
            # Average bins instead of sampling to avoid gaps
            values = []
            bin_size = len(band_values) / width
            for i in range(width):
                start_idx = int(i * bin_size)
                end_idx = int((i + 1) * bin_size)
                # Take the maximum value in each bin to preserve peaks
                bin_max = np.max(band_values[start_idx:end_idx])
                values.append(bin_max)
            values = np.array(values)
        else:
            values = band_values
        
        # Create ASCII bars
        height = 10
        for h in range(height, -1, -1):
            line = ""
            for v in values:
                threshold = h / height
                if v > threshold:
                    if v > 0.8:
                        line += "█"
                    elif v > 0.6:
                        line += "▓"
                    elif v > 0.4:
                        line += "▒"
                    elif v > 0.2:
                        line += "░"
                    else:
                        line += "▪"
                else:
                    line += " "
            print(line)
        
        # Frequency scale
        print("-" * 80)
        print("20Hz" + " " * 35 + "1kHz" + " " * 35 + "20kHz")
        
        # Stats
        if len(band_values) > 0:
            print(f"\nStats: Max={np.max(band_values):.2f}, Avg={np.mean(band_values):.2f}, Min={np.min(band_values):.2f}")
    
    def print_frequency_distribution(self, spectrum):
        """Print frequency band energy distribution"""
        print("\n[FREQUENCY DISTRIBUTION]")
        
        freq_bands = [
            (60, 250, "Bass"),
            (250, 500, "Low-mid"),
            (500, 2000, "Mid"),
            (2000, 4000, "High-mid"),
            (4000, 6000, "Presence"),
            (6000, 10000, "Brilliance"),
            (10000, 20000, "Air")
        ]
        
        freq_bin_width = SAMPLE_RATE / (2 * len(spectrum))
        
        for low, high, name in freq_bands:
            low_bin = int(low / freq_bin_width)
            high_bin = int(high / freq_bin_width)
            
            if high_bin > len(spectrum):
                high_bin = len(spectrum)
            
            if low_bin < high_bin:
                band_energy = np.mean(spectrum[low_bin:high_bin])
                max_energy = np.max(spectrum[low_bin:high_bin])
                
                # Create bar visualization
                bar_length = int(band_energy * 20)
                bar = "▂" * min(bar_length, 20)
                
                print(f"{name:12} [{low:5d}-{high:5d}Hz]: {bar} avg={band_energy:.2f} max={max_energy:.2f}")
    
    def print_metering_info(self):
        """Print professional metering information"""
        print("\n[PROFESSIONAL METERING - ITU-R BS.1770-4]")
        
        # Get metering data if available
        if hasattr(self.professional_meters_panel, 'momentary_lufs'):
            lufs_m = getattr(self.professional_meters_panel, 'momentary_lufs', -100.0)
            lufs_s = getattr(self.professional_meters_panel, 'short_term_lufs', -100.0)
            lufs_i = getattr(self.professional_meters_panel, 'integrated_lufs', -100.0)
            lra = getattr(self.professional_meters_panel, 'loudness_range', 0.0)
            true_peak = getattr(self.professional_meters_panel, 'true_peak_db', -100.0)
            
            print(f"Momentary LUFS (400ms): {lufs_m:6.1f} LUFS")
            print(f"Short-term LUFS (3s):   {lufs_s:6.1f} LUFS")
            print(f"Integrated LUFS:        {lufs_i:6.1f} LU")
            print(f"Loudness Range (LRA):   {lra:6.1f} LU")
            
            # Target recommendations
            if lufs_s > -14:
                print("Status: Loud (streaming/YouTube target: -14 LUFS)")
            elif lufs_s > -16:
                print("Status: Good for streaming")
            elif lufs_s > -23:
                print("Status: Good for broadcast")
            else:
                print("Status: Quiet")
            
            print(f"\nTrue Peak (4x oversampled): {true_peak:+6.1f} dBTP")
            if true_peak > -1:
                print("  ⚠️  WARNING: Potential clipping!")
        else:
            print("Metering data not available - enable professional meters panel (M)")
    
    def draw_header_panel(self, spectrum_data):
        """Draw comprehensive header with title and feature status table"""
        # Calculate dynamic spacing based on window width
        padding = max(20, int(self.width * 0.01))  # Scale padding with window size
        margin = 10
        
        # Ensure minimum font sizes at different resolutions
        if self.width > 2500:
            header_font = self.font_large
            feature_font = self.font_small
        elif self.width > 1920:
            header_font = self.font_large
            feature_font = self.font_tiny
        else:
            header_font = self.font_medium if self.width < 1400 else self.font_large
            feature_font = self.font_tiny
        
        # Calculate title height
        title_text = header_font.render("OMEGA-4 Professional Audio Analyzer", True, TEXT_COLOR)
        title_height = title_text.get_height()
        title_y = margin
        
        # Separator position
        separator_y = title_y + title_height + margin
        
        # Calculate row dimensions dynamically
        row_start_y = separator_y + margin
        row_height = feature_font.get_height() + 5  # Add some padding
        
        # Calculate column widths dynamically
        # Feature columns should take up ~65% of width, tech columns ~35%
        available_width = self.width - (2 * padding)
        feature_section_width = int(available_width * 0.65)
        tech_section_width = int(available_width * 0.35)
        
        # Feature columns (3 equal columns)
        feature_col_width = feature_section_width // 3
        col1_x = padding
        col2_x = padding + feature_col_width
        col3_x = padding + (2 * feature_col_width)
        
        # Tech columns (2 columns, label and value)
        tech_label_x = padding + feature_section_width + (margin * 2)
        tech_value_x = tech_label_x + (tech_section_width * 0.4)
        
        # Feature states (left side)
        features = [
            # Column 1
            [
                ("Voice Detection (V)", self.show_voice_detection),
                ("Professional Meters (M)", self.show_meters),
                ("Bass Zoom (Z)", self.show_bass_zoom),
                ("Grid Display (G)", self.show_grid),
            ],
            # Column 2
            [
                ("Harmonic Analysis (H)", self.show_harmonic),
                ("Pitch Detection (P)", self.show_pitch_detection),
                # ("Chromagram (C)", self.show_chromagram),  # Part of integrated panel
                ("Room Analysis (R)", self.show_room_analysis),
            ],
            # Column 3
            [
                ("VU Meters (U)", self.show_vu_meters),
                # ("Genre Classification (J)", self.show_genre_classification),  # Part of integrated panel
                ("Integrated Music", self.show_integrated_music),  # Always on, no toggle
                ("Test Mode (T)", self.test_mode),
                ("A/B Mode", self.compensation_ab_mode),
            ]
        ]
        
        # Calculate content bounds first to determine header height
        total_feature_rows = max(len(column) for column in features)
        
        # Get technical details count
        tech_details = []
        tech_details.append(("Bars", f"{self.bars}"))
        tech_details.append(("Gain", f"{self.input_gain:.1f}x ({20*np.log10(self.input_gain):+.1f}dB)"))
        tech_details.append(("Content", self.current_content_type.replace('_', ' ').title()))
        tech_details.append(("Latency", f"{(CHUNK_SIZE + FFT_SIZE_BASE)/SAMPLE_RATE*1000:.1f}ms"))
        
        if hasattr(self, 'professional_meters_panel') and hasattr(self.professional_meters_panel, 'lufs_integrated'):
            lufs_i = getattr(self.professional_meters_panel, 'lufs_integrated', -inf)
            if lufs_i > -70:
                tech_details.append(("LUFS-I", f"{lufs_i:.1f}"))
        
        if self.voice_active:
            tech_details.append(("Voice", f"YES ({self.voice_confidence:.0f}%)"))
        
        tech_details = tech_details[:5]  # Limit to 5 items
        total_tech_rows = len(tech_details)
        
        # Calculate total header height
        total_rows = max(total_feature_rows, total_tech_rows)
        content_height = row_start_y + (total_rows * row_height) + margin
        
        # Draw header background first
        header_bg = pygame.Surface((self.width, content_height))
        header_bg.set_alpha(240)
        header_bg.fill((15, 20, 30))
        self.screen.blit(header_bg, (0, 0))
        
        # Draw title
        title_rect = title_text.get_rect(centerx=self.width//2, y=title_y)
        self.screen.blit(title_text, title_rect)
        
        # Draw separator line
        pygame.draw.line(self.screen, (60, 70, 90), (padding, separator_y), (self.width - padding, separator_y), 2)
        
        # Draw feature states with proper text wrapping/truncation
        for col_idx, column in enumerate(features):
            x_pos = [col1_x, col2_x, col3_x][col_idx]
            max_feature_width = feature_col_width - 60  # Leave space for ON/OFF
            
            for row_idx, (feature_name, enabled) in enumerate(column):
                y_pos = row_start_y + row_idx * row_height
                
                # Truncate feature name if too long
                feature_text = feature_font.render(feature_name, True, (180, 180, 200))
                if feature_text.get_width() > max_feature_width:
                    # Truncate with ellipsis
                    truncated = feature_name
                    while feature_font.render(truncated + "...", True, (180, 180, 200)).get_width() > max_feature_width and len(truncated) > 5:
                        truncated = truncated[:-1]
                    feature_text = feature_font.render(truncated + "...", True, (180, 180, 200))
                
                self.screen.blit(feature_text, (x_pos, y_pos))
                
                # Status text (ON/OFF)
                status_color = (100, 255, 100) if enabled else (150, 150, 170)
                status_text = "ON" if enabled else "OFF"
                status_render = feature_font.render(status_text, True, status_color)
                
                # Right-align status within column
                status_x = x_pos + feature_col_width - status_render.get_width() - 10
                self.screen.blit(status_render, (status_x, y_pos))
        
        # Draw technical details with dynamic sizing
        for row_idx, (label, value) in enumerate(tech_details):
            y_pos = row_start_y + row_idx * row_height
            
            # Label
            label_text = feature_font.render(f"{label}:", True, (180, 180, 200))
            self.screen.blit(label_text, (tech_label_x, y_pos))
            
            # Value - truncate if too long
            value_text = feature_font.render(value, True, (220, 230, 240))
            max_value_width = tech_section_width * 0.6 - 10
            if value_text.get_width() > max_value_width:
                # Truncate value
                truncated_value = value
                while feature_font.render(truncated_value + "...", True, (220, 230, 240)).get_width() > max_value_width and len(truncated_value) > 3:
                    truncated_value = truncated_value[:-1]
                value_text = feature_font.render(truncated_value + "...", True, (220, 230, 240))
            
            self.screen.blit(value_text, (tech_value_x, y_pos))
        
        # Draw FPS in header (top-right corner)
        if self.show_fps and hasattr(self, 'frame_times'):
            avg_frame_time = np.mean(list(self.frame_times)) if self.frame_times else 0
            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            fps_color = (100, 255, 100) if fps > 55 else (255, 200, 100) if fps > 30 else (255, 100, 100)
            fps_text = self.font_small.render(f"FPS: {fps:.1f}", True, fps_color)
            fps_rect = fps_text.get_rect(right=self.width - padding, y=title_y)
            self.screen.blit(fps_text, fps_rect)
        
        return content_height  # Return actual header height
    
    def draw_footer(self):
        """Draw footer with help and copyright information"""
        footer_height = 80
        footer_y = self.height - footer_height
        
        # Draw footer background
        footer_rect = pygame.Rect(0, footer_y, self.width, footer_height)
        pygame.draw.rect(self.screen, (20, 25, 35), footer_rect)
        pygame.draw.line(self.screen, (60, 70, 90), (0, footer_y), (self.width, footer_y), 2)
        
        # Help text
        help_texts = [
            "Keys: [1-9] Size Presets (1:Min 2-4:Bass 5-7:Panels 8-9:Full) | [U] VU | [M]eters [Z]oom [H]arm [R]oom [P]itch [C]hroma [J]Genre [I]ntegrated",
            "[Space] A/B Compensation | [Q] Freq Comp | [W] Psychoacoustic | [E] Normalize | [S]creenshot | [F11] Fullscreen | [G]rid",
            "© 2024 OMEGA-4 Professional Audio Analyzer | github.com/anthropics/audio-geometric-visualizer"
        ]
        
        # Draw help text
        y_offset = footer_y + 10
        for i, text in enumerate(help_texts):
            if i < 2 and self.font_tiny:
                # Help text in smaller font
                text_surf = self.font_tiny.render(text, True, (150, 170, 190))
                text_rect = text_surf.get_rect(centerx=self.width // 2, y=y_offset)
                self.screen.blit(text_surf, text_rect)
                y_offset += 20
            elif i == 2 and self.font_tiny:
                # Copyright in even smaller font or same tiny font
                text_surf = self.font_tiny.render(text, True, (100, 120, 140))
                text_rect = text_surf.get_rect(centerx=self.width // 2, y=y_offset + 5)
                self.screen.blit(text_surf, text_rect)
    
    def draw_debug_info(self, spectrum_data):
        """Draw debug information overlay on screen"""
        if not self.show_debug:
            return
        
        # Draw debug info on screen (not terminal)
        debug_info = [
            f"Content Type: {self.current_content_type}",
            f"Input Gain: {self.input_gain:.2f}x ({20*np.log10(self.input_gain):.1f}dB)",
            f"Bass Allocation: {self.current_allocation:.1%}",
            f"Voice Active: {self.voice_active} ({self.voice_confidence:.1%})",
            f"Compensation: {'A (ON)' if not self.compensation_ab_mode else 'B (OFF)'}",
            f"Adaptive Alloc: {'ON' if self.adaptive_allocation_enabled else 'OFF'}",
            f"Frame Time: {np.mean(list(self.frame_times)) if self.frame_times else 0:.1f}ms",
            f"Process Time: {np.mean(list(self.process_times)) if self.process_times else 0:.1f}ms"
        ]
        
        y = self.height - 200
        for info in debug_info:
            text = self.font_tiny.render(info, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 20

    def calculate_required_height(self):
        """Calculate required window height based on active panels using grid layout"""
        # Fixed layout components
        header_height = 120
        spectrum_height = 800
        spectrum_top_margin = 10
        spectrum_bottom_margin = 100
        panel_start_offset = 40
        footer_height = 80
        
        # Panel dimensions
        panel_height = 250  # Reduced slightly to fit better with footer gap
        panel_padding = 10
        max_columns = 4  # Changed to 4 panels per row
        
        # Base height (header + spectrum + margins)
        base_height = header_height + spectrum_top_margin + spectrum_height + spectrum_bottom_margin + panel_start_offset
        
        # Add bass zoom panel height if active
        if self.show_bass_zoom:
            base_height += panel_height + panel_padding
        
        # Count active technical panels
        tech_panel_count = sum([
            self.show_meters,
            self.show_harmonic,
            self.show_pitch_detection,
            # self.show_chromagram,  # Disabled - part of integrated panel
            # self.show_genre_classification,  # Disabled - part of integrated panel
            self.show_room_analysis,
            self.show_integrated_music
        ])
        
        # Calculate rows needed for technical panels
        if tech_panel_count > 0:
            rows_needed = (tech_panel_count + max_columns - 1) // max_columns  # Ceiling division
            tech_panels_height = rows_needed * (panel_height + panel_padding) - panel_padding
            
            # Add extra height if integrated panel is shown
            if self.show_integrated_music:
                tech_panels_height += 50  # Extra 50px for integrated panel
            
            base_height += tech_panels_height + panel_padding
        
        # Add footer
        required_height = base_height + footer_height
        
        return required_height
    
    def adjust_window_for_panels(self, maintain_standard_size=False):
        """Adjust window height to accommodate active panels"""
        # Don't auto-adjust if in fullscreen, maintaining standard size, or preset is locked
        if self.fullscreen or maintain_standard_size or self.preset_locked:
            return
            
        required_height = self.calculate_required_height()
        # Ensure minimum height (must show at least the spectrum)
        min_height = 1150  # Minimum to show header + spectrum + footer
        required_height = max(required_height, min_height)
        
        # Only resize if height actually needs to change
        if abs(required_height - self.height) > 5:  # 5px tolerance
            self.resize_window(self.width, required_height)
    
    def run(self):
        """Main visualization loop"""
        # Start audio capture
        if not self.capture.start_capture():
            print("Failed to start audio capture")
            return
        
        # Start audio capture thread
        audio_thread = threading.Thread(target=self.capture_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
        # Wait for buffer to fill
        print("Waiting for audio buffer to fill...")
        while self.buffer_pos < FFT_SIZE_BASE and self.running:
            time.sleep(0.1)
        
        print("Starting visualization...")
        
        while self.running:
            frame_start = time.time()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keyboard(event)
                elif event.type == pygame.VIDEORESIZE:
                    self.resize_window(event.w, event.h)
                    # User manually resized, so unlock any preset
                    self.preset_locked = False
            
            # Process audio
            process_start = time.time()
            spectrum_data = self.process_audio_spectrum()
            process_time = (time.time() - process_start) * 1000
            self.process_times.append(process_time)
            
            if spectrum_data:
                # Auto-test panels if enabled
                if self.test_mode:
                    self.test_all_panels()
                
                # Clear screen
                self.screen.fill(BACKGROUND_COLOR)
                
                # Draw comprehensive header
                header_height = self.draw_header_panel(spectrum_data)
                
                # Draw main spectrum area with border
                # Calculate margins based on window size
                side_margin = max(50, int(self.width * 0.025))  # 2.5% margin, min 50px
                top_margin = 10
                bottom_margin = 100  # Leave space for frequency scale (footer is separate)
                
                vis_start_x = side_margin
                vis_start_y = header_height + top_margin  # Use returned header height
                
                # Calculate vis_width - leave room for VU meter if enabled
                vu_meter_space = 170 if self.show_vu_meters else 0  # 150px width + 20px margins
                vis_width = self.width - (2 * side_margin) - vu_meter_space
                
                # Use fixed spectrum height instead of remaining space
                vis_height = 800  # Doubled from 400px for better visibility
                
                vis_params = {
                    'vis_start_x': vis_start_x,
                    'vis_start_y': vis_start_y,
                    'vis_width': vis_width,
                    'vis_height': vis_height,
                    'center_y': vis_start_y + vis_height // 2,  # Center within the vis area
                    'max_bar_height': vis_height // 2 - 10,  # Leave small margin
                    # Additional params for grid alignment
                    'spectrum_left': vis_start_x,
                    'spectrum_right': vis_start_x + vis_width,
                    'spectrum_top': vis_start_y,
                    'spectrum_bottom': vis_start_y + vis_height,
                    'scale_y': vis_start_y + vis_height + 10  # Position for frequency scale
                }
                
                # Draw border around spectrum area
                pygame.draw.rect(self.screen, GRID_COLOR, 
                               (vis_params['vis_start_x'] - 2, 
                                vis_params['vis_start_y'] - 2,
                                vis_params['vis_width'] + 4,
                                vis_params['vis_height'] + 4), 2)
                
                # Draw spectrum bars
                self.display.draw_spectrum_bars(spectrum_data['band_values'], vis_params)
                
                # Draw grid and labels
                if self.show_grid:
                    self.display.draw_grid_and_labels(vis_params)
                    
                    # Create frequency-to-x mapping function based on band indices
                    def freq_to_x(freq):
                        # Find which band contains this frequency
                        freq_bin = freq * FFT_SIZE_BASE / SAMPLE_RATE
                        for i, (start_idx, end_idx) in enumerate(self.band_indices):
                            if start_idx <= freq_bin < end_idx:
                                # Map to bar position
                                bar_x = vis_start_x + (i / len(self.band_indices)) * vis_width
                                return int(bar_x)
                        # Fallback for frequencies outside range
                        if freq <= 20:
                            return vis_start_x
                        elif freq >= 20000:
                            return vis_start_x + vis_width
                        else:
                            # Log scale approximation
                            import math
                            log_pos = (math.log10(freq) - math.log10(20)) / (math.log10(20000) - math.log10(20))
                            return int(vis_start_x + log_pos * vis_width)
                    
                    self.display.draw_frequency_scale(vis_params, freq_to_x_func=freq_to_x, ui_scale=1.0)
                
                # Stats are now displayed in the header panel
                
                # Panel layout system with grid positioning
                panel_padding = 10
                panel_height = 250  # Adjusted to fit better with footer gap
                max_columns = 4  # Limit to 4 panels per row
                
                # VU meters on right side of spectrum with full spectrum height
                if self.show_vu_meters:
                    vu_meter_x = vis_params['spectrum_right'] + 10  # 10px gap from spectrum
                    vu_meter_y = vis_params['spectrum_top']
                    vu_meter_width = 150  # Narrower width
                    vu_meter_height = vis_params['vis_height']  # Match spectrum height
                    
                    self.vu_meters_panel.draw(self.screen, vu_meter_x, vu_meter_y, 
                                            vu_meter_width, vu_meter_height)
                    
                    # Draw frozen indicator for VU meters if frozen
                    if self.frozen_vu_meters:
                        frozen_overlay = pygame.Surface((vu_meter_width, vu_meter_height))
                        frozen_overlay.set_alpha(128)
                        frozen_overlay.fill((0, 0, 0))
                        self.screen.blit(frozen_overlay, (vu_meter_x, vu_meter_y))
                        
                        # Rotate text 90 degrees for vertical VU meter
                        frozen_text = self.font_medium.render("FROZEN", True, (255, 100, 100))
                        rotated_text = pygame.transform.rotate(frozen_text, 90)
                        text_rect = rotated_text.get_rect(center=(vu_meter_x + vu_meter_width // 2, vu_meter_y + vu_meter_height // 2))
                        self.screen.blit(rotated_text, text_rect)
                
                # Bass zoom panel directly below spectrum (full width)
                bass_zoom_y = header_height + 10 + 800 + 10  # 10px below spectrum
                if self.show_bass_zoom:
                    bass_zoom_x = vis_start_x
                    bass_zoom_width = vis_width
                    self.bass_zoom_panel.draw(self.screen, bass_zoom_x, bass_zoom_y, bass_zoom_width, panel_height)
                    
                    # Draw frozen indicator for bass zoom if frozen
                    if self.frozen_bass_zoom:
                        frozen_overlay = pygame.Surface((bass_zoom_width, panel_height))
                        frozen_overlay.set_alpha(128)
                        frozen_overlay.fill((0, 0, 0))
                        self.screen.blit(frozen_overlay, (bass_zoom_x, bass_zoom_y))
                        
                        frozen_text = self.font_medium.render("FROZEN", True, (255, 100, 100))
                        text_rect = frozen_text.get_rect(center=(bass_zoom_x + bass_zoom_width // 2, bass_zoom_y + panel_height // 2))
                        self.screen.blit(frozen_text, text_rect)
                
                # Technical panels grid below bass zoom
                tech_panels_start_y = bass_zoom_y + (panel_height + panel_padding if self.show_bass_zoom else 0)
                
                # Debug: Print panel positions
                if not hasattr(self, '_positions_logged'):
                    print(f"Window dimensions: {self.width}x{self.height}")
                    print(f"Spectrum left edge: {vis_start_x}")
                    print(f"Spectrum width: {vis_width}")
                    print(f"Bass zoom Y: {bass_zoom_y}")
                    print(f"Tech panels start Y: {tech_panels_start_y}")
                    print(f"Footer starts at: {self.height - 80}")
                    self._positions_logged = True
                
                # Collect all active technical panels
                active_panels = []
                if self.show_meters:
                    active_panels.append(('meters', self.professional_meters_panel))
                if self.show_harmonic:
                    active_panels.append(('harmonic', self.harmonic_analysis_panel))
                if self.show_pitch_detection:
                    active_panels.append(('pitch', self.pitch_detection_panel))
                if self.show_chromagram and not self.show_integrated_music:  # Don't show if integrated is on
                    active_panels.append(('chromagram', self.chromagram_panel))
                if self.show_genre_classification and not self.show_integrated_music:  # Don't show if integrated is on
                    active_panels.append(('genre', self.genre_classification_panel))
                if self.show_room_analysis:
                    active_panels.append(('room', None))  # Special case - drawn manually
                if self.show_integrated_music:
                    active_panels.append(('integrated', self.integrated_music_panel))
                
                # Calculate grid positions
                if active_panels:
                    # Debug: Print active panels
                    if not hasattr(self, '_panels_logged'):
                        print(f"Active technical panels: {[p[0] for p in active_panels]}")
                        self._panels_logged = True
                    
                    # Align grid with main visualizer (use same left margin as spectrum)
                    grid_start_x = vis_start_x
                    
                    # Calculate panel width to fit within spectrum width
                    # Account for padding between panels
                    panels_per_row = min(len(active_panels), max_columns)
                    total_padding = (panels_per_row - 1) * panel_padding
                    panel_width = (vis_width - total_padding) // panels_per_row
                    
                    # Debug panel dimensions
                    if not hasattr(self, '_panel_dims_logged'):
                        print(f"Panels per row: {panels_per_row}")
                        print(f"Panel width: {panel_width}")
                        print(f"Total rows needed: {(len(active_panels) + max_columns - 1) // max_columns}")
                        self._panel_dims_logged = True
                    
                    # Draw panels in grid layout
                    for i, (panel_type, panel) in enumerate(active_panels):
                        row = i // max_columns
                        col = i % max_columns
                        
                        panel_x = grid_start_x + col * (panel_width + panel_padding)
                        panel_y = tech_panels_start_y + row * (panel_height + panel_padding)
                        
                        # Calculate UI scale based on panel type and window size
                        ui_scale = self.width / 2000.0  # Scale based on window width
                        
                        # Special handling for integrated panel
                        actual_panel_height = panel_height
                        actual_panel_width = panel_width
                        if panel_type == 'integrated':
                            # Make integrated panel 2 columns wide
                            # Place it in column 2-3 (under Pitch Detection and Room Analysis)
                            col = 2
                            panel_x = grid_start_x + col * (panel_width + panel_padding)
                            actual_panel_width = (panel_width * 2) + panel_padding  # Span 2 columns
                            # Increase height for integrated panel to fit all content including hip-hop features
                            # Base height + genre/harmony sections + hip-hop + cross-analysis + chromagram + confidence graph
                            actual_panel_height = panel_height + 120  # 380px total to fit all content
                        
                        # Handle room panel specially
                        if panel_type == 'room':
                            # Draw room analysis panel background
                            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, actual_panel_height)
                            pygame.draw.rect(self.screen, (20, 25, 35), panel_rect)
                            pygame.draw.rect(self.screen, (60, 70, 90), panel_rect, 2)
                            
                            # Draw room modes content
                            y_pos = panel_y + 10
                            room_text = self.font_small.render("Room Modes:", True, TEXT_COLOR)
                            text_rect = room_text.get_rect(centerx=panel_x + panel_width // 2, y=y_pos)
                            self.screen.blit(room_text, text_rect)
                            y_pos += 35
                            
                            for j, mode in enumerate(self.room_modes[:8]):  # Show top 8 modes to fit in 260px
                                # Stop if we're running out of space within panel
                                if y_pos + 20 > panel_y + panel_height - 10:
                                    break
                                    
                                # Handle different possible keys for magnitude/strength
                                magnitude_key = 'magnitude' if 'magnitude' in mode else 'strength'
                                magnitude_val = mode.get(magnitude_key, 0.0)
                                
                                # Format mode info
                                mode_info = f"{mode.get('frequency', 0):.1f}Hz - {mode.get('type', 'Unknown')}"
                                if magnitude_val > 0:
                                    mode_info += f" ({magnitude_val:.1f}dB)"
                                
                                mode_text = self.font_tiny.render(mode_info, True, (200, 200, 100))
                                self.screen.blit(mode_text, (panel_x + 20, y_pos))
                                y_pos += 25
                        else:
                            # Regular panel
                            panel.draw(self.screen, panel_x, panel_y, actual_panel_width, actual_panel_height, ui_scale)
                        
                        # Draw frozen indicator if panel is frozen
                        frozen_state = False
                        if panel_type == 'meters' and self.frozen_meters:
                            frozen_state = True
                        elif panel_type == 'harmonic' and self.frozen_harmonic:
                            frozen_state = True
                        elif panel_type == 'pitch' and self.frozen_pitch_detection:
                            frozen_state = True
                        elif panel_type == 'chromagram' and self.frozen_chromagram:
                            frozen_state = True
                        elif panel_type == 'genre' and self.frozen_genre_classification:
                            frozen_state = True
                        elif panel_type == 'integrated' and self.frozen_integrated_music:
                            frozen_state = True
                        elif panel_type == 'room' and self.frozen_room_analysis:
                            frozen_state = True
                        
                        if frozen_state:
                            # Draw semi-transparent overlay
                            frozen_overlay = pygame.Surface((actual_panel_width, actual_panel_height))
                            frozen_overlay.set_alpha(128)
                            frozen_overlay.fill((0, 0, 0))
                            self.screen.blit(frozen_overlay, (panel_x, panel_y))
                            
                            # Draw "FROZEN" text
                            frozen_text = self.font_medium.render("FROZEN", True, (255, 100, 100))
                            text_rect = frozen_text.get_rect(center=(panel_x + actual_panel_width // 2, panel_y + actual_panel_height // 2))
                            self.screen.blit(frozen_text, text_rect)
                
                # Add bottom margin check to ensure panels don't overlap footer
                if active_panels:
                    # Calculate bottom of last row
                    num_rows = (len(active_panels) + max_columns - 1) // max_columns
                    last_panel_idx = min(num_rows * max_columns - 1, len(active_panels) - 1)
                    if last_panel_idx >= 0:
                        last_row = last_panel_idx // max_columns
                        # Get the actual panel height (some panels like integrated are taller)
                        last_panel_type = active_panels[last_panel_idx][0]
                        last_panel_height = panel_height + 120 if last_panel_type == 'integrated' else panel_height
                        last_row_bottom = tech_panels_start_y + last_row * (panel_height + panel_padding) + last_panel_height
                        
                        # Check if we're too close to footer (footer_y = height - 80)
                        footer_y = self.height - 80
                        if last_row_bottom > footer_y - 10:
                            # Draw warning that panels are overlapping
                            warning_text = self.font_tiny.render("! Panels too close to footer", True, (255, 100, 100))
                            self.screen.blit(warning_text, (10, footer_y - 15))
                
                # Draw voice detection info
                if self.show_voice_detection and self.voice_active:
                    # Create a semi-transparent background for the voice detection
                    voice_bg = pygame.Surface((280, 40))
                    voice_bg.set_alpha(200)
                    voice_bg.fill((20, 30, 25))
                    
                    # Position in upper right, below header
                    voice_x = self.width - 290
                    voice_y = 130  # Below header
                    
                    self.screen.blit(voice_bg, (voice_x - 10, voice_y - 5))
                    
                    voice_text = self.font_medium.render(
                        f"Voice Detected ({self.voice_confidence:.0f}%)",
                        True, (100, 255, 100)
                    )
                    self.screen.blit(voice_text, (voice_x, voice_y))
                
                # Draw debug info
                self.draw_debug_info(spectrum_data)
                
                # FPS tracking for header display
                if self.show_fps:
                    frame_time = (time.time() - frame_start) * 1000
                    self.frame_times.append(frame_time)
                
                # Draw footer with help and copyright info
                self.draw_footer()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
            
            # Tick adaptive updater for next frame
            self.adaptive_updater.tick()
        
        # Cleanup
        self.capture.stop_capture()
        pygame.quit()
        print("Analyzer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Professional Live Audio Analyzer v4.1 OMEGA-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Audio source name (default: system monitor)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=BARS_DEFAULT,
        help=f"Number of frequency bars (default: {BARS_DEFAULT}, max: {BARS_MAX})",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH, help=f"Window width (default: {DEFAULT_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT, help=f"Window height (default: {DEFAULT_HEIGHT})"
    )

    args = parser.parse_args()

    # Validate bars
    if args.bars > BARS_MAX:
        print(f"Warning: Limiting bars to maximum of {BARS_MAX}")
        args.bars = BARS_MAX

    print("=" * 90)
    print("Professional Live Audio Analyzer v4.1 OMEGA-2 - Genre Classification Edition")
    print("=" * 90)
    print(f"Configuration:")
    print(f"  Audio source: {args.source or 'System monitor (auto-detect)'}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz | FFT base size: {FFT_SIZE_BASE}")
    print(f"  Frequency bars: {args.bars} | Window: {args.width}x{args.height}")
    print(f"  Max frequency: {MAX_FREQ} Hz | Target FPS: {TARGET_FPS}")
    print("\nNew in v4.1 OMEGA-2:")
    print("  🎵 Genre Classification: Real-time music genre detection")
    print("  🎹 Musical Key Detection: Chromagram analysis for key identification") 
    print("  🎼 Multi-resolution FFT: Enhanced bass detail (20-250Hz)")
    print("  📊 Professional Metering: LUFS, K-weighting, True Peak")
    print("  🎸 Harmonic Analysis: Instrument identification")
    print("  🔊 Phase Coherence: Stereo imaging analysis")
    print("  💥 Transient Detection: Percussion tracking")
    print("  🏠 Room Mode Detection: Acoustic analysis")
    print("  🎯 Adaptive Frequency Allocation: Dynamic bass detail (60-80% based on content)")
    print("  🎨 Professional UI: Studio-grade visualization with multiple panels")
    print("Professional Controls:")
    print("  M: Freeze/Unfreeze Professional meters | U: Freeze/Unfreeze VU meters")
    print("  H: Freeze/Unfreeze Harmonic analysis | R: Freeze/Unfreeze Room analysis")
    print("  Z: Freeze/Unfreeze Bass zoom | P: Freeze/Unfreeze Pitch detection")
    print("  C: Freeze/Unfreeze Chromagram | J: Freeze/Unfreeze Genre classification")
    print("  I: Toggle Integrated Music Analysis (combines Chromagram + Genre)")
    print("  G: Toggle grid | S: Screenshot | V/F/A: Voice/Formant/Advanced")
    print("  K: Toggle adaptive frequency allocation | D: Debug snapshot (terminal)")
    print("  T: Test mode (auto-freeze panels) | SPACE: Toggle A/B mode | ↑/↓: Adjust gain")
    print("  1-9: Window size presets (1:Minimum, 2-4:With Bass, 5-7:With Panels, 8-9:Full)")
    print("  F11: Fullscreen | ESC: Exit")
    print("=" * 90)

    analyzer = ProfessionalLiveAudioAnalyzer(
        width=args.width, height=args.height, bars=args.bars, source_name=args.source
    )

    analyzer.run()


if __name__ == "__main__":
    main()