"""
Optimized Main Module Patch for OMEGA-4
This file contains optimized versions of key methods from omega4_main.py
Copy these methods to replace the existing ones in omega4_main.py
"""

# Add these imports to the top of omega4_main.py:
# from omega4.optimization.gpu_accelerated_fft import get_gpu_fft_processor, GPUAcceleratedFFT
# from omega4.optimization.parallel_panel_updater import ParallelPanelUpdater, OptimizedDataSharing

def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, bars=BARS_DEFAULT, source_name=None):
    """Add these lines to the existing __init__ method after other initializations"""
    
    # ... existing init code ...
    
    # Initialize GPU FFT processor
    self.gpu_fft = get_gpu_fft_processor()
    print(f"GPU FFT initialized: {'GPU acceleration enabled' if self.gpu_fft.gpu_available else 'CPU mode'}")
    
    # Initialize parallel panel updater
    self.parallel_updater = ParallelPanelUpdater(max_workers=4)  # Use 4 threads
    self.shared_data = OptimizedDataSharing()
    
    # Register panels for parallel updates
    self._register_panels_for_parallel_update()
    
    # Pre-allocate frequency arrays to avoid recalculation
    self.frequency_arrays = {}
    for name, size in [('base', FFT_SIZE_BASE), ('bass', 2048), ('sub_bass', 4096)]:
        self.frequency_arrays[name] = np.fft.rfftfreq(size, 1/SAMPLE_RATE)


def _register_panels_for_parallel_update(self):
    """Register all panels with the parallel updater"""
    
    # Critical panels - update every frame
    if hasattr(self, 'bass_zoom_panel'):
        self.parallel_updater.register_panel(
            'bass_zoom',
            lambda data: self.bass_zoom_panel.update(
                data['fft_base']['magnitude'],
                data['bass_spectrum'],
                data['audio_chunk']
            ),
            group='independent'
        )
    
    # Medium priority panels - can update less frequently
    if hasattr(self, 'professional_meters_panel'):
        self.parallel_updater.register_panel(
            'professional_meters',
            lambda data: self.professional_meters_panel.update(data['audio_chunk']),
            group='independent'
        )
    
    if hasattr(self, 'vu_meters_panel'):
        self.parallel_updater.register_panel(
            'vu_meters',
            lambda data: self.vu_meters_panel.update(data['audio_chunk']),
            group='independent'
        )
    
    if hasattr(self, 'harmonic_panel'):
        self.parallel_updater.register_panel(
            'harmonic_analysis',
            lambda data: self.harmonic_panel.update(
                data['fft_base']['complex'],
                data['frequencies_base'],
                SAMPLE_RATE
            ),
            group='independent'
        )
    
    # Dependent panels - need harmonic data
    if hasattr(self, 'pitch_detection_panel'):
        self.parallel_updater.register_panel(
            'pitch_detection',
            lambda data: self._update_pitch_detection_optimized(data),
            group='sequential',
            dependencies=['harmonic_analysis']
        )
    
    if hasattr(self, 'chromagram_panel'):
        self.parallel_updater.register_panel(
            'chromagram',
            lambda data: self._update_chromagram_optimized(data),
            group='sequential',
            dependencies=['harmonic_analysis']
        )
    
    # Low priority panels
    if hasattr(self, 'genre_classification_panel'):
        self.parallel_updater.register_panel(
            'genre_classification',
            lambda data: self._update_genre_classification_optimized(data),
            group='low_priority',
            dependencies=['chromagram']
        )
    
    if hasattr(self, 'room_analyzer'):
        self.parallel_updater.register_panel(
            'room_analysis',
            lambda data: self._update_room_analysis_optimized(data),
            group='low_priority'
        )


def process_audio_spectrum_optimized(self):
    """Optimized audio processing using GPU FFT and shared data"""
    with self.lock:
        if self.buffer_pos < FFT_SIZE_BASE:
            return None
        
        # Get audio data
        audio_data = self.audio_buffer[:self.buffer_pos].copy()
        
    # Clear previous frame's shared data
    self.shared_data.clear()
    
    # Set base audio data
    self.shared_data.set_base_data(audio_data, SAMPLE_RATE)
    
    # Get the chunk we'll analyze
    chunk_size = min(FFT_SIZE_BASE, len(audio_data))
    audio_chunk = audio_data[-chunk_size:] if len(audio_data) >= chunk_size else audio_data
    
    # Apply gain
    audio_chunk = audio_chunk * self.input_gain
    
    # Store in shared data
    self.shared_data.shared_data['audio_chunk'] = audio_chunk
    
    # Compute all FFT resolutions at once using GPU
    resolutions = {
        'base': FFT_SIZE_BASE,
        'bass': 2048,
        'sub_bass': 4096
    }
    
    fft_results = self.gpu_fft.compute_multi_resolution_fft(
        audio_chunk,
        resolutions,
        window_type='hann'
    )
    
    # Store FFT results in shared data
    for name, result in fft_results.items():
        self.shared_data.shared_data[f'fft_{name}'] = result
        self.shared_data.shared_data[f'frequencies_{name}'] = result['freqs']
        self.shared_data.computed_flags.add(f'fft_{name}')
    
    # Process multi-resolution spectrum (optimized)
    combined_spectrum = self._process_multi_resolution_optimized(fft_results)
    
    # Update drum detection
    if self.adaptive_updater.should_update('drum_detection'):
        self.drum_detector.update(
            combined_spectrum['band_values'][:50],  # Low frequency bands
            combined_spectrum['frequencies'][:50]
        )
    
    # Store drum info
    drum_info = {
        'kick': self.drum_detector.get_kick_info(),
        'snare': self.drum_detector.get_snare_info()
    }
    self.shared_data.shared_data['drum_info'] = drum_info
    
    # Update harmonic info if needed
    if self.adaptive_updater.should_update('harmonic'):
        harmonic_info = self.harmonic_panel.get_harmonic_info()
        self.shared_data.shared_data['harmonic_info'] = harmonic_info
    else:
        self.shared_data.shared_data['harmonic_info'] = getattr(self, '_last_harmonic_info', {})
    
    # Update all panels in parallel
    update_futures = self.parallel_updater.update_all_panels(self.shared_data.shared_data)
    
    # Prepare spectrum data for display
    spectrum_data = {
        'band_values': combined_spectrum['band_values'],
        'frequencies': combined_spectrum['frequencies'],
        'bass_detail': combined_spectrum.get('bass_detail', {}),
        'fft_data': fft_results['base']['magnitude'],
        'phase_data': np.angle(fft_results['base']['complex']) if fft_results['base']['complex'] is not None else None,
        'drum_info': drum_info,
        'peak_freq': combined_spectrum.get('peak_frequency', 0),
        'rms': float(np.sqrt(np.mean(audio_chunk**2))),
        'audio_chunk': audio_chunk
    }
    
    # Store last harmonic info
    self._last_harmonic_info = self.shared_data.shared_data.get('harmonic_info', {})
    
    # Adjust update frequencies based on performance
    if hasattr(self, 'frame_counter'):
        self.frame_counter += 1
        if self.frame_counter % 60 == 0:  # Every second
            self.parallel_updater.adjust_update_frequencies()
            
            # Print performance stats if debug enabled
            if self.show_debug:
                stats = self.parallel_updater.get_performance_stats()
                print(f"Avg frame time: {stats['avg_frame_time']:.1f}ms")
    
    return spectrum_data


def _process_multi_resolution_optimized(self, fft_results):
    """Optimized multi-resolution processing"""
    # Pre-allocate arrays
    total_bars = self.bars
    band_values = np.zeros(total_bars)
    frequencies = np.zeros(total_bars)
    
    # Get pre-computed frequency arrays
    base_freqs = fft_results['base']['freqs']
    bass_freqs = fft_results['bass']['freqs']
    sub_bass_freqs = fft_results['sub_bass']['freqs']
    
    # Allocate bands to different frequency ranges
    bass_bars = int(total_bars * self.current_allocation)
    mid_bars = int(total_bars * 0.3)
    high_bars = total_bars - bass_bars - mid_bars
    
    # Process bass frequencies (20-250 Hz) with high resolution
    bass_end_freq = 250
    current_bar = 0
    
    # Sub-bass processing (20-80 Hz) with 4096 FFT
    if bass_bars > 10:
        sub_bass_data = fft_results['sub_bass']['magnitude']
        sub_bass_mask = (sub_bass_freqs >= 20) & (sub_bass_freqs <= 80)
        sub_bass_bins = np.where(sub_bass_mask)[0]
        
        if len(sub_bass_bins) > 0:
            # Distribute sub-bass bins across first portion of bass bars
            sub_bass_bar_count = min(bass_bars // 2, len(sub_bass_bins))
            for i in range(sub_bass_bar_count):
                bin_start = int(i * len(sub_bass_bins) / sub_bass_bar_count)
                bin_end = int((i + 1) * len(sub_bass_bins) / sub_bass_bar_count)
                
                if bin_end > bin_start:
                    bin_indices = sub_bass_bins[bin_start:bin_end]
                    band_values[current_bar] = np.max(sub_bass_data[bin_indices])
                    frequencies[current_bar] = np.mean(sub_bass_freqs[bin_indices])
                    current_bar += 1
    
    # Regular bass processing (80-250 Hz) with 2048 FFT
    bass_data = fft_results['bass']['magnitude']
    bass_mask = (bass_freqs >= 80) & (bass_freqs <= bass_end_freq)
    bass_bins = np.where(bass_mask)[0]
    
    if len(bass_bins) > 0:
        remaining_bass_bars = bass_bars - current_bar
        for i in range(remaining_bass_bars):
            bin_start = int(i * len(bass_bins) / remaining_bass_bars)
            bin_end = int((i + 1) * len(bass_bins) / remaining_bass_bars)
            
            if bin_end > bin_start:
                bin_indices = bass_bins[bin_start:bin_end]
                band_values[current_bar] = np.max(bass_data[bin_indices])
                frequencies[current_bar] = np.mean(bass_freqs[bin_indices])
                current_bar += 1
    
    # Mid and high frequencies with base FFT
    base_data = fft_results['base']['magnitude']
    
    # Process remaining frequencies
    remaining_bars = total_bars - current_bar
    if remaining_bars > 0:
        # Use logarithmic scaling for mid/high frequencies
        start_freq = bass_end_freq
        end_freq = min(self.max_freq, SAMPLE_RATE / 2)
        
        freq_scale = np.logspace(np.log10(start_freq), np.log10(end_freq), remaining_bars + 1)
        
        for i in range(remaining_bars):
            freq_low = freq_scale[i]
            freq_high = freq_scale[i + 1]
            
            bin_mask = (base_freqs >= freq_low) & (base_freqs < freq_high)
            bins = np.where(bin_mask)[0]
            
            if len(bins) > 0:
                band_values[current_bar] = np.max(base_data[bins])
                frequencies[current_bar] = (freq_low + freq_high) / 2
                current_bar += 1
    
    # Apply smoothing and gain
    band_values = self._apply_smoothing_and_gain(band_values)
    
    # Find peak frequency
    if np.max(band_values) > 0:
        peak_idx = np.argmax(band_values)
        peak_frequency = frequencies[peak_idx]
    else:
        peak_frequency = 0
    
    return {
        'band_values': band_values,
        'frequencies': frequencies,
        'peak_frequency': peak_frequency,
        'bass_detail': {
            'bass_bars': bass_bars,
            'bass_spectrum': band_values[:bass_bars],
            'bass_freqs': frequencies[:bass_bars]
        }
    }


def _update_pitch_detection_optimized(self, data):
    """Optimized pitch detection update"""
    if self.show_pitch_detection and hasattr(self, 'pitch_detection_panel'):
        harmonic_info = data.get('harmonic_info', {})
        harmonics = harmonic_info.get('harmonic_peaks', [])
        
        if harmonics:
            self.pitch_detection_panel.update(
                data['fft_base']['magnitude'],
                data['frequencies_base'],
                fundamental_freq=harmonics[0]['frequency'] if harmonics else None,
                harmonic_series=harmonic_info.get('harmonic_series', [])
            )


def _update_chromagram_optimized(self, data):
    """Optimized chromagram update"""
    if self.show_chromagram and hasattr(self, 'chromagram_panel'):
        self.chromagram_panel.update(
            data['fft_base']['magnitude'],
            data['frequencies_base']
        )


def _update_genre_classification_optimized(self, data):
    """Optimized genre classification update"""
    if self.show_genre_classification and hasattr(self, 'genre_classification_panel'):
        # Get chromagram data if available
        chromagram_data = None
        if hasattr(self, 'chromagram_panel'):
            chromagram_data = self.chromagram_panel.get_chromagram()
        
        self.genre_classification_panel.update(
            data['fft_base']['magnitude'],
            data['audio_chunk'],
            data['frequencies_base'],
            data.get('drum_info', {}),
            data.get('harmonic_info', {}),
            chromagram_data=chromagram_data
        )


def _update_room_analysis_optimized(self, data):
    """Optimized room analysis update"""
    if self.show_room_analysis and hasattr(self, 'room_analyzer'):
        self.room_modes = self.room_analyzer.analyze_room_modes(
            data['fft_base']['magnitude'],
            data['frequencies_base']
        )


# Replace the process_audio_spectrum method in omega4_main.py with process_audio_spectrum_optimized
# Add the other helper methods as well