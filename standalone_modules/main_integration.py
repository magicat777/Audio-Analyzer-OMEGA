# Add this to your main analyzer file (wherever you handle audio processing)

class EnhancedAudioAnalyzer:
    """Enhanced analyzer with integrated music analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Keep existing analyzers for backward compatibility
        from chromagram import ChromagramPanel
        from genre_classification import GenreClassificationPanel
        from pitch_detection import PitchDetectionPanel
        
        # Add new integrated analyzer
        from integrated_music_panel import IntegratedMusicPanel
        
        self.chromagram_panel = ChromagramPanel(sample_rate)
        self.genre_panel = GenreClassificationPanel(sample_rate)
        self.pitch_panel = PitchDetectionPanel(sample_rate)
        
        # NEW: Integrated music analysis panel
        self.integrated_panel = IntegratedMusicPanel(sample_rate)
        
        # UI state for toggling between individual and integrated views
        self.show_integrated_view = True  # Set to True to use new integrated panel
    
    def update_analysis(self, fft_data: np.ndarray, audio_chunk: np.ndarray, 
                       freqs: np.ndarray, drum_info: Dict, harmonic_info: Dict):
        """Update all analysis modules"""
        
        if self.show_integrated_view:
            # Use integrated analysis (recommended)
            self.integrated_panel.update(fft_data, audio_chunk, freqs, drum_info, harmonic_info)
        else:
            # Use individual panels (backward compatibility)
            self.chromagram_panel.update(fft_data, freqs)
            self.genre_panel.update(fft_data, audio_chunk, drum_info, harmonic_info)
        
        # Always update pitch detection (independent)
        self.pitch_panel.update(audio_chunk)
    
    def draw_panels(self, screen: pygame.Surface, ui_scale: float = 1.0):
        """Draw analysis panels"""
        
        # Get screen dimensions for layout
        screen_width, screen_height = screen.get_size()
        
        if self.show_integrated_view:
            # Draw integrated panel (takes up more space)
            panel_width = int(400 * ui_scale)
            panel_height = int(300 * ui_scale)
            panel_x = screen_width - panel_width - int(20 * ui_scale)
            panel_y = int(100 * ui_scale)
            
            self.integrated_panel.draw(screen, panel_x, panel_y, panel_width, panel_height, ui_scale)
            
            # Draw pitch detection below
            pitch_y = panel_y + panel_height + int(10 * ui_scale)
            pitch_height = int(200 * ui_scale)
            self.pitch_panel.draw(screen, panel_x, pitch_y, panel_width, pitch_height, ui_scale)
            
        else:
            # Draw individual panels (original layout)
            panel_width = int(300 * ui_scale)
            panel_height = int(200 * ui_scale)
            panel_x = screen_width - panel_width - int(20 * ui_scale)
            
            # Chromagram panel
            chroma_y = int(100 * ui_scale)
            self.chromagram_panel.draw(screen, panel_x, chroma_y, panel_width, panel_height, ui_scale)
            
            # Genre panel
            genre_y = chroma_y + panel_height + int(10 * ui_scale)
            self.genre_panel.draw(screen, panel_x, genre_y, panel_width, panel_height, ui_scale)
            
            # Pitch panel
            pitch_y = genre_y + panel_height + int(10 * ui_scale)
            self.pitch_panel.draw(screen, panel_x, pitch_y, panel_width, panel_height, ui_scale)
    
    def toggle_view_mode(self):
        """Toggle between integrated and individual panel views"""
        self.show_integrated_view = not self.show_integrated_view
    
    def get_comprehensive_results(self) -> Dict[str, Any]:
        """Get results from all analysis modules"""
        if self.show_integrated_view:
            results = {
                'integrated_analysis': self.integrated_panel.get_results(),
                'temporal_analysis': self.integrated_panel.get_temporal_analysis(),
                'pitch_detection': self.pitch_panel.get_results()
            }
        else:
            results = {
                'chromagram': self.chromagram_panel.get_results(),
                'genre': self.genre_panel.get_results(),
                'pitch_detection': self.pitch_panel.get_results()
            }
        
        return results

# In your main audio processing loop, replace existing calls with:
def main_audio_loop():
    """Example integration into main audio loop"""
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAudioAnalyzer(sample_rate=48000)
    
    # Set fonts for all panels
    fonts = {
        'large': pygame.font.Font(None, 32),
        'medium': pygame.font.Font(None, 28),
        'small': pygame.font.Font(None, 24),
        'tiny': pygame.font.Font(None, 20)
    }
    
    analyzer.chromagram_panel.set_fonts(fonts)
    analyzer.genre_panel.set_fonts(fonts)
    analyzer.pitch_panel.set_fonts(fonts)
    analyzer.integrated_panel.set_fonts(fonts)
    
    # Main loop
    while running:
        # Get audio data (your existing code)
        audio_chunk, fft_data, freqs = get_audio_data()  # Your existing function
        
        # Get drum and harmonic info (your existing code)
        drum_info = get_drum_detection_info()  # Your existing function
        harmonic_info = get_harmonic_analysis_info()  # Your existing function
        
        # Update analysis
        analyzer.update_analysis(fft_data, audio_chunk, freqs, drum_info, harmonic_info)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:  # Tab key to toggle view
                    analyzer.toggle_view_mode()
        
        # Draw everything
        screen.fill((0, 0, 0))  # Clear screen
        
        # Draw your existing visualizations
        draw_spectrum_analyzer(screen)  # Your existing function
        
        # Draw enhanced music analysis panels
        analyzer.draw_panels(screen)
        
        pygame.display.flip()
    
    # Optional: Save performance data at end
    try:
        results = analyzer.get_comprehensive_results()
        if 'integrated_analysis' in results:
            temporal_data = results['temporal_analysis']
            print(f"Session Analysis Summary:")
            print(f"  Genre Stability: {temporal_data.get('genre_stability', 0):.0%}")
            print(f"  Key Stability: {temporal_data.get('key_stability', 0):.0%}")
            print(f"  Overall Confidence: {temporal_data.get('confidence_trend', 0):.0%}")
    except Exception as e:
        print(f"Could not generate session summary: {e}")

# Keyboard controls for your application:
# TAB - Toggle between integrated and individual panel views
# I - Toggle integrated analysis features on/off
# R - Reset analysis history (start fresh)
