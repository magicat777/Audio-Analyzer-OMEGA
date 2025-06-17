"""
Integrated Music Analysis Panel - Combined Chromagram + Genre Visualization
OMEGA-3 Feature: Unified music analysis display
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any
from integrated_music_analyzer import IntegratedMusicAnalyzer, MusicAnalysisConfig

class IntegratedMusicPanel:
    """Combined visualization for genre classification and chromagram analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Initialize integrated analyzer
        config = MusicAnalysisConfig(sample_rate=sample_rate)
        self.analyzer = IntegratedMusicAnalyzer(config)
        
        # Display state
        self.analysis_results = {
            'genre': {'top_genre': 'Unknown', 'confidence': 0.0, 'probabilities': {}},
            'harmony': {'key': 'C Major', 'chromagram': np.zeros(12)},
            'cross_analysis': {'overall_confidence': 0.0, 'harmonic_genre_consistency': 0.0}
        }
        
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
    
    def update(self, fft_data: np.ndarray, audio_chunk: np.ndarray, freqs: np.ndarray,
               drum_info: Dict, harmonic_info: Dict):
        """Update integrated music analysis"""
        if fft_data is not None and len(fft_data) > 0:
            self.analysis_results = self.analyzer.analyze_music(
                fft_data, audio_chunk, freqs, drum_info, harmonic_info
            )
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, 
             ui_scale: float = 1.0):
        """OMEGA-3: Draw integrated music analysis visualization"""
        
        # Semi-transparent background with gradient
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(230)
        
        # Create gradient background
        for i in range(height):
            color_intensity = 50 + int(20 * (i / height))
            color = (color_intensity, int(color_intensity * 0.8), int(color_intensity * 1.2))
            pygame.draw.line(overlay, color, (0, i), (width, i))
        
        screen.blit(overlay, (x, y))
        
        # Border with rounded corners effect
        pygame.draw.rect(screen, (120, 100, 160), (x, y, width, height), 3)
        
        # Split panel into two sections
        left_width = width // 2
        right_width = width - left_width
        
        # Left side: Genre Classification
        self._draw_genre_section(screen, x, y, left_width, height, ui_scale)
        
        # Right side: Harmonic Analysis
        self._draw_harmonic_section(screen, x + left_width, y, right_width, height, ui_scale)
        
        # Bottom: Cross-analysis insights
        cross_analysis_height = int(60 * ui_scale)
        self._draw_cross_analysis(screen, x, y + height - cross_analysis_height, 
                                width, cross_analysis_height, ui_scale)
    
    def _draw_genre_section(self, screen: pygame.Surface, x: int, y: int, 
                          width: int, height: int, ui_scale: float):
        """Draw genre classification section"""
        y_offset = y + int(10 * ui_scale)
        
        # Genre section title
        if self.font_small:
            title_surf = self.font_small.render("GENRE ANALYSIS", True, (255, 200, 150))
            screen.blit(title_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        genre_info = self.analysis_results.get('genre', {})
        top_genre = genre_info.get('top_genre', 'Unknown')
        confidence = genre_info.get('confidence', 0.0)
        
        # Main genre display
        if self.font_medium:
            genre_color = self._get_confidence_color(confidence)
            genre_surf = self.font_medium.render(top_genre, True, genre_color)
            screen.blit(genre_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(30 * ui_scale)
        
        # Confidence bar
        if self.font_tiny:
            conf_text = f"{confidence:.0%}"
            conf_surf = self.font_tiny.render(conf_text, True, (200, 200, 200))
            screen.blit(conf_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(15 * ui_scale)
        
        # Confidence bar visualization
        bar_width = width - int(20 * ui_scale)
        bar_height = int(8 * ui_scale)
        pygame.draw.rect(screen, (40, 40, 50), 
                        (x + int(10 * ui_scale), y_offset, bar_width, bar_height))
        
        conf_fill_width = int(bar_width * confidence)
        conf_color = self._get_confidence_color(confidence)
        pygame.draw.rect(screen, conf_color,
                        (x + int(10 * ui_scale), y_offset, conf_fill_width, bar_height))
        pygame.draw.rect(screen, (100, 100, 120),
                        (x + int(10 * ui_scale), y_offset, bar_width, bar_height), 1)
        
        y_offset += bar_height + int(15 * ui_scale)
        
        # Top 3 genres
        top_3 = genre_info.get('top_3', [])
        if self.font_tiny and top_3:
            for i, (genre, prob) in enumerate(top_3[:3]):
                if i == 0:  # Skip main genre, already shown
                    continue
                
                genre_text = f"{genre}: {prob:.0%}"
                color = (150, 150, 150) if i > 0 else (200, 200, 200)
                genre_surf = self.font_tiny.render(genre_text, True, color)
                screen.blit(genre_surf, (x + int(10 * ui_scale), y_offset))
                y_offset += int(12 * ui_scale)
    
    def _draw_harmonic_section(self, screen: pygame.Surface, x: int, y: int,
                             width: int, height: int, ui_scale: float):
        """Draw harmonic analysis section"""
        y_offset = y + int(10 * ui_scale)
        
        # Harmonic section title
        if self.font_small:
            title_surf = self.font_small.render("HARMONIC ANALYSIS", True, (150, 200, 255))
            screen.blit(title_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(25 * ui_scale)
        
        harmony_info = self.analysis_results.get('harmony', {})
        detected_key = harmony_info.get('key', 'C Major')
        chromagram = harmony_info.get('chromagram', np.zeros(12))
        
        # Key display
        if self.font_medium:
            key_surf = self.font_medium.render(detected_key, True, (150, 255, 200))
            screen.blit(key_surf, (x + int(10 * ui_scale), y_offset))
            y_offset += int(35 * ui_scale)
        
        # Mini chromagram visualization
        chroma_width = width - int(20 * ui_scale)
        chroma_height = int(40 * ui_scale)
        chroma_x = x + int(10 * ui_scale)
        
        # Background
        pygame.draw.rect(screen, (20, 20, 30), 
                        (chroma_x, y_offset, chroma_width, chroma_height))
        
        # Draw chromagram bars
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        bar_width = chroma_width // 12
        
        for i, (note, value) in enumerate(zip(note_names, chromagram)):
            bar_x = chroma_x + i * bar_width
            bar_height = int(chroma_height * min(value, 1.0))
            
            # Color based on prominence
            if value > 0.1:
                color = (100 + int(155 * value), 150, 100 + int(100 * value))
            else:
                color = (60, 60, 80)
            
            if bar_height > 0:
                pygame.draw.rect(screen, color,
                               (bar_x + 1, y_offset + chroma_height - bar_height,
                                bar_width - 2, bar_height))
        
        # Border around chromagram
        pygame.draw.rect(screen, (100, 100, 120),
                        (chroma_x, y_offset, chroma_width, chroma_height), 1)
    
    def _draw_cross_analysis(self, screen: pygame.Surface, x: int, y: int,
                           width: int, height: int, ui_scale: float):
        """Draw cross-analysis insights at bottom"""
        
        # Semi-transparent overlay for cross-analysis section
        cross_overlay = pygame.Surface((width, height))
        cross_overlay.set_alpha(180)
        cross_overlay.fill((40, 30, 50))
        screen.blit(cross_overlay, (x, y))
        
        # Border line at top
        pygame.draw.line(screen, (120, 100, 160), (x, y), (x + width, y), 2)
        
        y_offset = y + int(10 * ui_scale)
        
        cross_info = self.analysis_results.get('cross_analysis', {})
        overall_confidence = cross_info.get('overall_confidence', 0.0)
        consistency = cross_info.get('harmonic_genre_consistency', 0.0)
        
        if self.font_tiny:
            # Overall analysis confidence
            conf_text = f"Analysis Confidence: {overall_confidence:.0%}"
            conf_color = self._get_confidence_color(overall_confidence)
            conf_surf = self.font_tiny.render(conf_text, True, conf_color)
            screen.blit(conf_surf, (x + int(10 * ui_scale), y_offset))
            
            # Genre-harmony consistency
            consistency_text = f"Genre-Harmony Match: {consistency:.0%}"
            consistency_color = self._get_confidence_color(consistency)
            consistency_surf = self.font_tiny.render(consistency_text, True, consistency_color)
            consistency_rect = consistency_surf.get_rect(right=x + width - int(10 * ui_scale), y=y_offset)
            screen.blit(consistency_surf, consistency_rect)
            
            y_offset += int(15 * ui_scale)
            
            # Analysis insight
            insight = self._generate_analysis_insight()
            if insight:
                insight_surf = self.font_tiny.render(insight, True, (180, 180, 200))
                screen.blit(insight_surf, (x + int(10 * ui_scale), y_offset))
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level"""
        if confidence < 0.3:
            return (255, 150, 150)  # Red - low confidence
        elif confidence < 0.6:
            return (255, 255, 150)  # Yellow - medium confidence
        else:
            return (150, 255, 150)  # Green - high confidence
    
    def _generate_analysis_insight(self) -> str:
        """Generate contextual insight about the analysis"""
        genre_info = self.analysis_results.get('genre', {})
        harmony_info = self.analysis_results.get('harmony', {})
        cross_info = self.analysis_results.get('cross_analysis', {})
        
        top_genre = genre_info.get('top_genre', 'Unknown')
        detected_key = harmony_info.get('key', 'C Major')
        consistency = cross_info.get('harmonic_genre_consistency', 0.0)
        
        if consistency > 0.7:
            return f"{top_genre} characteristics strongly match {detected_key} harmonic content"
        elif consistency < 0.3:
            return f"Genre and harmonic analysis show conflicting characteristics"
        else:
            return f"Moderate confidence in {top_genre} classification with {detected_key}"
    
    def get_results(self) -> Dict[str, Any]:
        """Get current integrated analysis results"""
        return self.analysis_results.copy()
    
    def get_temporal_analysis(self) -> Dict[str, Any]:
        """Get temporal analysis from the integrated analyzer"""
        return self.analyzer.get_temporal_analysis()