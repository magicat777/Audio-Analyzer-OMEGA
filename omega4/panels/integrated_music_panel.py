"""
Integrated Music Analysis Panel for OMEGA-4 Audio Analyzer
Combines chromagram and genre classification for comprehensive music understanding
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

from ..panels.music_analysis_engine import MusicAnalysisEngine, MusicAnalysisConfig


class IntegratedMusicPanel:
    """Unified visualization panel showing genre classification and harmonic analysis"""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Initialize the unified analysis engine
        self.analyzer = MusicAnalysisEngine(sample_rate)
        
        # Display configuration
        self.display_mode = 'integrated'  # 'integrated', 'genre_focus', 'harmony_focus'
        self.show_confidence_correlation = True
        self.show_temporal_analysis = True
        
        # Visualization history
        self.confidence_history = deque(maxlen=120)  # 2 seconds at 60 FPS
        self.genre_history = deque(maxlen=60)
        self.key_history = deque(maxlen=60)
        
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
        
        # Pass fonts to sub-panels
        self.analyzer.chromagram.set_fonts(fonts)
        self.analyzer.genre_classifier.set_fonts(fonts)
    
    def update(self, fft_data: np.ndarray, audio_data: np.ndarray, 
               frequencies: np.ndarray, drum_info: Dict[str, Any], 
               harmonic_info: Dict[str, Any]):
        """Update integrated music analysis"""
        # Perform integrated analysis
        results = self.analyzer.analyze_music(
            fft_data, audio_data, frequencies, drum_info, harmonic_info
        )
        
        # Update history
        self.confidence_history.append(results['cross_analysis']['overall_confidence'])
        self.genre_history.append(results['genre']['top_genre'])
        self.key_history.append(results['harmony']['key'])
    
    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, ui_scale: float = 1.0):
        """Draw integrated music analysis visualization"""
        # Semi-transparent background
        overlay = pygame.Surface((width, height))
        overlay.set_alpha(240)
        overlay.fill((20, 20, 30))
        screen.blit(overlay, (x, y))
        
        # Border with genre-based color
        top_genre = 'Unknown'
        if 'genre' in self.analyzer.current_analysis and 'top_genre' in self.analyzer.current_analysis['genre']:
            top_genre = self.analyzer.current_analysis['genre']['top_genre']
        border_color = self._get_genre_color(top_genre)
        pygame.draw.rect(screen, border_color, (x, y, width, height), 3)
        
        # Layout based on display mode
        if self.display_mode == 'integrated':
            self._draw_integrated_view(screen, x, y, width, height, ui_scale)
        elif self.display_mode == 'genre_focus':
            self._draw_genre_focus_view(screen, x, y, width, height, ui_scale)
        elif self.display_mode == 'harmony_focus':
            self._draw_harmony_focus_view(screen, x, y, width, height, ui_scale)
    
    def _draw_integrated_view(self, screen: pygame.Surface, x: int, y: int, 
                             width: int, height: int, ui_scale: float):
        """Draw integrated view with both genre and harmonic info"""
        padding = int(15 * ui_scale)
        y_offset = y + padding
        
        # Title
        if self.font_large:
            title = "OMEGA Integrated Music Analysis"
            title_surf = self.font_large.render(title, True, (220, 220, 255))
            title_rect = title_surf.get_rect(centerx=x + width // 2, top=y_offset)
            screen.blit(title_surf, title_rect)
            y_offset += int(40 * ui_scale)
        
        # Main analysis results
        analysis = self.analyzer.current_analysis
        
        # Genre and Key side by side
        half_width = (width - 3 * padding) // 2
        
        # Genre section
        if 'genre' in analysis:
            self._draw_genre_section(screen, x + padding, y_offset, half_width, int(100 * ui_scale), analysis['genre'])
        
        # Harmony section
        if 'harmony' in analysis:
            self._draw_harmony_section(screen, x + half_width + 2 * padding, y_offset, half_width, int(100 * ui_scale), analysis['harmony'])
        
        y_offset += int(110 * ui_scale)
        
        # Hip-hop specific visualization if detected with high confidence
        if 'hip_hop' in analysis and analysis['hip_hop'].get('confidence', 0) > 0.6:
            self._draw_hip_hop_analysis(screen, x + padding, y_offset, width - 2 * padding, int(80 * ui_scale), analysis['hip_hop'])
            y_offset += int(90 * ui_scale)
        
        # Cross-analysis section
        if 'cross_analysis' in analysis:
            self._draw_cross_analysis(screen, x + padding, y_offset, width - 2 * padding, int(80 * ui_scale), analysis['cross_analysis'])
        
        y_offset += int(90 * ui_scale)
        
        # Chromagram visualization
        chroma_height = int(50 * ui_scale)  # Reduced height for better proportion
        if 'harmony' in analysis and 'chromagram' in analysis['harmony']:
            self._draw_chromagram_bar(screen, x + padding, y_offset, width - 2 * padding, chroma_height, analysis['harmony']['chromagram'])
        
        y_offset += chroma_height + int(30 * ui_scale)  # Extra space for labels and border
        
        # Confidence correlation graph
        if self.show_confidence_correlation and y_offset < y + height - int(80 * ui_scale):
            self._draw_confidence_graph(screen, x + padding, y_offset, width - 2 * padding, int(60 * ui_scale))
    
    def _draw_genre_section(self, screen: pygame.Surface, x: int, y: int, 
                           width: int, height: int, genre_data: Dict):
        """Draw genre classification section"""
        # Background
        genre_bg = pygame.Surface((width, height))
        genre_bg.set_alpha(180)
        genre_bg.fill((30, 25, 40))
        screen.blit(genre_bg, (x, y))
        
        # Border
        pygame.draw.rect(screen, (100, 90, 120), (x, y, width, height), 1)
        
        padding = 5
        y_offset = y + padding
        
        # Genre name with confidence
        if self.font_medium:
            genre = genre_data.get('top_genre', 'Unknown')
            confidence = genre_data.get('confidence', 0.0)
            color = self._get_genre_color(genre)
            
            genre_text = f"{genre}"
            
            # Add subgenre for hip-hop
            if genre == 'Hip-Hop':
                hip_hop_info = self.analyzer.current_analysis.get('hip_hop', {})
                if hip_hop_info and hip_hop_info.get('subgenre') and hip_hop_info['subgenre'] != 'unknown':
                    genre_text = f"{genre} ({hip_hop_info['subgenre']})"
            
            genre_surf = self.font_medium.render(genre_text, True, color)
            screen.blit(genre_surf, (x + padding, y_offset))
            y_offset += 25
            
            # Confidence bar
            if self.font_small:
                conf_text = f"Confidence: {confidence:.0%}"
                conf_surf = self.font_small.render(conf_text, True, (180, 180, 200))
                screen.blit(conf_surf, (x + padding, y_offset))
                y_offset += 20
                
                # Draw confidence bar
                bar_width = width - 2 * padding
                bar_height = 8
                bar_x = x + padding
                bar_y = y_offset
                
                # Background
                pygame.draw.rect(screen, (40, 40, 50), (bar_x, bar_y, bar_width, bar_height))
                # Fill
                fill_width = int(bar_width * confidence)
                pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
                # Border
                pygame.draw.rect(screen, (100, 100, 120), (bar_x, bar_y, bar_width, bar_height), 1)
                y_offset += bar_height + 5
                
            # Show hip-hop specific features if detected
            if genre == 'Hip-Hop' and self.font_tiny:
                # Get hip-hop analysis from the integrated results
                hip_hop_info = self.analyzer.current_analysis.get('hip_hop', {})
                if hip_hop_info and hip_hop_info.get('is_hip_hop'):
                    features = hip_hop_info.get('features', {})
                    
                    # Sub-bass presence (808s)
                    sub_bass = features.get('sub_bass_presence', 0)
                    if sub_bass > 0.3:
                        color = (255, 100, 100) if sub_bass > 0.7 else (255, 180, 100)
                        sub_bass_text = f"808/Sub-bass: {sub_bass:.0%}"
                        sub_bass_surf = self.font_tiny.render(sub_bass_text, True, color)
                        screen.blit(sub_bass_surf, (x + padding, y_offset))
                        y_offset += 12
                    
                    # Hi-hat density (important for trap)
                    hihat = features.get('hihat_density', 0)
                    if hihat > 0.3:
                        color = (150, 255, 150) if hihat > 0.7 else (200, 255, 150)
                        hihat_text = f"Hi-hat density: {hihat:.0%}"
                        hihat_surf = self.font_tiny.render(hihat_text, True, color)
                        screen.blit(hihat_surf, (x + padding, y_offset))
                        y_offset += 12
                    
                    # Beat strength
                    kick = features.get('kick_pattern_score', 0)
                    if kick > 0.5:
                        color = (150, 150, 255) if kick > 0.8 else (200, 200, 255)
                        beat_text = f"Beat strength: {kick:.0%}"
                        beat_surf = self.font_tiny.render(beat_text, True, color)
                        screen.blit(beat_surf, (x + padding, y_offset))
                        y_offset += 12
                    
                    # Vocal presence
                    vocal = features.get('vocal_presence', 0)
                    if vocal > 0.4:
                        color = (255, 255, 150) if vocal > 0.7 else (255, 255, 200)
                        vocal_text = f"Rap vocals: {vocal:.0%}"
                        vocal_surf = self.font_tiny.render(vocal_text, True, color)
                        screen.blit(vocal_surf, (x + padding, y_offset))
                        y_offset += 12
    
    def _draw_harmony_section(self, screen: pygame.Surface, x: int, y: int, 
                             width: int, height: int, harmony_data: Dict):
        """Draw harmonic analysis section"""
        # Background
        harmony_bg = pygame.Surface((width, height))
        harmony_bg.set_alpha(180)
        harmony_bg.fill((25, 30, 40))
        screen.blit(harmony_bg, (x, y))
        
        # Border
        pygame.draw.rect(screen, (90, 100, 120), (x, y, width, height), 1)
        
        padding = 5
        y_offset = y + padding
        
        # Key with confidence
        if self.font_medium:
            key = harmony_data.get('key', 'Unknown')
            confidence = harmony_data.get('key_confidence', 0.0)
            
            # Color based on major/minor
            color = (255, 220, 150) if 'Major' in key else (150, 200, 255)
            
            key_text = f"Key: {key}"
            key_surf = self.font_medium.render(key_text, True, color)
            screen.blit(key_surf, (x + padding, y_offset))
            y_offset += 25
            
            # Chords
            if self.font_small and harmony_data.get('chords'):
                chords = harmony_data['chords'][-3:]  # Last 3 chords
                chord_text = "Chords: " + " → ".join(chords)
                chord_surf = self.font_small.render(chord_text, True, (180, 200, 220))
                # Truncate if too long
                if chord_surf.get_width() > width - 2 * padding:
                    chord_text = "Chords: " + chords[-1] if chords else "N/A"
                    chord_surf = self.font_small.render(chord_text, True, (180, 200, 220))
                screen.blit(chord_surf, (x + padding, y_offset))
                y_offset += 20
                
            # Harmonic complexity
            if self.font_tiny:
                complexity = harmony_data.get('harmonic_complexity', 0)
                complexity_text = f"Complexity: {complexity:.0%}"
                complexity_color = (
                    (150, 255, 150) if complexity < 0.3 else
                    (255, 255, 150) if complexity < 0.7 else
                    (255, 150, 150)
                )
                complexity_surf = self.font_tiny.render(complexity_text, True, complexity_color)
                screen.blit(complexity_surf, (x + padding, y_offset))
    
    def _draw_cross_analysis(self, screen: pygame.Surface, x: int, y: int, 
                            width: int, height: int, cross_data: Dict):
        """Draw cross-analysis results"""
        # Background
        cross_bg = pygame.Surface((width, height))
        cross_bg.set_alpha(180)
        cross_bg.fill((30, 30, 35))
        screen.blit(cross_bg, (x, y))
        
        # Border
        pygame.draw.rect(screen, (110, 110, 130), (x, y, width, height), 1)
        
        padding = 5
        y_offset = y + padding
        
        if self.font_small:
            # Overall confidence
            overall_conf = cross_data.get('overall_confidence', 0.0)
            conf_text = f"Integrated Confidence: {overall_conf:.0%}"
            conf_color = (
                (255, 150, 150) if overall_conf < 0.5 else
                (255, 255, 150) if overall_conf < 0.7 else
                (150, 255, 150)
            )
            conf_surf = self.font_small.render(conf_text, True, conf_color)
            screen.blit(conf_surf, (x + padding, y_offset))
            y_offset += 20
            
            # Genre-harmony consistency
            consistency = cross_data.get('harmonic_genre_consistency', 0.0)
            consistency_text = f"Genre-Harmony Match: {consistency:.0%}"
            consistency_surf = self.font_small.render(consistency_text, True, (180, 200, 220))
            screen.blit(consistency_surf, (x + padding, y_offset))
            y_offset += 20
            
            # Suggested progression
            if self.font_tiny:
                progression = cross_data.get('genre_typical_progression', '')
                if progression:
                    prog_text = f"Typical progression: {progression}"
                    prog_surf = self.font_tiny.render(prog_text, True, (160, 180, 200))
                    screen.blit(prog_surf, (x + padding, y_offset))
    
    def _draw_hip_hop_analysis(self, screen: pygame.Surface, x: int, y: int,
                                width: int, height: int, hip_hop_data: Dict):
        """Draw dedicated hip-hop analysis visualization"""
        # Background with hip-hop themed color
        hip_hop_bg = pygame.Surface((width, height))
        hip_hop_bg.set_alpha(200)
        hip_hop_bg.fill((35, 25, 35))  # Purple-ish tint
        screen.blit(hip_hop_bg, (x, y))
        
        # Border with confidence-based color
        confidence = hip_hop_data.get('confidence', 0)
        border_color = (
            (200, 100, 200) if confidence > 0.8 else
            (150, 100, 150) if confidence > 0.6 else
            (100, 80, 100)
        )
        pygame.draw.rect(screen, border_color, (x, y, width, height), 2)
        
        padding = 10
        y_offset = y + padding
        
        if self.font_small:
            # Title with subgenre
            subgenre = hip_hop_data.get('subgenre', 'Hip-Hop')
            title = f"Hip-Hop Analysis: {subgenre.upper()}"
            title_surf = self.font_small.render(title, True, (255, 200, 255))
            title_rect = title_surf.get_rect(centerx=x + width // 2, top=y_offset)
            screen.blit(title_surf, title_rect)
            y_offset += 25
            
            # Feature bars visualization
            features = hip_hop_data.get('features', {})
            bar_height = 8
            bar_spacing = 12
            label_width = 100
            bar_x_start = x + padding + label_width
            bar_width = width - 2 * padding - label_width - 60  # Leave room for percentage
            
            # Define features to display with colors
            feature_display = [
                ('sub_bass_presence', '808/Sub-bass', (255, 100, 100)),
                ('kick_pattern_score', 'Kick Pattern', (100, 150, 255)),
                ('hihat_density', 'Hi-hat Density', (100, 255, 100)),
                ('spectral_tilt', 'Bass Emphasis', (255, 150, 100)),
                ('vocal_presence', 'Rap Vocals', (255, 255, 100))
            ]
            
            if self.font_tiny:
                for feature_key, feature_name, color in feature_display:
                    if feature_key in features:
                        value = features[feature_key]
                        
                        # Feature label
                        label_surf = self.font_tiny.render(feature_name + ":", True, (180, 180, 200))
                        screen.blit(label_surf, (x + padding, y_offset))
                        
                        # Background bar
                        bar_bg_rect = pygame.Rect(bar_x_start, y_offset, bar_width, bar_height)
                        pygame.draw.rect(screen, (40, 40, 50), bar_bg_rect)
                        
                        # Value bar
                        fill_width = int(bar_width * value)
                        if fill_width > 0:
                            # Gradient effect
                            for i in range(bar_height):
                                gradient_color = tuple(int(c * (1 - i * 0.05)) for c in color)
                                pygame.draw.line(screen, gradient_color,
                                               (bar_x_start, y_offset + i),
                                               (bar_x_start + fill_width, y_offset + i))
                        
                        # Border
                        pygame.draw.rect(screen, (80, 80, 100), bar_bg_rect, 1)
                        
                        # Percentage
                        pct_text = f"{value:.0%}"
                        pct_surf = self.font_tiny.render(pct_text, True, color)
                        pct_rect = pct_surf.get_rect(left=bar_x_start + bar_width + 5, centery=y_offset + bar_height // 2)
                        screen.blit(pct_surf, pct_rect)
                        
                        y_offset += bar_spacing
    
    def _draw_chromagram_bar(self, screen: pygame.Surface, x: int, y: int, 
                            width: int, height: int, chromagram: np.ndarray):
        """Draw chromagram as horizontal bar with border and color coding"""
        if chromagram is None or len(chromagram) != 12:
            return
        
        # Draw background and border for chromagram area (including labels)
        total_height = height + 20  # Include space for labels
        pygame.draw.rect(screen, (25, 25, 35), (x - 5, y - 5, width + 10, total_height + 10))
        pygame.draw.rect(screen, (80, 80, 100), (x - 5, y - 5, width + 10, total_height + 10), 2)
        
        note_width = width // 12
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Color scheme for chromagram (rainbow-like)
        note_colors = [
            (255, 100, 100),  # C - Red
            (255, 150, 100),  # C# - Red-Orange
            (255, 200, 100),  # D - Orange
            (255, 255, 100),  # D# - Yellow
            (200, 255, 100),  # E - Yellow-Green
            (100, 255, 100),  # F - Green
            (100, 255, 200),  # F# - Green-Cyan
            (100, 200, 255),  # G - Cyan
            (100, 150, 255),  # G# - Cyan-Blue
            (100, 100, 255),  # A - Blue
            (150, 100, 255),  # A# - Blue-Purple
            (200, 100, 255),  # B - Purple
        ]
        
        # Scale factor to make bars more visible (amplify the values)
        scale_factor = 1.5
        
        for i, (note, value, base_color) in enumerate(zip(note_names, chromagram, note_colors)):
            note_x = x + i * note_width
            
            # Scale and clamp the value
            scaled_value = min(value * scale_factor, 1.0)
            
            # Bar height based on scaled value
            bar_height = int(height * scaled_value * 0.9) if scaled_value > 0 else 0  # 0.9 to leave some margin
            bar_y = y + height - bar_height
            
            # Adjust color brightness based on value
            if bar_height > 0:
                brightness = 0.5 + 0.5 * scaled_value  # Range from 0.5 to 1.0
                bar_color = tuple(int(c * brightness) for c in base_color)
                
                # Draw bar with slight padding
                pygame.draw.rect(screen, bar_color, (note_x + 3, bar_y, note_width - 6, bar_height))
                
                # Draw thin separator lines between notes
                if i < 11:
                    pygame.draw.line(screen, (50, 50, 60), 
                                   (note_x + note_width, y), 
                                   (note_x + note_width, y + height), 1)
            
            # Draw note label with better contrast
            if self.font_tiny:
                # Background for label
                label_bg_color = (40, 40, 50) if '#' in note else (50, 50, 60)
                pygame.draw.rect(screen, label_bg_color, 
                               (note_x + 2, y + height + 2, note_width - 4, 16))
                
                # Label text
                label_color = (220, 220, 240) if '#' not in note else (180, 180, 200)
                label_surf = self.font_tiny.render(note, True, label_color)
                label_rect = label_surf.get_rect(centerx=note_x + note_width // 2, 
                                                centery=y + height + 10)
                screen.blit(label_surf, label_rect)
    
    def _draw_confidence_graph(self, screen: pygame.Surface, x: int, y: int, 
                              width: int, height: int):
        """Draw confidence history graph"""
        if len(self.confidence_history) < 2:
            return
            
        # Background
        pygame.draw.rect(screen, (25, 25, 35), (x, y, width, height))
        pygame.draw.rect(screen, (60, 60, 80), (x, y, width, height), 1)
        
        # Draw graph
        points = []
        for i, conf in enumerate(self.confidence_history):
            x_pos = x + int(i * width / len(self.confidence_history))
            y_pos = y + height - int(conf * height)
            points.append((x_pos, y_pos))
        
        if len(points) > 1:
            pygame.draw.lines(screen, (150, 200, 255), False, points, 2)
        
        # Label
        if self.font_tiny:
            label = "Confidence History"
            label_surf = self.font_tiny.render(label, True, (150, 150, 170))
            screen.blit(label_surf, (x + 5, y - 12))
    
    def _get_genre_color(self, genre: str) -> Tuple[int, int, int]:
        """Get color associated with genre"""
        genre_colors = {
            'Rock': (200, 100, 100),
            'Pop': (255, 150, 200),
            'Jazz': (150, 150, 255),
            'Classical': (200, 200, 150),
            'Electronic': (100, 255, 200),
            'Hip-Hop': (220, 100, 220),  # Purple for hip-hop
            'Metal': (150, 50, 50),
            'Country': (200, 180, 120),
            'R&B': (200, 150, 255),
            'Folk': (150, 200, 150),
            'Unknown': (150, 150, 150)
        }
        
        return genre_colors.get(genre, (150, 150, 150))
    
    def toggle_view_mode(self):
        """Cycle through view modes"""
        modes = ['integrated', 'genre_focus', 'harmony_focus']
        current_idx = modes.index(self.display_mode)
        self.display_mode = modes[(current_idx + 1) % len(modes)]
    
    def get_results(self) -> Dict[str, Any]:
        """Get current analysis results"""
        return self.analyzer.get_results()