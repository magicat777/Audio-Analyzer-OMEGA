#!/usr/bin/env python3
"""
Test script to verify the integrated music panel is working
"""

import sys
import pygame
import numpy as np
from omega4.panels.integrated_music_panel import IntegratedMusicPanel

def test_integrated_panel():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Integrated Music Panel Test")
    
    # Create fonts
    fonts = {
        'large': pygame.font.Font(None, 24),
        'medium': pygame.font.Font(None, 20),
        'small': pygame.font.Font(None, 16),
        'tiny': pygame.font.Font(None, 14)
    }
    
    # Create the integrated panel
    panel = IntegratedMusicPanel(sample_rate=48000)
    panel.set_fonts(fonts)
    
    # Create some test data
    fft_data = np.random.rand(2048) * 0.5
    audio_data = np.random.rand(4096) * 0.1
    frequencies = np.linspace(0, 24000, 2048)
    
    drum_info = {
        'kick': {'magnitude': 0.7, 'kick_detected': True},
        'snare': {'magnitude': 0.5, 'snare_detected': False}
    }
    
    harmonic_info = {
        'harmonic_series': [440, 880, 1320, 1760],
        'fundamental': 440
    }
    
    print("Testing Integrated Music Panel...")
    print("Press 'I' to update the panel with new data")
    print("Press ESC to exit")
    
    # Initial update to populate the panel
    panel.update(fft_data, audio_data, frequencies, drum_info, harmonic_info)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    # Update with new random data
                    fft_data = np.random.rand(2048) * 0.5
                    audio_data = np.random.rand(4096) * 0.1
                    panel.update(fft_data, audio_data, frequencies, drum_info, harmonic_info)
                    print("Panel updated with new data")
        
        # Clear screen
        screen.fill((30, 30, 40))
        
        # Draw the panel
        panel.draw(screen, 50, 50, 700, 500, ui_scale=1.0)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("Test completed successfully!")

if __name__ == "__main__":
    test_integrated_panel()