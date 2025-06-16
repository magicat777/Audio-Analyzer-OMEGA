#!/usr/bin/env python3
"""Test Phase 2 Steps 8-10: Sub-bass, Adaptive Allocation, Technical Overlay"""

import pygame
import numpy as np
import time
import sys

# Test display interface with multiple components
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Initialize pygame and display
pygame.init()
screen = pygame.display.set_mode((1400, 900))
display = SpectrumDisplay(screen, 1400, 900, 128)

# Set up fonts
pygame.font.init()
fonts = {
    'large': pygame.font.Font(None, 36),
    'medium': pygame.font.Font(None, 28),
    'small': pygame.font.Font(None, 24),
    'tiny': pygame.font.Font(None, 20),
    'grid': pygame.font.Font(None, 18)
}
display.set_fonts(fonts)

# Create test data
spectrum_data = np.random.rand(128) * 0.5
peak_data = np.maximum(spectrum_data, np.random.rand(128) * 0.4)

# Test parameters
vis_params = {
    'vis_start_x': 100,
    'vis_start_y': 150,
    'vis_width': 1000,
    'vis_height': 500,
    'center_y': 400,
    'max_bar_height': 200
}

grid_params = {
    'spectrum_left': 100,
    'spectrum_right': 1100,
    'spectrum_top': 150,
    'spectrum_bottom': 650,
    'center_y': 400
}

# Test drawing all components
try:
    display.clear_screen()
    
    # Basic display components
    display.draw_grid_and_labels(grid_params, ui_scale=1.0)
    display.draw_spectrum_bars(spectrum_data, vis_params)
    display.draw_peak_hold_indicators(peak_data, vis_params)
    
    # Sub-bass indicator
    sub_bass_pos = {'x': 20, 'y': 350, 'width': 25, 'height': 200}
    display.draw_sub_bass_indicator(0.7, True, sub_bass_pos)
    print("✓ Sub-bass indicator drawn")
    
    # Adaptive allocation indicator
    adaptive_pos = {'x': 1150, 'y': 80, 'width': 150, 'height': 60}
    display.draw_adaptive_allocation_indicator(True, 'music', 0.75, adaptive_pos)
    print("✓ Adaptive allocation indicator drawn")
    
    # Technical overlay
    tech_data = {
        'bass_energy': 0.45,
        'mid_energy': 0.35,
        'high_energy': 0.20,
        'spectral_tilt': -2.5,
        'tilt_description': 'Balanced',
        'crest_factor': 12.3,
        'dynamic_range': 8.5,
        'room_modes': [
            {'freq': 45.2, 'duration': 1.2},
            {'freq': 90.5, 'duration': 0.8}
        ]
    }
    tech_pos = {'x': 950, 'y': 450, 'width': 320, 'height': 350}
    display.draw_technical_overlay(tech_data, tech_pos, True)
    print("✓ Technical overlay drawn")
    
    pygame.display.flip()
    print("\n✅ All new display components working!")
    
    # Keep window open to verify
    time.sleep(3)
    
except Exception as e:
    print(f"✗ Failed to draw components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Steps 8-10 Complete!")
print("\nProgress Summary:")
print("  ✓ Sub-bass indicator")
print("  ✓ Adaptive allocation indicator")
print("  ✓ Technical analysis overlay")
print("\nTotal Phase 2 Progress: ~80% Complete")
pygame.quit()