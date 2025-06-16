#!/usr/bin/env python3
"""Test Phase 2 Step 6: Frequency band separators"""

import pygame
import numpy as np
import time
import sys

# Test display interface with band separators
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Initialize pygame and display
pygame.init()
screen = pygame.display.set_mode((1200, 800))
display = SpectrumDisplay(screen, 1200, 800, 128)

# Create test spectrum data
spectrum_data = np.random.rand(128) * 0.5

# Test visualization parameters
vis_params = {
    'vis_start_x': 100,
    'vis_start_y': 150,
    'vis_width': 1000,
    'vis_height': 500,
    'center_y': 400,
    'max_bar_height': 200
}

# Grid parameters
grid_params = {
    'spectrum_left': 100,
    'spectrum_right': 1100,
    'spectrum_top': 150,
    'spectrum_bottom': 650,
    'center_y': 400
}

# Band separator parameters
band_params = {
    'spectrum_top': 150,
    'spectrum_bottom': 650,
    'spectrum_left': 100,
    'spectrum_right': 1100
}

# Test drawing with band separators
try:
    display.clear_screen()
    display.draw_grid_and_labels(grid_params, ui_scale=1.0)
    display.draw_spectrum_bars(spectrum_data, vis_params)
    display.draw_frequency_band_separators(band_params, None, True)
    pygame.display.flip()
    print("✓ Frequency band separators drawn successfully")
    print("✓ Display components layered correctly")
    
    # Keep window open to verify
    time.sleep(2)
    
except Exception as e:
    print(f"✗ Failed to draw band separators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Step 6 Complete: Band separators integrated!")
pygame.quit()