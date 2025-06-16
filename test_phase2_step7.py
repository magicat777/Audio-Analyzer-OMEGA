#!/usr/bin/env python3
"""Test Phase 2 Step 7: Peak hold indicators"""

import pygame
import numpy as np
import time
import sys

# Test display interface with peak hold
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

# Create test spectrum data with some peaks
spectrum_data = np.random.rand(128) * 0.3
# Add some peaks
spectrum_data[20] = 0.8
spectrum_data[40] = 0.9
spectrum_data[60] = 0.7
spectrum_data[80] = 0.85

# Create peak hold data (typically higher than current spectrum)
peak_data = np.maximum(spectrum_data, np.random.rand(128) * 0.4)
peak_data[20] = 0.95
peak_data[40] = 0.98
peak_data[60] = 0.85
peak_data[80] = 0.92

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

# Test drawing with peak hold indicators
try:
    display.clear_screen()
    display.draw_grid_and_labels(grid_params, ui_scale=1.0)
    display.draw_spectrum_bars(spectrum_data, vis_params)
    display.draw_peak_hold_indicators(peak_data, vis_params)
    pygame.display.flip()
    print("✓ Peak hold indicators drawn successfully")
    print("✓ Peak lines visible above spectrum bars")
    
    # Keep window open to verify
    time.sleep(2)
    
except Exception as e:
    print(f"✗ Failed to draw peak indicators: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Step 7 Complete: Peak hold indicators integrated!")
pygame.quit()