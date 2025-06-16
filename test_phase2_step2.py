#!/usr/bin/env python3
"""Test Phase 2 Step 2: Spectrum bar drawing via display interface"""

import pygame
import numpy as np
import time
import sys

# Test display interface spectrum drawing
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Initialize pygame and display
pygame.init()
screen = pygame.display.set_mode((800, 600))
display = SpectrumDisplay(screen, 800, 600, 64)

# Create test spectrum data
spectrum_data = np.random.rand(64) * 0.8  # Random bars

# Test visualization parameters
vis_params = {
    'vis_start_x': 50,
    'vis_start_y': 100,
    'vis_width': 700,
    'vis_height': 400,
    'center_y': 300,
    'max_bar_height': 150
}

# Test drawing
try:
    display.clear_screen()
    display.draw_spectrum_bars(spectrum_data, vis_params)
    pygame.display.flip()
    print("✓ Spectrum bars drawn successfully")
    
    # Keep window open briefly to verify
    time.sleep(1)
    
except Exception as e:
    print(f"✗ Failed to draw spectrum bars: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Step 2 Complete: Spectrum bars rendering via display interface!")
pygame.quit()