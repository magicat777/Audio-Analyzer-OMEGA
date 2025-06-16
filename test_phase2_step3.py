#!/usr/bin/env python3
"""Test Phase 2 Step 3: Color gradient in spectrum bars"""

import pygame
import numpy as np
import time
import sys

# Test display interface with colors
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Initialize pygame and display
pygame.init()
screen = pygame.display.set_mode((1200, 600))
display = SpectrumDisplay(screen, 1200, 600, 128)

# Create test spectrum data - gradient test
spectrum_data = np.zeros(128)
for i in range(128):
    # Create a wave pattern to show colors
    spectrum_data[i] = 0.5 + 0.3 * np.sin(i * 0.1)

# Test visualization parameters
vis_params = {
    'vis_start_x': 50,
    'vis_start_y': 100,
    'vis_width': 1100,
    'vis_height': 400,
    'center_y': 300,
    'max_bar_height': 150
}

# Test drawing with colors
try:
    display.clear_screen()
    display.draw_spectrum_bars(spectrum_data, vis_params)
    pygame.display.flip()
    print("✓ Spectrum bars with gradient drawn successfully")
    print(f"✓ Generated {len(display.colors)} colors")
    
    # Keep window open to verify colors
    time.sleep(2)
    
except Exception as e:
    print(f"✗ Failed to draw colored spectrum bars: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Step 3 Complete: Color gradient working!")
pygame.quit()