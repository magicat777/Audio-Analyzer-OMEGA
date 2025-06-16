#!/usr/bin/env python3
"""Test Phase 2 Step 1: Display interface with clear screen"""

import pygame
import time
import sys

# Test display interface can be imported and used
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported successfully")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Test display can be initialized
try:
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    display = SpectrumDisplay(screen, 800, 600, 256)
    print("✓ Display initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize display: {e}")
    sys.exit(1)

# Test clear screen works
try:
    display.clear_screen()
    pygame.display.flip()
    print("✓ Clear screen works")
except Exception as e:
    print(f"✗ Clear screen failed: {e}")
    sys.exit(1)

print("\nPhase 2 Step 1 Complete: Display interface with clear screen working!")
pygame.quit()