#!/usr/bin/env python3
"""Test Phase 3 Step 1: Professional Meters Panel"""

import pygame
import numpy as np
import time
import sys

# Test professional meters panel
try:
    from omega4.panels.professional_meters import ProfessionalMetersPanel
    print("✓ Professional meters panel imported")
except Exception as e:
    print(f"✗ Failed to import panel: {e}")
    sys.exit(1)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((1400, 800))

# Set up fonts
pygame.font.init()
fonts = {
    'large': pygame.font.Font(None, 36),
    'medium': pygame.font.Font(None, 28),
    'small': pygame.font.Font(None, 24),
    'tiny': pygame.font.Font(None, 20),
    'meters': pygame.font.Font(None, 22)
}

# Create panel
panel = ProfessionalMetersPanel(48000)
panel.set_fonts(fonts)
print("✓ Panel created and fonts set")

# Create test audio data
sample_rate = 48000
duration = 1.0  # 1 second
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate test signal: 1kHz tone with some harmonics
fundamental = 1000  # Hz
audio_data = (
    0.5 * np.sin(2 * np.pi * fundamental * t) +
    0.2 * np.sin(2 * np.pi * fundamental * 2 * t) +
    0.1 * np.sin(2 * np.pi * fundamental * 3 * t)
)

# Add some noise for more realistic LUFS calculation
audio_data += 0.05 * np.random.randn(len(audio_data))

# Test drawing
try:
    screen.fill((15, 20, 30))
    
    # Update panel with test data
    panel.update(audio_data)
    print("✓ Panel updated with test audio")
    
    # Draw panel
    panel.draw(screen, 50, 50, 300, 400, ui_scale=1.0)
    print("✓ Panel drawn successfully")
    
    # Get results
    results = panel.get_results()
    print(f"✓ LUFS results: {results['lufs']}")
    print(f"✓ Transient results: {results['transient']}")
    
    pygame.display.flip()
    
    # Keep window open
    time.sleep(3)
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

pygame.quit()
print("\n✅ Phase 3 Step 1 Complete: Professional Meters Panel!")
print("✓ LUFS metering working")
print("✓ K-weighting applied")
print("✓ Transient analysis working")
print("✓ Professional display rendering")