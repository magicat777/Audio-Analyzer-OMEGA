#!/usr/bin/env python3
"""Test Phase 2 Steps 11-12: Analysis Grid and Header"""

import pygame
import numpy as np
import time
import sys

# Test display interface with grid and header
try:
    from omega4.visualization.display_interface import SpectrumDisplay
    print("✓ Display interface imported")
except Exception as e:
    print(f"✗ Failed to import display interface: {e}")
    sys.exit(1)

# Initialize pygame and display
pygame.init()
screen = pygame.display.set_mode((1600, 900))
display = SpectrumDisplay(screen, 1600, 900, 128)

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

# Test parameters
vis_params = {
    'vis_start_x': 100,
    'vis_start_y': 300,
    'vis_width': 1200,
    'vis_height': 500,
    'center_y': 550,
    'max_bar_height': 200
}

grid_params = {
    'spectrum_left': 100,
    'spectrum_right': 1300,
    'spectrum_top': 300,
    'spectrum_bottom': 800,
    'center_y': 550
}

# Test header data
header_data = {
    'title': 'Professional Audio Analyzer v4.1 OMEGA-2',
    'subtitle': 'Musical Perceptual Mapping • Professional Metering • Harmonic Analysis • Room Acoustics',
    'features': [
        '🎯 Peak Hold • ⚠️ Sub-Bass Monitor • 📊 Band Separators',
        '🔬 Technical Overlay • ⚖️ A/B Comparison • 📏 Analysis Grid',
        '🎛️ Gain: +/- keys • 0: Toggle Auto-gain • ESC: Exit'
    ],
    'audio_source': 'Professional Monitor',
    'sample_rate': 48000,
    'bars': 128,
    'fft_info': 'Multi-FFT: 8192/4096/2048/1024',
    'latency': 8.5,
    'fps': 58.7,
    'quality_mode': 'quality',
    'gain_db': 3.2,
    'auto_gain': False,
    'active_features': ['Grid', 'Bands', 'Tech', 'A/B'],
    'ui_scale': 1.0
}

# Test drawing all components
try:
    display.clear_screen()
    
    # Draw header first
    header_rect = pygame.Rect(0, 0, 1600, 280)
    pygame.draw.rect(screen, (18, 22, 32), header_rect)
    display.draw_header(header_data, 280)
    print("✓ Header drawn successfully")
    
    # Draw spectrum components
    display.draw_grid_and_labels(grid_params, ui_scale=1.0)
    display.draw_analysis_grid(grid_params, None, True)
    display.draw_spectrum_bars(spectrum_data, vis_params)
    display.draw_frequency_band_separators(grid_params, None, True)
    print("✓ Analysis grid drawn")
    
    pygame.display.flip()
    print("\n✅ Header and grid components working!")
    
    # Keep window open to verify
    time.sleep(3)
    
except Exception as e:
    print(f"✗ Failed to draw components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPhase 2 Steps 11-12 Complete!")
print("\nTotal Phase 2 Progress: ~90% Complete")
print("\nRemaining Phase 2 items:")
print("  - Help menu overlay")
print("  - Simple info overlays (voice, formants, A/B)")
pygame.quit()