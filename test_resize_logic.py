#!/usr/bin/env python3
"""Test the window resize logic for bass zoom panel"""

# Simulate the height calculation logic
def calculate_height(show_bass_zoom):
    # Base components
    header_height = 120
    spectrum_height = 400
    spectrum_top_margin = 10
    spectrum_bottom_margin = 100
    panel_start_offset = 40
    footer_height = 80
    
    base_height = (header_height + spectrum_top_margin + spectrum_height + 
                  spectrum_bottom_margin + panel_start_offset + footer_height)
    
    # Panel configuration
    panel_height = 200
    panel_spacing = 20
    active_panels = 1 if show_bass_zoom else 0
    
    if active_panels > 0:
        total_panel_height = active_panels * (panel_height + panel_spacing)
    else:
        total_panel_height = 0
    
    required_height = base_height + total_panel_height
    
    print(f"Bass zoom: {show_bass_zoom}, Active panels: {active_panels}, Base: {base_height}, Panels: {total_panel_height}, Total: {required_height}")
    
    return required_height

print("Testing height calculations:")
print("=" * 50)

# Test with bass zoom OFF
height_off = calculate_height(False)
print(f"Height with bass zoom OFF: {height_off}px")

# Test with bass zoom ON
height_on = calculate_height(True)
print(f"Height with bass zoom ON: {height_on}px")

# Expected behavior
print(f"\nExpected height change: {height_on - height_off}px")
print(f"This should be panel_height (200) + panel_spacing (20) = 220px")

# Simulate the toggle sequence
print("\n" + "=" * 50)
print("Simulating toggle sequence:")
print("=" * 50)

current_height = 1080  # Start with a preset height
show_bass_zoom = False

for i in range(6):
    print(f"\nToggle {i+1}:")
    print(f"  Before: bass_zoom={show_bass_zoom}, height={current_height}")
    
    # Toggle
    show_bass_zoom = not show_bass_zoom
    required_height = calculate_height(show_bass_zoom)
    
    if abs(required_height - current_height) > 5:
        print(f"  Resizing from {current_height} to {required_height}")
        current_height = required_height
    else:
        print(f"  No resize needed")
    
    print(f"  After: bass_zoom={show_bass_zoom}, height={current_height}")

print("\n✅ The logic appears correct - each toggle should alternate between 750px and 970px")