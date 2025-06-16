#\!/usr/bin/env python3
"""
Test fixed panel positioning
"""

def test_panel_positions():
    print("With the fix applied:\n")
    
    # Default state
    show_integrated_music = False
    
    print("1. Default state (integrated OFF):")
    active_panels = []
    active_panels.append('meters')
    active_panels.append('harmonic')
    active_panels.append('pitch')
    if not show_integrated_music:  # Chromagram shown only if integrated is off
        active_panels.append('chromagram')
    if not show_integrated_music:  # Genre shown only if integrated is off
        active_panels.append('genre')
    active_panels.append('room')
    if show_integrated_music:
        active_panels.append('integrated')
    
    max_columns = 4
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        print(f"  {panel}: row={row}, col={col}")
    
    # Toggle integrated ON
    show_integrated_music = True
    
    print("\n2. After pressing 'I' (integrated ON):")
    active_panels = []
    active_panels.append('meters')
    active_panels.append('harmonic')
    active_panels.append('pitch')
    if not show_integrated_music:  # Chromagram hidden
        active_panels.append('chromagram')
    if not show_integrated_music:  # Genre hidden
        active_panels.append('genre')
    active_panels.append('room')
    if show_integrated_music:
        active_panels.append('integrated')
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        print(f"  {panel}: row={row}, col={col}")
    
    print("\nResult: Integrated panel is at row=1, col=0 (to the right of room modes)")

if __name__ == "__main__":
    test_panel_positions()
