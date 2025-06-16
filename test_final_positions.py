#\!/usr/bin/env python3
"""
Test final panel positioning with integrated panel at column 1
"""

def test_panel_positions():
    print("Final positioning with integrated panel forced to column 1:\n")
    
    max_columns = 4
    
    print("1. When integrated panel is ON:")
    active_panels = ['meters', 'harmonic', 'pitch', 'room', 'integrated']
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        
        # Special handling for integrated
        if panel == 'integrated':
            col = 1  # Force to column 1
            print(f"  {panel}: row={row}, col={col} (forced to col 1, height=310px)")
        else:
            print(f"  {panel}: row={row}, col={col} (height=260px)")
    
    print("\nLayout visualization:")
    print("Row 0: [Meters] [Harmonic] [Pitch] [Room]")
    print("Row 1: [      ] [Integrated         ]")
    print("             ^-- Column 1 (under Harmonic)")

if __name__ == "__main__":
    test_panel_positions()
