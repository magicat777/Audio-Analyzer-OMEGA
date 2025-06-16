#\!/usr/bin/env python3
"""
Test final layout with integrated panel in column 2
"""

def test_panel_positions():
    print("Final layout with integrated panel in column 2:\n")
    
    max_columns = 4
    
    print("When integrated panel is ON:")
    active_panels = ['meters', 'harmonic', 'pitch', 'room', 'integrated']
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        
        # Special handling for integrated
        if panel == 'integrated':
            col = 2  # Force to column 2
            print(f"  {panel}: row={row}, col={col} (forced to col 2, height=310px)")
        else:
            print(f"  {panel}: row={row}, col={col} (height=260px)")
    
    print("\nLayout visualization:")
    print("Row 0: [Meters] [Harmonic] [Pitch] [Room]")
    print("Row 1: [      ] [        ] [Integrated] [    ]")
    print("                          ^-- Column 2 (under Pitch)")
    
    print("\nChromagram features:")
    print("- Bordered display area")
    print("- Color-coded bars (C=Red through B=Purple)")
    print("- 1.5x scaling for better visibility")
    print("- Separated note labels with backgrounds")

if __name__ == "__main__":
    test_panel_positions()
