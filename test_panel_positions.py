#\!/usr/bin/env python3
"""
Test to debug panel positioning
"""

# Simulate the panel positioning logic
def test_panel_positions():
    # Test case 1: All panels active
    print("Test 1: All panels active")
    active_panels = ['meters', 'harmonic', 'pitch', 'chromagram', 'genre', 'room', 'integrated']
    max_columns = 4
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        print(f"  {panel}: row={row}, col={col}")
    
    print("\nTest 2: Integrated panel ON (chromagram and genre hidden)")
    active_panels = ['meters', 'harmonic', 'pitch', 'room', 'integrated']
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        print(f"  {panel}: row={row}, col={col}")
    
    print("\nTest 3: What user sees by default")
    active_panels = ['meters', 'harmonic', 'pitch', 'chromagram', 'genre', 'room']
    
    for i, panel in enumerate(active_panels):
        row = i // max_columns
        col = i % max_columns
        print(f"  {panel}: row={row}, col={col}")

if __name__ == "__main__":
    test_panel_positions()
