#!/usr/bin/env python3
"""Test dynamic header resizing functionality"""

import subprocess
import time

def test_dynamic_header():
    print("Testing OMEGA-4 Dynamic Header Implementation")
    print("-" * 60)
    print("\nKey improvements implemented:")
    print("1. Header scales with window width")
    print("2. Font sizes adjust at different resolutions:")
    print("   - < 1400px: Medium title font")
    print("   - 1400-1920px: Large title, tiny features")
    print("   - 1920-2500px: Large title, tiny features")
    print("   - > 2500px: Large title, small features")
    print("\n3. Column widths are proportional:")
    print("   - Features: 65% of width (3 columns)")
    print("   - Technical: 35% of width (2 columns)")
    print("\n4. Header height adapts to content")
    print("5. Spectrum area dynamically positions below header")
    print("\n6. Text truncation prevents overlap")
    print("\nStarting OMEGA-4...")
    print("Press number keys 1-6 to test different resolutions:")
    print("  1: 1400x900 (Compact)")
    print("  2: 1600x900 (Standard)")
    print("  3: 1920x1080 (Full HD)")
    print("  4: 2560x1440 (QHD)")
    print("  5: 3000x1600 (Ultra)")
    print("  6: 3440x1800 (Super Ultra)")
    
    # Start the application
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Let it run for 15 seconds
    print("\nApplication running for 15 seconds...")
    print("Try resizing the window and pressing number keys!")
    
    start_time = time.time()
    while time.time() - start_time < 15:
        line = proc.stdout.readline()
        if line and ("Window resized" in line or "Starting visualization" in line):
            print(f"  {line.strip()}")
    
    # Cleanup
    proc.terminate()
    proc.wait()
    
    print("\n✅ Test complete")
    print("\nWhat to verify:")
    print("- Header maintains proper proportions at all sizes")
    print("- No text overlap at any resolution")
    print("- Spectrum area always starts below header")
    print("- Feature columns stay aligned")
    print("- Technical info doesn't overflow")

if __name__ == "__main__":
    test_dynamic_header()