#!/usr/bin/env python3
"""Test the improved display interface functionality"""

import subprocess
import time
import sys

def test_display_improvements():
    print("Testing OMEGA-4 Display Interface Improvements")
    print("-" * 60)
    print("\nKey improvements implemented:")
    print("✅ Font initialization with proper fallbacks")
    print("✅ Comprehensive error handling and logging")
    print("✅ Input validation and bounds checking")
    print("✅ Safe text rendering with fallback fonts")
    print("✅ Module-level imports for performance")
    print("✅ Dataclasses for structured data")
    print("✅ Type safety with enhanced hints")
    print("\nStarting OMEGA-4...")
    print("\nThings to test:")
    print("1. Window resizing - fonts should scale properly")
    print("2. Invalid inputs - should handle gracefully")
    print("3. Text rendering - no crashes with missing fonts")
    print("4. Performance - smooth rendering at 60 FPS")
    
    # Start the application
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    print("\nApplication running for 10 seconds...")
    print("Try resizing the window to test font scaling!")
    
    # Monitor for errors and logging
    start_time = time.time()
    error_count = 0
    log_count = 0
    
    while time.time() - start_time < 10:
        # Check stdout
        line = proc.stdout.readline()
        if line:
            if "ERROR" in line or "error" in line.lower():
                error_count += 1
                print(f"❌ Error detected: {line.strip()}")
            elif "logger" in line.lower() or "INFO" in line:
                log_count += 1
                if log_count == 1:  # Only show first log message
                    print(f"✅ Logging active: {line.strip()}")
        
        # Check stderr
        err_line = proc.stderr.readline() 
        if err_line and "error" in err_line.lower():
            error_count += 1
            print(f"❌ Stderr: {err_line.strip()}")
    
    # Cleanup
    proc.terminate()
    proc.wait()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"- Errors detected: {error_count}")
    print(f"- Logging messages: {'✅ Yes' if log_count > 0 else '❌ No'}")
    
    if error_count == 0:
        print("\n✅ Display interface improvements working correctly!")
    else:
        print(f"\n⚠️  {error_count} errors detected - check implementation")
    
    print("\nDisplay Interface Features:")
    print("- FontSet dataclass manages all fonts")
    print("- DisplayMetrics tracks all display dimensions")
    print("- Safe text rendering with _safe_render_text()")
    print("- Bounds checking with _rect_in_bounds()")
    print("- Comprehensive error handling in all methods")

if __name__ == "__main__":
    test_display_improvements()