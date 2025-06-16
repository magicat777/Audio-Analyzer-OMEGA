#!/usr/bin/env python3
"""Simple test to verify the fixes by checking debug output"""

import subprocess
import time
import signal
import sys

def test_with_real_audio():
    print("Starting OMEGA-4 analyzer to test fixes...")
    print("Please play some music with vocals (e.g., Taylor Swift)")
    print("Press 'D' in the application window to generate a debug snapshot")
    print("Then check the terminal output for:")
    print("  1. Spectrum values should not all be maxed out")
    print("  2. Voice detection should show YES when vocals are present")
    print("  3. Bass detail dB values should be between -60 and 0 dB")
    print("\nStarting application...\n")
    
    # Start the application
    proc = subprocess.Popen(['python3', 'omega4_main.py'])
    
    try:
        # Wait for user to test
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_with_real_audio()