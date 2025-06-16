#!/usr/bin/env python3
"""Test the header panel implementation"""

import subprocess
import time
import sys

def test_header():
    print("Testing OMEGA-4 Header Panel Implementation")
    print("-" * 60)
    print("The header should display:")
    print("- Title: 'OMEGA-4 Professional Audio Analyzer'")
    print("- 3 columns of feature toggles with ON/OFF status")
    print("- 2 columns of technical information on the right")
    print("- FPS counter in top-right of header")
    print("\nStarting OMEGA-4...")
    
    # Start the application
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Let it run for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        line = proc.stdout.readline()
        if line:
            print(line, end='')
            
            # Look for startup messages
            if "Starting visualization" in line:
                print("\n✅ Application started successfully")
                print("✅ Check the window for the new header panel")
                break
    
    # Give user time to see the window
    print("\nApplication running for 5 more seconds...")
    time.sleep(5)
    
    # Cleanup
    proc.terminate()
    proc.wait()
    
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_header()