#!/usr/bin/env python3
"""Test debug snapshot functionality"""

import subprocess
import time
import sys

print("Testing OMEGA-4 Debug Snapshot Feature")
print("=====================================")
print()
print("1. Starting the analyzer...")
print("2. Wait for it to load...")
print("3. Press 'D' in the application window to print a debug snapshot")
print("4. The snapshot will appear in this terminal")
print()
print("Starting in 3 seconds...")
time.sleep(3)

# Run the analyzer
try:
    subprocess.run([sys.executable, "omega4_main.py", "--width", "1400", "--height", "900", "--bars", "1024"])
except KeyboardInterrupt:
    print("\nAnalyzer stopped by user")
except Exception as e:
    print(f"\nError: {e}")