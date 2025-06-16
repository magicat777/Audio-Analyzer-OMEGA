#!/usr/bin/env python3
"""Final comprehensive test of OMEGA-4"""

import subprocess
import time
import signal
import sys

def main():
    print("=" * 80)
    print("OMEGA-4 FINAL TEST")
    print("=" * 80)
    print("\nStarting OMEGA-4 analyzer...")
    print("The application will run for 30 seconds")
    print("Please play pop music with vocals through your audio system")
    print("\nPress 'D' in the application window to generate debug snapshots")
    print("Press 'M' to enable professional meters")
    print("Press 'Z' to show bass zoom panel")
    print("\n" + "-" * 80)
    
    # Start the application
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Set timeout
    def timeout_handler(signum, frame):
        print("\n\nTimeout reached, closing application...")
        proc.terminate()
        proc.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        # Read and display output
        for line in proc.stdout:
            print(line, end='')
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()