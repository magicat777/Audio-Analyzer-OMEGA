#!/usr/bin/env python3
"""Test OMEGA-4 with automatic debug snapshot"""

import subprocess
import time
import threading
import pyautogui
import sys

def simulate_d_press():
    """Simulate pressing 'D' key after delay"""
    print("Waiting 10 seconds for application to start and audio to play...")
    time.sleep(10)
    
    print("Simulating 'D' key press for debug snapshot...")
    pyautogui.press('d')
    
    # Wait for debug output
    time.sleep(2)
    
    print("\nWaiting 30 seconds for louder audio section...")
    time.sleep(30)
    
    print("Taking second debug snapshot...")
    pyautogui.press('d')
    
    # Wait a bit more then close
    time.sleep(5)
    print("\nClosing application...")
    pyautogui.press('escape')

def main():
    print("Starting OMEGA-4 analyzer test with debug output...")
    print("Please ensure audio is playing through your system")
    print("-" * 80)
    
    # Start the key press thread
    key_thread = threading.Thread(target=simulate_d_press)
    key_thread.daemon = True
    key_thread.start()
    
    # Run the application and capture output
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    debug_snapshots = []
    current_snapshot = []
    in_snapshot = False
    
    # Read output line by line
    for line in proc.stdout:
        print(line, end='')
        
        # Detect debug snapshot start/end
        if "PROFESSIONAL AUDIO ANALYZER V4 OMEGA-4 - DEBUG SNAPSHOT" in line:
            in_snapshot = True
            current_snapshot = [line]
        elif in_snapshot and "=" * 80 in line and len(current_snapshot) > 5:
            current_snapshot.append(line)
            debug_snapshots.append(''.join(current_snapshot))
            in_snapshot = False
            current_snapshot = []
        elif in_snapshot:
            current_snapshot.append(line)
    
    proc.wait()
    
    # Analyze debug snapshots
    print("\n" + "=" * 80)
    print("DEBUG SNAPSHOT ANALYSIS")
    print("=" * 80)
    
    for i, snapshot in enumerate(debug_snapshots):
        print(f"\nSnapshot {i+1}:")
        
        # Check key metrics
        if "Has Voice: YES" in snapshot:
            print("✅ Voice detection working")
        else:
            print("❌ Voice not detected")
            
        # Check frequency distribution
        lines = snapshot.split('\n')
        for line in lines:
            if "High-mid" in line and "avg=" in line:
                avg_val = float(line.split("avg=")[1].split()[0])
                if avg_val > 0.001:
                    print(f"✅ High-mid frequencies present (avg={avg_val})")
                else:
                    print(f"❌ High-mid frequencies missing (avg={avg_val})")
                    
            if "Brilliance" in line and "avg=" in line:
                avg_val = float(line.split("avg=")[1].split()[0])
                if avg_val > 0.0001:
                    print(f"✅ High frequencies present (avg={avg_val})")
                else:
                    print(f"❌ High frequencies missing (avg={avg_val})")
                    
            if "Active bars:" in line:
                match = line.split("Active bars: ")[1].split()[0]
                active, total = match.split('/')
                percentage = float(line.split('(')[1].split('%')[0])
                print(f"Active bars: {active}/{total} ({percentage}%)")
                
        # Check bass detail
        if "BASS DETAIL" in snapshot:
            print("Bass detail analysis:")
            bass_lines = []
            in_bass = False
            for line in lines:
                if "BASS DETAIL" in line:
                    in_bass = True
                elif in_bass and "VOICE DETECTION" in line:
                    break
                elif in_bass and "Hz):" in line and "dB" in line:
                    bass_lines.append(line.strip())
            
            if bass_lines:
                for line in bass_lines[:3]:  # Show first 3
                    print(f"  {line}")

if __name__ == "__main__":
    try:
        import pyautogui
    except ImportError:
        print("Installing pyautogui for key simulation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
        import pyautogui
    
    main()