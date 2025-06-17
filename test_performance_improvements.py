#!/usr/bin/env python3
"""
Test script to verify performance improvements
"""

import subprocess
import time
import sys
import threading

def monitor_output(process):
    """Monitor process output for FPS information"""
    fps_values = []
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Run for 10 seconds
        line = process.stdout.readline()
        if not line:
            break
            
        line = line.strip()
        print(line)  # Echo output
        
        # Look for FPS information
        if "FPS:" in line and "Average=" in line:
            try:
                # Extract average FPS
                avg_start = line.find("Average=") + 8
                avg_end = line.find(",", avg_start)
                if avg_end == -1:
                    avg_end = line.find(" ", avg_start)
                if avg_end != -1:
                    avg_fps = float(line[avg_start:avg_end])
                    fps_values.append(avg_fps)
                    print(f"[TEST] Detected Average FPS: {avg_fps}")
            except:
                pass
    
    return fps_values

def test_performance():
    """Run the application and monitor performance"""
    print("Starting performance test...")
    print("This will run for 10 seconds and monitor FPS values")
    print("-" * 60)
    
    # Start the application
    process = subprocess.Popen(
        [sys.executable, 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    try:
        # Monitor output
        fps_values = monitor_output(process)
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        # Analyze results
        print("\n" + "-" * 60)
        print("PERFORMANCE TEST RESULTS:")
        print("-" * 60)
        
        if fps_values:
            avg_fps = sum(fps_values) / len(fps_values)
            min_fps = min(fps_values)
            max_fps = max(fps_values)
            
            print(f"FPS Samples collected: {len(fps_values)}")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Min FPS: {min_fps:.1f}")
            print(f"Max FPS: {max_fps:.1f}")
            
            if avg_fps >= 50:
                print("\n✅ PERFORMANCE: EXCELLENT (50+ FPS)")
            elif avg_fps >= 40:
                print("\n✅ PERFORMANCE: GOOD (40-50 FPS)")
            elif avg_fps >= 30:
                print("\n⚠️  PERFORMANCE: ACCEPTABLE (30-40 FPS)")
            else:
                print("\n❌ PERFORMANCE: POOR (<30 FPS)")
                
        else:
            print("No FPS data collected. The application may not have started properly.")
            
    except Exception as e:
        print(f"Error during test: {e}")
        process.kill()
        
if __name__ == "__main__":
    test_performance()