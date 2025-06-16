#!/usr/bin/env python3
"""Test final fixes for OMEGA-4"""

import subprocess
import time
import sys
import re

def analyze_debug_output(output):
    """Analyze debug output for issues"""
    issues = []
    
    # Check band mapping
    if "[BANDS] First 10 frequency bands:" in output:
        band_lines = []
        lines = output.split('\n')
        capture = False
        for line in lines:
            if "Band 0:" in line:
                capture = True
            if capture and "Band" in line:
                band_lines.append(line)
                if len(band_lines) >= 10:
                    break
        
        # Check if bands are properly distributed
        if len(band_lines) >= 5:
            # Extract frequency ranges
            same_count = 0
            last_range = None
            for line in band_lines[:5]:
                match = re.search(r'(\d+\.?\d*)-(\d+\.?\d*) Hz', line)
                if match:
                    current_range = f"{match.group(1)}-{match.group(2)}"
                    if current_range == last_range:
                        same_count += 1
                    last_range = current_range
            
            if same_count >= 3:
                issues.append("❌ Band mapping: Multiple bands at same frequency")
            else:
                print("✅ Band mapping: Properly distributed")
    
    # Check voice detection
    if "Has Voice: YES" in output:
        print("✅ Voice detection: Working")
    else:
        issues.append("❌ Voice detection: Not detecting vocals")
    
    # Check high frequencies
    if "Brilliance" in output and "avg=" in output:
        for line in output.split('\n'):
            if "Brilliance" in line and "avg=" in line:
                match = re.search(r'avg=(\d+\.?\d*)', line)
                if match:
                    avg_val = float(match.group(1))
                    if avg_val > 0.005:
                        print(f"✅ High frequencies: Present (avg={avg_val})")
                    else:
                        issues.append(f"❌ High frequencies: Too low (avg={avg_val})")
                        
    # Check bass dB values
    bass_ok = True
    for line in output.split('\n'):
        if "Hz):" in line and "dB" in line and "+0.0 dB" in line:
            issues.append("❌ Bass detail: Showing unrealistic 0dB")
            bass_ok = False
            break
    if bass_ok and "BASS DETAIL" in output:
        print("✅ Bass detail: Realistic dB values")
    
    return issues

def main():
    print("=" * 80)
    print("OMEGA-4 FINAL FIXES TEST")
    print("=" * 80)
    print("\nStarting OMEGA-4 and capturing debug output...")
    print("This test will run for 20 seconds")
    print("Please ensure music with vocals is playing\n")
    
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    output_buffer = []
    debug_snapshots = []
    start_time = time.time()
    
    while time.time() - start_time < 20:
        line = proc.stdout.readline()
        if line:
            output_buffer.append(line)
            print(line, end='')
            
            # Capture debug snapshots
            if "PROFESSIONAL AUDIO ANALYZER V4 OMEGA-4 - DEBUG SNAPSHOT" in line:
                snapshot = [line]
                while True:
                    line = proc.stdout.readline()
                    if line:
                        output_buffer.append(line)
                        print(line, end='')
                        snapshot.append(line)
                        if "=" * 80 in line and len(snapshot) > 10:
                            debug_snapshots.append(''.join(snapshot))
                            break
                    else:
                        break
    
    # Terminate the process
    proc.terminate()
    proc.wait()
    
    # Analyze results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    full_output = ''.join(output_buffer)
    
    # Check startup band mapping
    print("\nStartup checks:")
    if "[WARNING] freq_points length" in full_output:
        print("⚠️  Frequency points needed regeneration")
    
    # Analyze each debug snapshot
    all_issues = []
    for i, snapshot in enumerate(debug_snapshots):
        print(f"\nDebug Snapshot {i+1} Analysis:")
        issues = analyze_debug_output(snapshot)
        all_issues.extend(issues)
    
    # Summary
    print("\n" + "-" * 80)
    if not all_issues:
        print("✅ ALL TESTS PASSED!")
    else:
        print("Issues found:")
        for issue in set(all_issues):  # Remove duplicates
            print(f"  {issue}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()