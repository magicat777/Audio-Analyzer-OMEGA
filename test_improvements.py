#!/usr/bin/env python3
"""Test improvements to OMEGA-4"""

import subprocess
import time
import re

def check_improvements(output):
    """Check if improvements are working"""
    print("\n" + "="*60)
    print("IMPROVEMENT CHECK")
    print("="*60)
    
    # Check latency
    latency_match = re.search(r'Total latency: (\d+\.?\d*)', output)
    if latency_match:
        latency = float(latency_match.group(1))
        if latency < 60:
            print(f"✅ Latency: {latency:.1f}ms (target: <60ms)")
        else:
            print(f"❌ Latency: {latency:.1f}ms (too high)")
    
    # Check frequency distribution
    print("\nFrequency Balance:")
    freq_checks = [
        ("Bass", r'Bass\s+\[.*?\]:\s*[▂]+\s*avg=(\d+\.?\d*)', 0.10, 0.50),
        ("Mid", r'Mid\s+\[.*?\]:\s*[▂]*\s*avg=(\d+\.?\d*)', 0.02, 0.10),
        ("High-mid", r'High-mid\s+\[.*?\]:\s*[▂]*\s*avg=(\d+\.?\d*)', 0.01, 0.10),
        ("Brilliance", r'Brilliance\s+\[.*?\]:\s*[▂]*\s*avg=(\d+\.?\d*)', 0.001, 0.05)
    ]
    
    for name, pattern, min_val, max_val in freq_checks:
        match = re.search(pattern, output)
        if match:
            value = float(match.group(1))
            if min_val <= value <= max_val:
                print(f"  ✅ {name}: {value:.3f} (good range)")
            else:
                print(f"  ⚠️  {name}: {value:.3f} (expected {min_val}-{max_val})")
    
    # Check voice detection
    if "Has Voice: YES" in output:
        conf_match = re.search(r'Confidence: (\d+\.?\d*)', output)
        conf = float(conf_match.group(1)) if conf_match else 0
        print(f"\n✅ Voice Detection: Working (confidence: {conf:.1f}%)")
    else:
        print("\n❌ Voice Detection: Not detecting vocals")
    
    # Check bass detail
    print("\nBass Detail Check:")
    bass_0db = re.search(r'\+0\.0 dB', output)
    if bass_0db:
        print("  ❌ Unrealistic 0dB values present")
    else:
        print("  ✅ Realistic dB values")
    
    # Check band mapping
    if "Band 0: 11.7-23.4 Hz" in output and "Band 5: 11.7-23.4 Hz" in output:
        print("\n❌ Band Mapping: Still clustered at low frequencies")
    else:
        print("\n✅ Band Mapping: Properly distributed")

def main():
    print("OMEGA-4 Improvements Test")
    print("Please ensure music with vocals is playing")
    print("-" * 60)
    
    # Start the application
    proc = subprocess.Popen(
        ['python3', 'omega4_main.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    output_buffer = []
    start_time = time.time()
    found_debug = False
    
    # Run for up to 20 seconds or until we get a debug snapshot
    while time.time() - start_time < 20:
        line = proc.stdout.readline()
        if line:
            print(line, end='')
            output_buffer.append(line)
            
            # Look for debug snapshot
            if "PROFESSIONAL AUDIO ANALYZER V4 OMEGA-4 - DEBUG SNAPSHOT" in line:
                found_debug = True
            
            # If we completed a debug snapshot, analyze it
            if found_debug and "=" * 80 in line and len(output_buffer) > 50:
                full_output = ''.join(output_buffer)
                check_improvements(full_output)
                break
    
    # Cleanup
    proc.terminate()
    proc.wait()
    
    if not found_debug:
        print("\n⚠️  No debug snapshot found. Press 'D' in the application to generate one.")

if __name__ == "__main__":
    main()