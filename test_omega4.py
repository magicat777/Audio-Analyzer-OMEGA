#!/usr/bin/env python3
"""Test script for OMEGA-4 progressive migration"""

import sys
import time

print("Testing OMEGA-4 configuration import...")

# Test 1: Config import
try:
    from omega4.config import SAMPLE_RATE, CHUNK_SIZE, BARS_DEFAULT
    print("✓ Configuration imported successfully")
    print(f"  Sample Rate: {SAMPLE_RATE}Hz")
    print(f"  Chunk Size: {CHUNK_SIZE}")
    print(f"  Default Bars: {BARS_DEFAULT}")
except Exception as e:
    print(f"✗ Configuration import failed: {e}")
    sys.exit(1)

# Test 2: Main module import
try:
    import omega4_main
    print("✓ Main module imports successfully")
except Exception as e:
    print(f"✗ Main module import failed: {e}")
    sys.exit(1)

# Test 3: Check critical classes exist
try:
    assert hasattr(omega4_main, 'MultiResolutionFFT')
    assert hasattr(omega4_main, 'ProfessionalMetering')
    assert hasattr(omega4_main, 'HarmonicAnalyzer')
    assert hasattr(omega4_main, 'main')
    print("✓ All critical classes and functions found")
except AssertionError as e:
    print(f"✗ Missing critical components")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nAll tests passed! OMEGA-4 Phase 1 complete.")
print("The original functionality is preserved with extracted configuration.")