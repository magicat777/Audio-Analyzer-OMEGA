#!/usr/bin/env python3
"""
Test script for enhanced professional meters panel
"""

import numpy as np
import pygame
from omega4.panels.professional_meters import ProfessionalMetersPanel

def generate_test_signal(duration_ms: int, sample_rate: int = 48000) -> np.ndarray:
    """Generate test signal with known characteristics"""
    samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms/1000, samples)
    
    # Generate complex test signal
    # 1kHz tone at -20dB (for calibration)
    tone_1k = 0.1 * np.sin(2 * np.pi * 1000 * t)
    
    # Low frequency content (100Hz at -30dB)
    tone_100 = 0.03 * np.sin(2 * np.pi * 100 * t)
    
    # High frequency content (10kHz at -40dB)
    tone_10k = 0.01 * np.sin(2 * np.pi * 10000 * t)
    
    # Transient (click) at 100ms
    transient = np.zeros_like(t)
    transient_idx = int(0.1 * sample_rate)
    if transient_idx < len(transient):
        transient[transient_idx:transient_idx+10] = 0.5
    
    # Combine signals
    signal = tone_1k + tone_100 + tone_10k + transient
    
    # Add some noise
    noise = 0.001 * np.random.randn(len(signal))
    signal += noise
    
    return signal

def test_weighting_filters():
    """Test the different weighting filters"""
    print("\n=== Testing Weighting Filters ===")
    
    panel = ProfessionalMetersPanel(sample_rate=48000)
    
    # Generate test signal
    test_signal = generate_test_signal(100)  # 100ms
    
    # Test each weighting mode
    for mode in ['K', 'A', 'C', 'Z']:
        panel.set_weighting(mode)
        panel.update(test_signal)
        
        lufs = panel.get_results()['lufs']
        print(f"\n{mode}-weighted measurements:")
        print(f"  Momentary: {lufs['momentary']:.1f} LUFS")
        print(f"  Short-term: {lufs['short_term']:.1f} LUFS")
        print(f"  True Peak: {lufs['true_peak']:.1f} dBTP")

def test_true_peak_detection():
    """Test true peak detection with oversampling"""
    print("\n=== Testing True Peak Detection ===")
    
    panel = ProfessionalMetersPanel(sample_rate=48000)
    
    # Generate signal that will have inter-sample peaks
    samples = 480  # 10ms
    t = np.linspace(0, 0.01, samples)
    
    # Square wave (worst case for inter-sample peaks)
    square_wave = 0.9 * np.sign(np.sin(2 * np.pi * 1000 * t))
    
    # Update panel
    panel.update(square_wave)
    
    results = panel.get_results()['lufs']
    print(f"Square wave (0.9 amplitude):")
    print(f"  Sample Peak: {20 * np.log10(np.max(np.abs(square_wave))):.1f} dB")
    print(f"  True Peak: {results['true_peak']:.1f} dBTP")
    print(f"  Difference: {results['true_peak'] - 20 * np.log10(np.max(np.abs(square_wave))):.1f} dB")

def test_level_histogram():
    """Test level histogram functionality"""
    print("\n=== Testing Level Histogram ===")
    
    panel = ProfessionalMetersPanel(sample_rate=48000)
    
    # Feed varying levels over time
    levels_db = [-40, -30, -23, -20, -14, -10, -23, -23, -23]  # Target -23 LUFS
    
    for i, target_db in enumerate(levels_db):
        # Generate signal at target level
        amplitude = 10 ** (target_db / 20)
        samples = int(48000 * 0.1)  # 100ms chunks
        signal = amplitude * np.sin(2 * np.pi * 1000 * np.linspace(0, 0.1, samples))
        
        # Update panel multiple times to build history
        for _ in range(10):
            panel.update(signal)
    
    # Get histogram
    bins, hist = panel.get_level_histogram()
    
    # Find peak bin
    if hist.sum() > 0:
        peak_idx = np.argmax(hist)
        peak_level = bins[peak_idx]
        print(f"Level histogram peak at: {peak_level:.0f} dB")
        print(f"Distribution spread: {np.std(bins[hist > 0]):.1f} dB")

def test_peak_hold():
    """Test peak hold functionality"""
    print("\n=== Testing Peak Hold ===")
    
    panel = ProfessionalMetersPanel(sample_rate=48000)
    panel.set_peak_hold_time(2.0)  # 2 seconds
    
    # Generate peaks at different times
    sample_rate = 48000
    
    print("Generating peaks...")
    for i, peak_db in enumerate([-6, -3, -1, -10]):
        amplitude = 10 ** (peak_db / 20)
        samples = int(sample_rate * 0.1)  # 100ms
        
        # Short peak
        signal = np.zeros(samples)
        signal[samples//2:samples//2+100] = amplitude
        
        panel.update(signal)
        print(f"  Frame {i+1}: Peak = {peak_db:.0f} dB, Hold = {panel.peak_hold_value:.1f} dB")
    
    # Test hold decay
    print("\nTesting hold decay...")
    quiet_signal = 0.001 * np.random.randn(int(sample_rate * 0.1))
    
    for i in range(30):  # 3 seconds at 10 FPS
        panel.update(quiet_signal)
        if i % 10 == 0:
            print(f"  After {i/10:.1f}s: Hold = {panel.peak_hold_value:.1f} dB")

def test_gating():
    """Test gated vs ungated measurements"""
    print("\n=== Testing Gating ===")
    
    panel = ProfessionalMetersPanel(sample_rate=48000)
    
    # Generate signal with quiet and loud parts
    sample_rate = 48000
    duration = 0.5  # 500ms
    samples = int(sample_rate * duration)
    
    # Loud part (-20 LUFS)
    loud_signal = 0.1 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration/2, samples//2))
    
    # Quiet part (-60 LUFS)
    quiet_signal = 0.001 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration/2, samples//2))
    
    # Combine
    signal = np.concatenate([loud_signal, quiet_signal])
    
    # Test with gating on
    panel.use_gated_measurement = True
    for _ in range(10):  # Build up history
        panel.update(signal)
    
    gated_results = panel.get_results()['lufs'].copy()
    
    # Test with gating off
    panel.toggle_gating()
    panel.metering.lufs_integrated_history.clear()  # Clear history
    
    for _ in range(10):  # Build up history
        panel.update(signal)
    
    ungated_results = panel.get_results()['lufs']
    
    print(f"Gated integrated: {gated_results['integrated']:.1f} LUFS")
    print(f"Ungated integrated: {ungated_results['integrated']:.1f} LUFS")
    print(f"Difference: {gated_results['integrated'] - ungated_results['integrated']:.1f} dB")

def main():
    """Run all tests"""
    print("Enhanced Professional Meters Panel Test Suite")
    print("=" * 50)
    
    test_weighting_filters()
    test_true_peak_detection()
    test_level_histogram()
    test_peak_hold()
    test_gating()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nKeyboard controls in main app:")
    print("  M - Toggle professional meters panel")
    print("  Shift+W - Cycle weighting mode (K/A/C/Z)")
    print("  Shift+G - Toggle gated/ungated measurement")
    print("  Shift+R - Reset peak hold")
    print("  Shift+T - Cycle peak hold time")

if __name__ == "__main__":
    main()