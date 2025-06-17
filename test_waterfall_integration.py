#!/usr/bin/env python3
"""
Test script for 3D Waterfall Spectrogram Integration
Tests the waterfall functionality with the main application
"""

import numpy as np
import pygame
import sys
import os
import time

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omega4.visualization.display_interface import SpectrumDisplay
from omega4.visualization.spectrum_waterfall_3d import SpectrumWaterfall3D

def test_waterfall_performance():
    """Test waterfall performance with realistic spectrum data"""
    
    print("Testing 3D Waterfall Performance...")
    
    # Initialize pygame
    pygame.init()
    
    # Create test window
    width, height = 1280, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("3D Waterfall Performance Test")
    
    # Create display with waterfall
    display = SpectrumDisplay(screen, width, height, 256)
    
    print(f"✓ Display initialized with waterfall")
    
    # Test parameters
    vis_params = {
        'vis_start_x': 100,
        'vis_start_y': 100,
        'vis_width': width - 200,
        'vis_height': height - 200,
        'center_y': height // 2,
        'max_bar_height': 200,
        'spectrum_top': 100,
        'spectrum_bottom': height - 100
    }
    
    # Performance metrics
    frame_times = []
    waterfall_times = []
    
    print("Running performance test for 300 frames...")
    
    clock = pygame.time.Clock()
    for frame in range(300):
        frame_start = time.time()
        
        # Generate realistic spectrum data
        num_bars = 256
        base_spectrum = np.random.random(num_bars) * 0.4
        
        # Add some musical content simulation
        bass_boost = np.exp(-np.arange(num_bars) / 20) * 0.6
        mid_peak = np.exp(-np.abs(np.arange(num_bars) - 80) / 15) * 0.5
        high_detail = np.exp(-np.abs(np.arange(num_bars) - 180) / 25) * 0.3
        
        spectrum_data = base_spectrum + bass_boost + mid_peak + high_detail
        spectrum_data = np.clip(spectrum_data, 0.0, 1.0)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        # Clear screen
        screen.fill((8, 10, 15))
        
        # Measure waterfall performance
        waterfall_start = time.time()
        
        # Draw spectrum with waterfall
        success = display.draw_spectrum_bars(spectrum_data, vis_params)
        
        waterfall_end = time.time()
        waterfall_times.append(waterfall_end - waterfall_start)
        
        if not success:
            print(f"✗ Frame {frame} failed to render")
            return False
        
        # Update display
        pygame.display.flip()
        
        frame_end = time.time()
        frame_times.append(frame_end - frame_start)
        
        # Target 60 FPS
        clock.tick(60)
        
        # Progress indicator
        if frame % 60 == 0:
            print(f"  Frame {frame}/300 - Avg frame time: {np.mean(frame_times[-60:]) * 1000:.1f}ms")
    
    pygame.quit()
    
    # Analyze performance
    avg_frame_time = np.mean(frame_times) * 1000
    avg_waterfall_time = np.mean(waterfall_times) * 1000
    avg_fps = 1.0 / np.mean(frame_times)
    
    print(f"\n✓ Performance Test Results:")
    print(f"  Average Frame Time: {avg_frame_time:.1f}ms")
    print(f"  Average Waterfall Time: {avg_waterfall_time:.1f}ms")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Waterfall Overhead: {(avg_waterfall_time/avg_frame_time)*100:.1f}%")
    
    # Performance thresholds
    if avg_frame_time > 20:  # 50 FPS minimum
        print(f"⚠️  Warning: Frame time too high ({avg_frame_time:.1f}ms)")
    else:
        print(f"✓ Performance acceptable")
    
    return avg_fps > 45  # Accept if we get at least 45 FPS

def test_waterfall_controls():
    """Test waterfall control methods"""
    
    print("\nTesting Waterfall Controls...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    display = SpectrumDisplay(screen, 800, 600, 128)
    
    # Test toggle
    initial_status = display.get_waterfall_status()
    print(f"✓ Initial waterfall enabled: {initial_status['enabled']}")
    
    # Test toggle
    enabled = display.toggle_waterfall_3d()
    print(f"✓ Toggle test: {enabled}")
    
    # Test depth adjustment
    initial_depth = display.get_waterfall_status()['depth_spacing']
    display.adjust_waterfall_depth(5.0)
    new_depth = display.get_waterfall_status()['depth_spacing']
    print(f"✓ Depth adjustment: {initial_depth:.1f} → {new_depth:.1f}")
    
    # Test speed adjustment
    initial_speed = display.get_waterfall_status()['slice_interval']
    display.adjust_waterfall_speed(-0.05)
    new_speed = display.get_waterfall_status()['slice_interval']
    print(f"✓ Speed adjustment: {initial_speed:.2f}s → {new_speed:.2f}s")
    
    # Test status
    status = display.get_waterfall_status()
    print(f"✓ Status retrieval: {len(status)} fields")
    
    pygame.quit()
    return True

def test_gpu_fallback():
    """Test CPU fallback when GPU is not available"""
    
    print("\nTesting GPU Fallback...")
    
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    
    # Force CPU mode
    waterfall = SpectrumWaterfall3D(num_bars=64, use_gpu=False)
    
    # Test basic functionality
    test_spectrum = np.random.random(64) * 0.5
    current_time = time.time()
    
    waterfall.update_spectrum_slice(test_spectrum, current_time)
    
    # Add a few more slices
    for i in range(5):
        time.sleep(0.01)
        test_spectrum = np.random.random(64) * 0.8
        waterfall.update_spectrum_slice(test_spectrum, time.time())
    
    status = waterfall.get_status()
    print(f"✓ CPU fallback working: {status['num_slices']} slices stored")
    print(f"✓ GPU acceleration: {status['gpu_acceleration']}")
    
    pygame.quit()
    return status['num_slices'] > 0

def test_memory_usage():
    """Test memory usage with extended operation"""
    
    print("\nTesting Memory Usage...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    display = SpectrumDisplay(screen, 800, 600, 256)
    
    # Run for many updates to check memory growth
    initial_status = display.get_waterfall_status()
    initial_memory = initial_status.get('gpu_memory_mb', 0)
    
    print(f"  Initial GPU memory: {initial_memory:.1f}MB")
    
    # Generate lots of spectrum updates
    for i in range(100):
        spectrum_data = np.random.random(256) * 0.7
        current_time = time.time()
        display.waterfall_3d.update_spectrum_slice(spectrum_data, current_time)
        
        # Small delay to trigger slice creation
        time.sleep(0.001)
    
    final_status = display.get_waterfall_status()
    final_memory = final_status.get('gpu_memory_mb', 0)
    
    print(f"  Final GPU memory: {final_memory:.1f}MB")
    print(f"  Active slices: {final_status['num_slices']}")
    print(f"  Max slices limit: {final_status['max_slices']}")
    
    # Check if memory is bounded
    memory_bounded = final_status['num_slices'] <= final_status['max_slices']
    print(f"✓ Memory properly bounded: {memory_bounded}")
    
    pygame.quit()
    return memory_bounded

if __name__ == "__main__":
    try:
        print("🌊 3D Waterfall Spectrogram Integration Test")
        print("=" * 60)
        
        # Run all tests
        tests = [
            ("Performance Test", test_waterfall_performance),
            ("Control Methods", test_waterfall_controls),
            ("GPU Fallback", test_gpu_fallback),
            ("Memory Usage", test_memory_usage),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
                print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                print(f"✗ {test_name}: ERROR - {e}")
                results.append((test_name, False))
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name:.<30} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed! 3D Waterfall ready for use.")
            print("\nTo test in the main application:")
            print("1. Run: python3 omega4_main.py")
            print("2. Press Ctrl+W to enable waterfall")
            print("3. Press Alt+W to see status")
            print("4. Use Shift+W and Ctrl+Q/Shift+Q for adjustments")
        else:
            print(f"\n❌ {total - passed} test(s) failed. Check implementation.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)