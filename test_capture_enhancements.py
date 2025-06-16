#!/usr/bin/env python3
"""Test enhanced audio capture capabilities"""

import time
import numpy as np
from omega4.audio.capture import AudioCaptureConfig, AudioCaptureManager

def test_capture_stats():
    """Test performance monitoring capabilities"""
    print("🎵 Testing Enhanced Audio Capture with Performance Monitoring")
    print("=" * 70)
    
    # Create configuration with stats enabled
    config = AudioCaptureConfig(
        sample_rate=48000,
        chunk_size=512,
        enable_stats=True,
        stats_interval=2.0,  # Report every 2 seconds
        prefer_focusrite=True,
        buffer_size=20,  # Larger buffer for testing
        max_consecutive_errors=3,
        restart_delay=0.5
    )
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print(f"❌ Configuration issues: {', '.join(issues)}")
        return
    
    print("✅ Configuration validated successfully")
    print(f"\nSettings:")
    print(f"  Sample rate: {config.sample_rate}Hz")
    print(f"  Chunk size: {config.chunk_size} samples")
    print(f"  Buffer size: {config.buffer_size} frames")
    print(f"  Stats interval: {config.stats_interval}s")
    
    # Create capture manager
    manager = AudioCaptureManager(config=config)
    
    print("\n🎯 Starting audio capture...")
    if not manager.start():
        print("❌ Failed to start capture")
        return
    
    print("✅ Capture started successfully")
    print("\n📊 Monitoring performance (press Ctrl+C to stop)...")
    print("-" * 70)
    
    try:
        frames_received = 0
        last_stats_time = time.time()
        
        while True:
            # Get audio data
            audio_data = manager.get_audio_data()
            
            if audio_data is not None:
                frames_received += 1
                
                # Calculate RMS level
                rms = np.sqrt(np.mean(audio_data**2))
                
                # Print status every second
                current_time = time.time()
                if current_time - last_stats_time >= 1.0:
                    # Get capture statistics
                    stats = manager.get_stats()
                    
                    if stats:
                        print(f"\r📊 FPS: {stats['fps']:.1f} | "
                              f"Drop rate: {stats['drop_rate_percent']:.1f}% | "
                              f"Latency: {stats['avg_latency_ms']:.1f}ms | "
                              f"Buffer: {stats['buffer_count']}/{stats['buffer_capacity']} | "
                              f"RMS: {rms:.4f}", end='', flush=True)
                    
                    last_stats_time = current_time
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping capture...")
    
    # Get final statistics
    final_stats = manager.get_stats()
    if final_stats:
        print("\n📈 Final Statistics:")
        print(f"  Total frames processed: {final_stats['frames_processed']}")
        print(f"  Total frames dropped: {final_stats['frames_dropped']}")
        print(f"  Average FPS: {final_stats['fps']:.1f}")
        print(f"  Drop rate: {final_stats['drop_rate_percent']:.2f}%")
        print(f"  Average latency: {final_stats['avg_latency_ms']:.2f}ms")
        print(f"  Max latency: {final_stats['max_latency_ms']:.2f}ms")
        print(f"  Min latency: {final_stats['min_latency_ms']:.2f}ms")
        print(f"  Buffer drops: {final_stats['buffer_dropped']}")
        print(f"  Uptime: {final_stats['uptime_seconds']:.1f}s")
    
    # Stop capture
    manager.stop()
    print("\n✅ Capture stopped successfully")

def test_device_validation():
    """Test device validation capabilities"""
    print("\n🔍 Testing Device Validation")
    print("=" * 70)
    
    from omega4.audio.capture import PipeWireMonitorCapture
    
    config = AudioCaptureConfig(sample_rate=48000)
    capture = PipeWireMonitorCapture(config=config)
    
    # List available sources
    sources = capture.list_monitor_sources()
    
    if sources:
        # Test validation on first source
        test_source = sources[0][1]
        print(f"\nValidating device: {test_source}")
        
        is_valid, device_info = capture.validate_audio_device(test_source)
        
        if is_valid:
            print("✅ Device validation passed")
            print(f"  State: {device_info.get('state', 'unknown')}")
            print(f"  Description: {device_info.get('description', 'N/A')}")
            print(f"  Channels: {device_info.get('channels', 0)}")
            print(f"  Sample rates: {device_info.get('sample_rates', [])}")
        else:
            print(f"❌ Device validation failed: {device_info.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Test device validation first
    test_device_validation()
    
    print("\n" + "=" * 70 + "\n")
    
    # Test capture with stats
    test_capture_stats()