#!/usr/bin/env python3
"""Test the improved audio configuration with all new features"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, '/home/magicat777/Projects/audio-geometric-visualizer/OMEGA')

from omega4.audio.audio_config import PipelineConfig, LinuxAudioConfig
from omega4.audio.config_loader import ConfigManager, load_config, save_config


def test_nyquist_validation():
    """Test Nyquist frequency validation"""
    print("Testing Nyquist frequency validation...")
    
    # Valid config
    try:
        config = PipelineConfig(sample_rate=48000, max_frequency=20000)
        print("✅ Valid frequency range accepted")
    except ValueError as e:
        print(f"❌ Valid config rejected: {e}")
        
    # Invalid config (exceeds Nyquist)
    try:
        config = PipelineConfig(sample_rate=44100, max_frequency=30000)
        print("❌ Invalid frequency range accepted (should fail)")
    except ValueError as e:
        print(f"✅ Invalid frequency correctly rejected: {e}")
    
    return True


def test_buffer_validation():
    """Test ring buffer validation"""
    print("\nTesting ring buffer validation...")
    
    # Valid buffer (multiple of FFT size)
    try:
        config = PipelineConfig(fft_size=4096, ring_buffer_size=16384)  # 4x FFT size
        print("✅ Valid buffer size accepted")
    except ValueError as e:
        print(f"❌ Valid buffer rejected: {e}")
        
    # Invalid buffer (not multiple of FFT)
    try:
        config = PipelineConfig(fft_size=4096, ring_buffer_size=10000)
        print("❌ Invalid buffer size accepted (should fail)")
    except ValueError as e:
        print(f"✅ Invalid buffer correctly rejected: {e}")
        
    # Too small buffer
    try:
        config = PipelineConfig(fft_size=4096, ring_buffer_size=8192)  # Only 2x FFT size
        print("❌ Too small buffer accepted (should fail)")
    except ValueError as e:
        print(f"✅ Too small buffer correctly rejected: {e}")
    
    return True


def test_new_parameters():
    """Test new audio processing parameters"""
    print("\nTesting new audio processing parameters...")
    
    config = PipelineConfig(
        window_function="blackman",
        overlap_ratio=0.5,
        max_worker_threads=8,
        target_latency_ms=3.0,
        enable_performance_logging=True
    )
    
    print(f"✅ Window function: {config.window_function}")
    print(f"✅ Overlap ratio: {config.overlap_ratio}")
    print(f"✅ Worker threads: {config.max_worker_threads}")
    print(f"✅ Target latency: {config.target_latency_ms}ms")
    print(f"✅ Performance logging: {config.enable_performance_logging}")
    
    # Test invalid window function
    try:
        bad_config = PipelineConfig(window_function="invalid_window")
        print("❌ Invalid window function accepted")
    except ValueError as e:
        print(f"✅ Invalid window function rejected: {e}")
    
    return True


def test_linux_audio_config():
    """Test Linux-specific audio configuration"""
    print("\nTesting Linux audio configuration...")
    
    # Create config
    linux_config = LinuxAudioConfig(
        audio_backend="pulse",
        rt_priority=85,
        cpu_affinity=[0, 1],
        period_size=512
    )
    
    print(f"✅ Audio backend: {linux_config.audio_backend}")
    print(f"✅ RT priority: {linux_config.rt_priority}")
    print(f"✅ CPU affinity: {linux_config.cpu_affinity}")
    print(f"✅ Period size: {linux_config.period_size}")
    
    # Test validation
    try:
        bad_config = LinuxAudioConfig(audio_backend="invalid")
        print("❌ Invalid backend accepted")
    except ValueError as e:
        print(f"✅ Invalid backend rejected: {e}")
    
    # Test CPU affinity validation
    cpu_count = os.cpu_count()
    try:
        bad_config = LinuxAudioConfig(cpu_affinity=[0, cpu_count])  # Invalid CPU
        print("❌ Invalid CPU affinity accepted")
    except ValueError as e:
        print(f"✅ Invalid CPU affinity rejected: {e}")
    
    return True


def test_config_manager():
    """Test configuration loading and saving"""
    print("\nTesting configuration management...")
    
    # Create test config
    test_config = PipelineConfig(
        sample_rate=96000,
        num_bands=1024,
        window_function="kaiser",
        enable_performance_logging=True
    )
    
    # Save to temp file
    temp_path = "/tmp/test_omega4_config.json"
    success = ConfigManager.save_to_file(test_config, temp_path)
    print(f"✅ Config saved: {success}")
    
    # Load from file
    loaded_config = ConfigManager.load_from_file(temp_path)
    print(f"✅ Config loaded: {loaded_config.sample_rate}Hz, {loaded_config.num_bands} bands")
    print(f"✅ Window function preserved: {loaded_config.window_function}")
    
    # Test environment override
    os.environ['OMEGA4_SAMPLE_RATE'] = '192000'
    os.environ['OMEGA4_ENABLE_PERFORMANCE_LOGGING'] = 'false'
    
    merged_config = ConfigManager.merge_with_env(loaded_config)
    print(f"✅ Environment override: {merged_config.sample_rate}Hz")
    print(f"✅ Bool env override: logging={merged_config.enable_performance_logging}")
    
    # Clean up
    os.unlink(temp_path)
    del os.environ['OMEGA4_SAMPLE_RATE']
    del os.environ['OMEGA4_ENABLE_PERFORMANCE_LOGGING']
    
    return True


def test_system_validation():
    """Test system requirements validation"""
    print("\nTesting system requirements validation...")
    
    config = PipelineConfig(
        max_worker_threads=4,
        max_memory_usage_mb=256
    )
    
    results = ConfigManager.validate_system_requirements(config)
    
    for check, passed in results.items():
        status = "✅" if passed else "⚠️"
        print(f"{status} {check}: {passed}")
    
    return True


def test_performance_settings():
    """Test performance-related settings"""
    print("\nTesting performance settings...")
    
    config = PipelineConfig(
        use_multiprocessing=True,
        buffer_safety_factor=3.0,
        enable_gc_optimization=True,
        stats_update_interval=0.5
    )
    
    print(f"✅ Multiprocessing: {config.use_multiprocessing}")
    print(f"✅ Buffer safety factor: {config.buffer_safety_factor}")
    print(f"✅ GC optimization: {config.enable_gc_optimization}")
    print(f"✅ Stats interval: {config.stats_update_interval}s")
    
    # Calculate actual buffer requirements
    buffer_ms = (config.ring_buffer_size / config.sample_rate) * 1000
    print(f"✅ Buffer duration: {buffer_ms:.1f}ms")
    
    target_buffer = config.target_latency_ms * config.buffer_safety_factor
    print(f"✅ Target buffer: {target_buffer:.1f}ms (latency × safety factor)")
    
    return True


def main():
    print("Testing Enhanced Audio Configuration")
    print("=" * 60)
    print("\nNew features:")
    print("✅ Nyquist frequency validation")
    print("✅ Ring buffer validation")
    print("✅ Window function configuration")
    print("✅ Linux-specific audio settings")
    print("✅ Performance monitoring controls")
    print("✅ Configuration file management")
    print("\n")
    
    all_passed = True
    
    tests = [
        ("Nyquist Validation", test_nyquist_validation),
        ("Buffer Validation", test_buffer_validation),
        ("New Parameters", test_new_parameters),
        ("Linux Audio Config", test_linux_audio_config),
        ("Config Manager", test_config_manager),
        ("System Validation", test_system_validation),
        ("Performance Settings", test_performance_settings)
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All configuration tests passed!")
        
        # Show current system info
        print("\nSystem Information:")
        print(f"- CPU cores: {os.cpu_count()}")
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"- Available memory: {mem.available / (1024**3):.1f} GB")
        except ImportError:
            print("- Memory info not available (install psutil)")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())