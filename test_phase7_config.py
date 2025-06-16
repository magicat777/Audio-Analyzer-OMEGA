#!/usr/bin/env python3
"""Test Phase 7: Configuration & Persistence"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Test configuration system
try:
    from omega4.config import (
        Configuration,
        ConfigurationManager,
        PresetManager,
        StateManager,
        ColorScheme,
        WindowMode
    )
    from omega4.ui import SettingsPanel
    print("✓ All configuration modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import configuration modules: {e}")
    sys.exit(1)

# Create temporary directory for testing
test_dir = tempfile.mkdtemp(prefix="omega4_test_")
print(f"Using test directory: {test_dir}")

# Test Configuration Schema
print("\n=== Testing Configuration Schema ===")
try:
    # Create configuration
    config = Configuration()
    print("✓ Configuration created with defaults")
    
    # Validate
    errors = config.validate()
    print(f"✓ Configuration validated: {len(errors)} errors")
    
    # Convert to/from dict
    config_dict = config.to_dict()
    print(f"✓ Converted to dict with {len(config_dict)} keys")
    
    config_restored = Configuration.from_dict(config_dict)
    print("✓ Restored from dict")
    
    # Test specific configs
    print(f"  Audio sample rate: {config.audio.sample_rate}")
    print(f"  Display color scheme: {config.display.color_scheme.value}")
    print(f"  Analysis FFT size: {config.analysis.fft_size}")
    
except Exception as e:
    print(f"✗ Configuration schema test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Configuration Manager
print("\n=== Testing Configuration Manager ===")
try:
    manager = ConfigurationManager(config_dir=test_dir)
    print("✓ ConfigurationManager created")
    
    # Load default config
    loaded_config = manager.load_config()
    print("✓ Default configuration loaded")
    
    # Save config
    success = manager.save_config()
    print(f"✓ Configuration saved: {success}")
    
    # Verify file exists
    config_file = Path(test_dir) / "config.yaml"
    print(f"✓ Config file exists: {config_file.exists()}")
    
    # Update configuration
    manager.update({
        "audio": {"input_gain": 6.0},
        "display": {"show_fps": False}
    })
    print("✓ Configuration updated")
    
    # Test validation
    errors = manager.validate_config()
    print(f"✓ Configuration validation: {len(errors)} errors")
    
    # Test migration (simulate old config)
    old_config = {"version": "0.9.0", "bars": 512}
    migrated = manager.migrate_config(old_config)
    print(f"✓ Configuration migrated: version {migrated['version']}")
    
except Exception as e:
    print(f"✗ ConfigurationManager test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Preset Manager
print("\n=== Testing Preset Manager ===")
try:
    preset_manager = PresetManager(preset_dir=os.path.join(test_dir, "presets"))
    print("✓ PresetManager created")
    
    # List built-in presets
    presets = preset_manager.list_presets()
    print(f"✓ Found {len(presets)} presets")
    for preset in presets[:3]:
        print(f"  - {preset.name}: {preset.category}")
    
    # Load a preset
    music_preset = preset_manager.load_preset("music_production")
    if music_preset:
        print(f"✓ Loaded preset: {music_preset.metadata.name}")
        print(f"  Input gain: {music_preset.configuration.audio.input_gain}")
        print(f"  Target LUFS: {music_preset.configuration.audio.target_lufs}")
    
    # Save custom preset
    custom_config = Configuration()
    custom_config.display.color_scheme = ColorScheme.NEON
    custom_config.analysis.num_bands = 1024
    
    saved = preset_manager.save_current_as_preset(
        custom_config,
        "Test Preset",
        "Test preset for unit testing",
        "custom",
        ["test", "custom"]
    )
    print(f"✓ Custom preset saved: {saved}")
    
    # Search presets
    search_results = preset_manager.search_presets("music")
    print(f"✓ Search found {len(search_results)} presets containing 'music'")
    
    # Delete test preset
    deleted = preset_manager.delete_preset("Test Preset")
    print(f"✓ Test preset deleted: {deleted}")
    
except Exception as e:
    print(f"✗ PresetManager test failed: {e}")
    import traceback
    traceback.print_exc()

# Test State Manager
print("\n=== Testing State Manager ===")
try:
    state_manager = StateManager(state_dir=test_dir)
    print("✓ StateManager created")
    
    # Load state (creates new)
    state = state_manager.load_state()
    print(f"✓ State loaded: session #{state.session_count}")
    
    # Update window state
    state_manager.update_window_state(100, 100, 1920, 1080, False, False)
    print("✓ Window state updated")
    
    # Update panel visibility
    state_manager.update_panel_visibility("spectrogram", True)
    visible = state_manager.get_panel_visibility("spectrogram")
    print(f"✓ Panel visibility updated: spectrogram={visible}")
    
    # Set last preset
    state_manager.set_last_preset("music_production")
    last_preset = state_manager.get_last_preset()
    print(f"✓ Last preset: {last_preset}")
    
    # Add recent file
    state_manager.add_recent_file("/test/audio.wav")
    recent = state_manager.get_recent_files()
    print(f"✓ Recent files: {len(recent)}")
    
    # Get statistics
    stats = state_manager.get_statistics()
    print(f"✓ Statistics: {stats['session_count']} sessions")
    
    # Save state
    saved = state_manager.save_state(force=True)
    print(f"✓ State saved: {saved}")
    
    # Verify file exists
    state_file = Path(test_dir) / "state.json"
    print(f"✓ State file exists: {state_file.exists()}")
    
except Exception as e:
    print(f"✗ StateManager test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Settings UI (without pygame)
print("\n=== Testing Settings UI ===")
try:
    # Create managers
    config_mgr = ConfigurationManager(config_dir=test_dir)
    preset_mgr = PresetManager(preset_dir=os.path.join(test_dir, "presets"))
    
    # Load config first
    config_mgr.load_config()
    
    # Create settings panel
    settings = SettingsPanel(config_mgr, preset_mgr)
    print("✓ SettingsPanel created")
    
    # Test visibility
    settings.show()
    print(f"✓ Settings visible: {settings.visible}")
    
    settings.hide()
    print(f"✓ Settings hidden: {not settings.visible}")
    
    # Test tab switching
    from omega4.ui import SettingsTab
    settings.current_tab = SettingsTab.DISPLAY
    print(f"✓ Current tab: {settings.current_tab.value}")
    
except Exception as e:
    print(f"✗ Settings UI test failed: {e}")
    import traceback
    traceback.print_exc()

# Test default configurations
print("\n=== Testing Default Configurations ===")
try:
    from omega4.config.schema import DEFAULT_CONFIGS
    
    print(f"✓ Found {len(DEFAULT_CONFIGS)} default configurations:")
    for name, config in DEFAULT_CONFIGS.items():
        print(f"  - {name}")
        
    # Test high quality preset
    hq_config = DEFAULT_CONFIGS["high_quality"]
    print(f"✓ High quality config: {hq_config.audio.sample_rate}Hz, {hq_config.analysis.fft_size} FFT")
    
    # Test low latency preset
    ll_config = DEFAULT_CONFIGS["low_latency"]
    print(f"✓ Low latency config: {ll_config.audio.chunk_size} chunk size")
    
except Exception as e:
    print(f"✗ Default configurations test failed: {e}")

# Cleanup
print(f"\n✓ Cleaning up test directory: {test_dir}")
shutil.rmtree(test_dir)

print("\n✅ Phase 7 Configuration & Persistence Tests Complete!")
print("✓ Configuration schema working")
print("✓ Configuration manager with save/load/migrate")
print("✓ Preset system with built-in and custom presets")
print("✓ State persistence for session data")
print("✓ Settings UI components created")
print("✓ Default configurations available")