# OMEGA-4 Architecture Overview

## Current Architecture (After Phase 3)

```
omega4_main.py (Main Application - ~5000 lines)
│
├── omega4/config.py (Configuration - Phase 1)
│
├── omega4/visualization/
│   └── display_interface.py (Display Logic - Phase 2)
│
└── omega4/panels/ (Complex Panels - Phase 3)
    ├── professional_meters.py
    ├── vu_meters.py
    ├── bass_zoom.py
    ├── harmonic_analysis.py
    ├── pitch_detection.py (OMEGA)
    ├── chromagram.py (OMEGA-1)
    └── genre_classification.py (OMEGA-2)
```

## Target Architecture (After Phase 7)

```
omega4_main.py (Main Application - <1000 lines)
│
├── omega4/
│   ├── config.py (Base Configuration)
│   │
│   ├── config/ (Phase 7)
│   │   ├── manager.py (Configuration Management)
│   │   ├── presets.py (User Presets)
│   │   └── state.py (Persistent State)
│   │
│   ├── visualization/
│   │   └── display_interface.py (Core Display Logic)
│   │
│   ├── panels/ (Core Panels)
│   │   ├── professional_meters.py
│   │   ├── vu_meters.py
│   │   ├── bass_zoom.py
│   │   ├── harmonic_analysis.py
│   │   ├── pitch_detection.py
│   │   ├── chromagram.py
│   │   └── genre_classification.py
│   │
│   ├── analyzers/ (Phase 4)
│   │   ├── harmonic.py
│   │   ├── phase_coherence.py
│   │   ├── transient.py
│   │   ├── room_modes.py
│   │   └── drum_detection.py
│   │
│   ├── audio/ (Phase 5)
│   │   ├── capture.py (Audio Input)
│   │   ├── multi_resolution_fft.py (FFT Processing)
│   │   ├── pipeline.py (Processing Pipeline)
│   │   └── voice_detection.py (Voice Analysis)
│   │
│   ├── plugins/ (Phase 6)
│   │   ├── base.py (Plugin Interfaces)
│   │   ├── manager.py (Plugin Management)
│   │   ├── config.py (Plugin Configuration)
│   │   │
│   │   ├── panels/ (Plugin Panels)
│   │   │   ├── spectrogram.py
│   │   │   └── waterfall.py
│   │   │
│   │   └── analyzers/ (Plugin Analyzers)
│   │       └── midi_out.py
│   │
│   └── ui/ (Phase 7)
│       └── settings.py (Settings Interface)
```

## Module Dependencies

### Core Dependencies
```
omega4_main.py
    ↓
omega4/config.py ← omega4/config/manager.py
    ↓
omega4/audio/capture.py
    ↓
omega4/audio/multi_resolution_fft.py
    ↓
omega4/audio/pipeline.py
    ↓                      ↓
omega4/analyzers/*    omega4/panels/*
    ↓                      ↓
omega4/visualization/display_interface.py
```

### Plugin System
```
omega4/plugins/manager.py
    ↓
omega4/plugins/base.py
    ↓
omega4/plugins/panels/* ← omega4/panels/* (converted)
omega4/plugins/analyzers/*
```

## Data Flow

### Audio Processing Pipeline (Phase 5)
```
Audio Input (PulseAudio)
    ↓
AudioCapture.get_audio_data()
    ↓
AudioProcessingPipeline.process_frame()
    ├── MultiResolutionFFT.process()
    ├── Gain Control
    └── Normalization
    ↓
Analyzer Modules
    ├── DrumDetector.process()
    ├── HarmonicAnalyzer.analyze()
    ├── TransientAnalyzer.detect()
    └── VoiceDetector.detect()
    ↓
Panel Modules
    ├── Panel.update(data)
    └── Panel.get_results()
    ↓
Display Interface
    └── Panel.draw(screen)
```

### Plugin Lifecycle (Phase 6)
```
Application Start
    ↓
PluginManager.discover_plugins()
    ↓
Load Plugin Configurations
    ↓
Initialize Plugins
    ├── plugin.initialize()
    ├── plugin.register_settings()
    └── plugin.get_dependencies()
    ↓
Runtime Loop
    ├── plugin.update(data)
    ├── plugin.draw(screen)
    └── plugin.handle_event(event)
    ↓
Application Shutdown
    └── plugin.cleanup()
```

### Configuration Flow (Phase 7)
```
User Action
    ↓
SettingsPanel.handle_input()
    ↓
ConfigurationManager.update()
    ├── Validate Changes
    ├── Apply to Modules
    └── Save to Disk
    ↓
PresetManager (optional)
    ├── Save as Preset
    └── Load Preset
    ↓
State Persistence
    └── Auto-save on Exit
```

## Key Design Principles

1. **Separation of Concerns**
   - Audio processing separate from visualization
   - Business logic separate from UI
   - Configuration separate from implementation

2. **Dependency Injection**
   - Modules receive dependencies via constructor
   - No hard-coded dependencies between modules
   - Testable with mock objects

3. **Event-Driven Architecture**
   - Panels subscribe to data updates
   - Analyzers publish results
   - Loose coupling between components

4. **Plugin Architecture**
   - Core functionality in base modules
   - Extensions via plugin interface
   - Hot-reload capability for development

5. **Performance Optimization**
   - Multi-threaded audio processing
   - Lazy evaluation where possible
   - Caching of expensive computations

## Benefits of Final Architecture

1. **Maintainability**
   - Each module under 500 lines
   - Clear interfaces and contracts
   - Comprehensive test coverage

2. **Extensibility**
   - Add new panels via plugins
   - Custom analyzers without core changes
   - Theme and preset system

3. **Performance**
   - Optimized audio pipeline
   - Parallel processing where beneficial
   - Minimal coupling reduces overhead

4. **Usability**
   - User presets for common scenarios
   - Persistent settings
   - Plugin marketplace potential

5. **Development Experience**
   - Clear module boundaries
   - Hot-reload for rapid development
   - Comprehensive documentation