# OMEGA-4 Professional Audio Analyzer

A professional-grade real-time audio spectrum analyzer with studio-quality features and multi-resolution FFT analysis.

## Quick Start

```bash
# Run the main application
python omega4_main.py

# Or use the convenience script
./run_omega4.py
```

## Features

- **Multi-resolution FFT** for enhanced bass detail
- **Professional metering** (LUFS, K-weighting, True Peak)
- **Music analysis** with genre classification and key detection
- **Advanced visualizations** including harmonic analysis, pitch detection, and more
- **Adaptive performance** optimization for smooth 60 FPS operation

## Directory Structure

```
OMEGA/
├── omega4_main.py          # Main application
├── run_omega4.py           # Convenience launcher
├── omega4/                 # Core modules
│   ├── analyzers/          # Audio analysis algorithms
│   ├── audio/              # Audio capture and processing
│   ├── config/             # Configuration management
│   ├── optimization/       # Performance optimizations
│   ├── panels/             # Visualization panels
│   ├── plugins/            # Plugin system
│   ├── ui/                 # User interface
│   └── visualization/      # Display and rendering
├── tests/                  # Test files
├── docs/                   # Documentation
├── screenshots/            # Screenshot captures
├── archive/                # Old versions (git-ignored)
└── standalone_modules/     # Standalone panel implementations
```

## Key Bindings

See [docs/OMEGA4_QUICK_REFERENCE.md](docs/OMEGA4_QUICK_REFERENCE.md) for complete keyboard shortcuts.

### Essential Controls
- `Space` - Toggle all panels
- `1-8` - Window size presets
- `M` - Professional meters
- `I` - Integrated music analysis
- `S` - Screenshot
- `ESC/Q` - Quit

## Documentation

- [Program Summary](docs/OMEGA4_PROGRAM_SUMMARY.md) - Detailed feature overview
- [Architecture](docs/OMEGA4_ARCHITECTURE.md) - System design and data flow
- [Panel Status](docs/PANEL_STATUS_REPORT.md) - Current functionality status
- [Quick Reference](docs/OMEGA4_QUICK_REFERENCE.md) - Keyboard shortcuts

## Requirements

- Python 3.8+
- PipeWire audio system
- See requirements.txt for Python dependencies

## Development

The project follows a modular architecture. Each panel is self-contained and can be developed independently. See the architecture documentation for details on adding new features.

## License

This project is part of the audio-geometric-visualizer suite.