# OMEGA-4 Migration Quick Reference

## Completed Phases (✅)

### Phase 1: Configuration Extraction
- **File**: `omega4/config.py`
- **Status**: ✅ Complete
- **Lines Saved**: ~200

### Phase 2: Display Interface Extraction
- **File**: `omega4/visualization/display_interface.py`
- **Status**: ✅ Complete
- **Lines Saved**: ~1000

### Phase 3: Complex Panel Extraction
- **Directory**: `omega4/panels/`
- **Status**: ✅ Complete (7 panels)
- **Lines Saved**: ~2000+
- **Panels**:
  1. ✅ professional_meters.py
  2. ✅ vu_meters.py
  3. ✅ bass_zoom.py
  4. ✅ harmonic_analysis.py
  5. ✅ pitch_detection.py (OMEGA)
  6. ✅ chromagram.py (OMEGA-1)
  7. ✅ genre_classification.py (OMEGA-2)

## Upcoming Phases (📋)

### Phase 4: Extract Remaining Analyzers
- **Directory**: `omega4/analyzers/`
- **Modules to Extract**:
  - [ ] harmonic.py (HarmonicAnalyzer)
  - [ ] phase_coherence.py (PhaseCoherenceAnalyzer)
  - [ ] transient.py (TransientAnalyzer)
  - [ ] room_modes.py (RoomModeAnalyzer)
  - [ ] drum_detection.py (EnhancedKickDetector, etc.)
- **Estimated Lines**: ~800

### Phase 5: Extract Audio Pipeline
- **Directory**: `omega4/audio/`
- **Modules to Extract**:
  - [ ] capture.py (AudioCapture, AudioCaptureThread)
  - [ ] multi_resolution_fft.py (MultiResolutionFFT)
  - [ ] pipeline.py (AudioProcessingPipeline)
  - [ ] voice_detection.py (Voice detection integration)
- **Estimated Lines**: ~600

### Phase 6: Plugin Architecture
- **Directory**: `omega4/plugins/`
- **New Modules**:
  - [ ] base.py (Plugin interfaces)
  - [ ] manager.py (Plugin management)
  - [ ] config.py (Plugin configuration)
  - [ ] Example plugins
- **Estimated Lines**: ~400 (new code)

### Phase 7: Configuration & Persistence
- **Directory**: `omega4/config/`
- **New Modules**:
  - [ ] manager.py (Configuration management)
  - [ ] presets.py (User presets)
  - [ ] state.py (Persistent state)
- **UI Module**: `omega4/ui/settings.py`
- **Estimated Lines**: ~500 (new code)

## Quick Commands

### Run Tests
```bash
# Test specific panel
python test_phase3_step1.py  # Professional meters
python test_phase3_step2.py  # VU meters
python test_phase3_step3.py  # Bass zoom
python test_phase3_step4.py  # Harmonic analysis
python test_phase3_step5.py  # Pitch detection
python test_phase3_step6.py  # Chromagram
python test_phase3_step7.py  # Genre classification

# Run main application
python omega4_main.py
```

### Module Locations
```bash
# Configuration
omega4/config.py

# Display
omega4/visualization/display_interface.py

# Panels
omega4/panels/professional_meters.py
omega4/panels/vu_meters.py
omega4/panels/bass_zoom.py
omega4/panels/harmonic_analysis.py
omega4/panels/pitch_detection.py
omega4/panels/chromagram.py
omega4/panels/genre_classification.py
```

## Migration Checklist Template

For each analyzer/module extraction:

1. [ ] Create new module file in appropriate directory
2. [ ] Copy class(es) from main file
3. [ ] Update imports in new module
4. [ ] Add module docstring
5. [ ] Update main file imports
6. [ ] Replace instantiation in main file
7. [ ] Update method calls to use new module
8. [ ] Comment out old code in main file
9. [ ] Create test file
10. [ ] Run tests to verify functionality
11. [ ] Update documentation

## Common Issues and Solutions

### Import Errors
- Check relative imports: `from omega4.panels import PanelName`
- Ensure `__init__.py` exists in directories

### Missing Dependencies
- Check if module needs config import: `from omega4.config import SAMPLE_RATE`
- Import numpy/pygame where needed

### Font/UI Issues
- Ensure `set_fonts()` is called for new panels
- Pass `ui_scale` parameter to draw methods

### Performance Issues
- Profile before and after extraction
- Check for unnecessary data copies
- Maintain threading where it existed

## Success Metrics

### Phase 3 (Completed)
- ✅ Main file reduced from ~7000 to ~5000 lines
- ✅ 7 independent panel modules
- ✅ All tests passing
- ✅ No performance regression

### Target After Phase 7
- [ ] Main file under 1000 lines
- [ ] 15+ independent modules
- [ ] Plugin system operational
- [ ] Configuration persistence working
- [ ] 90%+ test coverage
- [ ] Load time under 2 seconds