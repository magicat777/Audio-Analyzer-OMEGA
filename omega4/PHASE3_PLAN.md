# Phase 3 Plan: Complex Panel Extraction

## Overview
Phase 3 focuses on extracting complex panels that contain both display logic and business logic. These are self-contained modules that can be extracted as complete units.

## Target Panels for Extraction

### 1. Professional Meters Panel (Highest Priority)
- **Complexity**: High
- **Dependencies**: Audio processing, LUFS calculation, K-weighting
- **Size**: ~400 lines
- **Business Logic**: LUFS/RMS/Peak calculations, K-weighting filters
- **Display Logic**: Meter rendering, peak indicators

### 2. VU Meters Panel
- **Complexity**: Medium
- **Dependencies**: Audio level processing
- **Size**: ~250 lines
- **Business Logic**: VU ballistics simulation
- **Display Logic**: Classic VU meter rendering

### 3. Bass Zoom Window
- **Complexity**: Medium
- **Dependencies**: Spectrum data, frequency mapping
- **Size**: ~300 lines
- **Business Logic**: Zoom frequency calculations
- **Display Logic**: Detailed bass spectrum rendering

### 4. Harmonic Analysis Panel
- **Complexity**: Very High
- **Dependencies**: FFT data, harmonic detection algorithms
- **Size**: ~500 lines
- **Business Logic**: Harmonic series detection, instrument identification
- **Display Logic**: Harmonic visualization

### 5. Pitch Detection Panel (OMEGA)
- **Complexity**: High
- **Dependencies**: Cepstral analysis, pitch algorithms
- **Size**: ~400 lines
- **Business Logic**: Multiple pitch detection methods
- **Display Logic**: Musical note display, pitch tracking

### 6. Chromagram Panel (OMEGA-1)
- **Complexity**: High
- **Dependencies**: Chroma extraction, key detection
- **Size**: ~350 lines
- **Business Logic**: Chromagram calculation, key detection
- **Display Logic**: Chroma visualization, key display

### 7. Genre Classification Panel (OMEGA-2)
- **Complexity**: Very High
- **Dependencies**: Feature extraction, ML inference
- **Size**: ~400 lines
- **Business Logic**: Genre classification logic
- **Display Logic**: Genre probability display

## Extraction Strategy

### Module Structure
Each panel will be extracted as a complete module:
```
omega4/
├── panels/
│   ├── __init__.py
│   ├── professional_meters.py
│   ├── vu_meters.py
│   ├── bass_zoom.py
│   ├── harmonic_analysis.py
│   ├── pitch_detection.py
│   ├── chromagram.py
│   └── genre_classification.py
```

### Interface Pattern
Each panel module will follow this pattern:
```python
class PanelName:
    def __init__(self, config, display_params):
        # Initialize panel
        
    def update(self, audio_data, fft_data, other_inputs):
        # Update calculations
        
    def draw(self, screen, position, size):
        # Render panel
        
    def get_results(self):
        # Return analysis results
```

### Extraction Order
1. **Professional Meters** - Most self-contained, good starting point
2. **VU Meters** - Similar to professional meters, builds on pattern
3. **Bass Zoom** - Simpler visualization, good next step
4. **Harmonic Analysis** - Complex but well-isolated
5. **Pitch Detection** - OMEGA feature, complex algorithms
6. **Chromagram** - OMEGA-1 feature, builds on pitch
7. **Genre Classification** - OMEGA-2, most complex

## Migration Steps for Each Panel

### Step 1: Create Module Structure
- Create new file in `omega4/panels/`
- Define class with standard interface
- Import necessary dependencies

### Step 2: Move Business Logic
- Extract calculation methods
- Move algorithm implementations
- Preserve all functionality

### Step 3: Move Display Logic
- Extract rendering methods
- Use display interface where possible
- Maintain visual consistency

### Step 4: Update Main File
- Replace old code with module instantiation
- Update calls to use new interface
- Test thoroughly

### Step 5: Create Panel Tests
- Unit tests for calculations
- Integration tests for rendering
- Performance tests if needed

## Success Criteria
- All panels work exactly as before
- Clean module boundaries
- Reduced main file size by 50%+
- Improved testability
- Better code organization

## Risk Mitigation
- Test after each panel extraction
- Keep old code commented until verified
- Create integration tests early
- Document any behavior changes

## Expected Outcomes
- Main file reduced from ~7000 to ~3500 lines
- 7 new panel modules
- Improved maintainability
- Easier to add new panels
- Better separation of concerns