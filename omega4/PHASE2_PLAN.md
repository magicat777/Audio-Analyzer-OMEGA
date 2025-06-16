# OMEGA-4 Phase 2 Plan: Extract Display Layer

## Current State
- ✓ Phase 1 Complete: Configuration extracted
- Main application (`omega4_main.py`) is a working monolith with ~6917 lines
- All pygame rendering is mixed with audio processing logic

## Phase 2 Goals
Extract ONLY the display/rendering code while keeping all processing in main file.

## Identified Display Components

### 1. Pygame Initialization (found at main())
- Screen setup
- Font system
- Color generation
- Display caption

### 2. Rendering Functions to Extract
- Spectrum bar drawing
- Grid/scale drawing  
- Text overlays (stats, info)
- Panel backgrounds
- Peak indicators
- Frequency labels

### 3. What to Keep in Main
- All audio processing
- FFT calculations
- Analyzer logic
- Data structures
- Main loop control

## Implementation Plan

### Step 1: Create Display Interface
```python
# visualization/display_interface.py
class SpectrumDisplay:
    def __init__(self, screen, width, height, bars):
        self.screen = screen
        self.width = width
        self.height = height
        self.bars = bars
        self._setup_display()
    
    def render_frame(self, spectrum_data, analysis_results):
        """Single entry point for all rendering"""
        pass
```

### Step 2: Gradual Extraction
1. Start with the simplest rendering (background clear)
2. Add spectrum bars
3. Add grid and labels
4. Add overlay panels
5. Test after each addition

### Step 3: Integration Pattern
```python
# In main file
self.display = SpectrumDisplay(self.screen, width, height, self.bars)

# In main loop
self.display.render_frame(
    spectrum_data=self.bar_heights,
    analysis_results={
        'genre': genre_result,
        'voice': voice_result,
        'harmonic': harmonic_result
    }
)
```

## Success Criteria
- Application runs exactly as before
- No visual differences
- Same performance
- Display code is isolated from processing
- Simple interface between main and display

## Risk Mitigation
- Keep backup of working version
- Test after each small change
- Use git commits for each working state
- Don't refactor processing code yet