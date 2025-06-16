# Phase 2 Test Plan: Display Layer Extraction

## Pre-Extraction Baseline

### 1. Capture Screenshots
```bash
# Run and take screenshots of:
# - Main spectrum view
# - With genre detection active
# - With voice detection active  
# - All panels visible
# - Different window sizes
```

### 2. Performance Baseline
```bash
# Measure current performance
time python3 run_omega4.py --bars 1024 --benchmark
# Record: FPS, CPU usage, Memory usage
```

### 3. Functionality Checklist
- [ ] Spectrum bars display correctly
- [ ] Colors gradient properly
- [ ] Grid lines align with frequencies
- [ ] Frequency labels are accurate
- [ ] dB scale is correct
- [ ] Peak indicators work
- [ ] Text overlays readable
- [ ] Panels render properly
- [ ] Window resizing works

## During Extraction Tests

### After Each Component Move:

#### 1. Smoke Test
```python
# Quick test that it runs
python3 run_omega4.py --test-mode --duration 5
```

#### 2. Visual Comparison
- Run application
- Compare with baseline screenshots
- Check for:
  - Missing elements
  - Misaligned elements
  - Color differences
  - Font issues

#### 3. Data Flow Test
```python
# Verify display receives correct data
def test_display_data_flow():
    # Create mock data
    spectrum_data = np.random.rand(1024) * 0.5
    
    # Pass to display
    display.render_frame(spectrum_data, {})
    
    # Verify no crashes
    # Verify data used correctly
```

## Post-Extraction Validation

### 1. Complete Functional Test
Run through all features:
- [ ] Keyboard shortcuts work
- [ ] All panels toggle correctly
- [ ] Window resizing maintains layout
- [ ] No visual artifacts
- [ ] Smooth animation

### 2. Performance Test
```bash
# Should be same or better than baseline
python3 omega4/tests/benchmark_display.py
```

### 3. Code Quality Checks
- [ ] Display module under 1000 lines
- [ ] Clear interface with main
- [ ] No processing logic in display
- [ ] Well-documented methods
- [ ] No tight coupling

### 4. Integration Test Suite
```python
# omega4/tests/test_display_integration.py
class TestDisplayIntegration:
    def test_display_handles_empty_data(self):
        """Display should handle empty/zero data gracefully"""
        
    def test_display_handles_various_bar_counts(self):
        """Test with 256, 512, 1024, 2048 bars"""
        
    def test_display_updates_on_resize(self):
        """Window resize should update display correctly"""
        
    def test_display_performance(self):
        """Rendering should stay under 16ms"""
```

## Rollback Criteria

Rollback if any of these occur:
- Visual differences from baseline
- Performance degradation >10%
- Any feature stops working
- Code becomes more complex
- Tests become too difficult

## Success Criteria

Phase 2 is complete when:
- ✅ All tests pass
- ✅ No visual differences
- ✅ Same or better performance  
- ✅ Display code is isolated
- ✅ Main file reduced by 20-30%
- ✅ Clear, simple interface
- ✅ Easy to understand and modify