# OMEGA-4 Migration Principles

## Core Strategy: Risk Minimization Through Incremental Change

### 1. Extract Config First ✓ COMPLETE
**Why**: Least risky, highest value
- No logic changes, just moving constants
- Immediately improves maintainability
- Easy to verify correctness
- Sets foundation for future changes

### 2. Extract Display Layer Second 🔄 NEXT
**Why**: Clear interface boundary
- Display only needs data, not processing logic
- Natural separation point
- Easy to test visually
- Reduces main file complexity significantly

### 3. Add Integration Tests Before Audio Processing
**Why**: Safety net for complex changes
```python
# Example integration test structure
def test_audio_pipeline():
    # Test data flows correctly through:
    # Audio capture → FFT → Spectrum mapping → Display
    
def test_analyzer_pipeline():
    # Test each analyzer receives correct data:
    # Audio → FFT → Analyzer → Results → Display
    
def test_configuration_propagation():
    # Test config changes affect all components correctly
```

### 4. Extract One Analyzer at a Time
**Why**: Maintain working state
- Start with simplest analyzer (e.g., DrumDetector)
- Full testing between each extraction
- Learn from each extraction
- Can rollback individual changes

## Extraction Order (Recommended)

### Phase 1: Configuration ✓
- Status: COMPLETE
- Risk: Minimal
- Value: High

### Phase 2: Display Layer
- Risk: Low (clear interface)
- Value: High (major complexity reduction)
- Approach:
  1. Create display interface
  2. Move background/grid rendering
  3. Move spectrum bars
  4. Move text overlays
  5. Move panels one by one

### Phase 3: Integration Tests
- Risk: None (only adding tests)
- Value: Critical for safety
- Coverage needed:
  - Data flow tests
  - Configuration tests
  - Performance benchmarks
  - Visual regression tests

### Phase 4: Simple Analyzers
Order by complexity (simplest first):
1. DrumDetector (simple threshold detection)
2. VoiceAnalyzer (if using external module)
3. HarmonicAnalyzer (moderate complexity)
4. GenreClassifier (complex, many dependencies)

### Phase 5: Audio Infrastructure
- Risk: High (core functionality)
- Approach: Only after everything else works
- Components:
  - Audio capture
  - FFT processing
  - Spectrum mapping

## Testing Strategy

### After Each Extraction:
1. **Functional Test**: Does it still work?
2. **Performance Test**: Is it still fast?
3. **Visual Test**: Does it look the same?
4. **Integration Test**: Do all parts communicate?

### Test Commands:
```bash
# Quick functional test
python3 test_omega4.py

# Visual test (manual)
python3 run_omega4.py --bars 1024

# Performance test
time python3 run_omega4.py --bars 1024 --test-duration 60

# Integration tests (to be created)
python3 -m pytest omega4/tests/test_integration.py
```

## Red Flags to Avoid

### From OMEGA-3 Experience:
- ❌ Changing multiple things at once
- ❌ Refactoring while extracting
- ❌ Creating complex abstractions too early
- ❌ Losing track of data flow
- ❌ Not testing after each change

### Best Practices:
- ✅ One change at a time
- ✅ Test immediately after each change
- ✅ Keep interfaces simple
- ✅ Document data flow
- ✅ Commit working states frequently

## Success Metrics

Each phase is complete when:
1. All tests pass
2. No visual differences
3. No performance degradation
4. Code is cleaner than before
5. Next person can understand it

## Emergency Rollback Plan

If something breaks:
1. Git reset to last working commit
2. Smaller extraction attempt
3. Add more tests first
4. Consider leaving complex parts in main

Remember: **A working monolith is better than a broken modular system**