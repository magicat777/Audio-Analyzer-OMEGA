# Performance Optimization Plan for OMEGA-4

## Current Performance
- **FPS**: 40-45 at 512 bars (target: 60 FPS)
- **Bottlenecks**: FFT processing, drawing operations, panel updates

## Optimization Strategies

### 1. FFT Optimization
```python
# Current: Multiple FFT calls per frame
# Optimization: Cache FFT results between frames if audio hasn't changed significantly

class OptimizedMultiResolutionFFT:
    def __init__(self):
        self.fft_cache = {}
        self.last_audio_hash = None
    
    def process_with_cache(self, audio_chunk):
        # Quick hash of audio data
        audio_hash = hash(audio_chunk.tobytes()[::64])  # Sample every 64th byte
        
        if audio_hash == self.last_audio_hash:
            return self.fft_cache['result']
        
        # Compute FFT only if audio changed
        result = self._compute_fft(audio_chunk)
        self.fft_cache['result'] = result
        self.last_audio_hash = audio_hash
        return result
```

### 2. Drawing Optimization
```python
# Use pygame's batch drawing capabilities
def draw_spectrum_bars_optimized(self, bars_data):
    # Instead of individual rect draws:
    # pygame.draw.rect(screen, color, rect)
    
    # Use batch drawing with a single surface:
    bars_surface = pygame.Surface((width, height))
    
    # Or use pygame.draw.rects() for multiple rectangles:
    rects_and_colors = [(rect, color) for rect, color in bars_data]
    for rect, color in rects_and_colors:
        pygame.draw.rect(bars_surface, color, rect)
    
    screen.blit(bars_surface, (0, 0))
```

### 3. Numpy Optimization
```python
# Current: Multiple operations on arrays
magnitude = np.abs(fft_data)
normalized = magnitude / np.max(magnitude)

# Optimized: Combine operations
# Use out parameter to avoid allocation
np.abs(fft_data, out=magnitude_buffer)
np.divide(magnitude_buffer, magnitude_buffer.max(), out=normalized_buffer)
```

### 4. Threading/Async Processing
```python
# Move heavy computations to background threads
import concurrent.futures

class AsyncPanelProcessor:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.futures = {}
    
    def update_panels_async(self, data):
        # Submit panel updates to thread pool
        self.futures['harmonic'] = self.executor.submit(
            self.harmonic_panel.update, data
        )
        self.futures['pitch'] = self.executor.submit(
            self.pitch_panel.update, data
        )
        
    def get_results(self):
        # Collect results when needed for drawing
        results = {}
        for name, future in self.futures.items():
            if future.done():
                results[name] = future.result()
        return results
```

### 5. Dirty Rectangle Optimization
```python
# Only redraw changed portions
class DirtyRectManager:
    def __init__(self):
        self.dirty_rects = []
        self.last_frame = {}
    
    def mark_dirty(self, rect):
        self.dirty_rects.append(rect)
    
    def update_display(self):
        if self.dirty_rects:
            pygame.display.update(self.dirty_rects)
            self.dirty_rects.clear()
        else:
            pygame.display.update()  # Full update as fallback
```

### 6. Pre-computation and Caching
```python
# Pre-compute frequency mappings
class FrequencyMapper:
    def __init__(self, num_bars, sample_rate):
        # Pre-compute all frequency-to-bar mappings
        self.freq_to_bar = self._precompute_mappings(num_bars, sample_rate)
        self.mel_scale_factors = self._precompute_mel_scale()
        self.compensation_curve = self._precompute_compensation()
    
    def map_spectrum_fast(self, spectrum):
        # Use pre-computed mappings instead of calculating each frame
        return spectrum[self.freq_to_bar] * self.compensation_curve
```

### 7. Reduce Panel Update Frequency
```python
# Update panels at different rates based on their needs
class AdaptiveUpdater:
    def __init__(self):
        self.update_counters = {
            'spectrum': 1,      # Every frame
            'meters': 2,        # Every 2 frames (30 FPS)
            'harmonic': 3,      # Every 3 frames (20 FPS)
            'genre': 6,         # Every 6 frames (10 FPS)
        }
        self.frame_count = 0
    
    def should_update(self, panel_name):
        divisor = self.update_counters.get(panel_name, 1)
        return self.frame_count % divisor == 0
```

### 8. Memory Pool for Arrays
```python
# Reuse arrays instead of allocating new ones
class ArrayPool:
    def __init__(self):
        self.pools = {}
    
    def get_array(self, shape, dtype=np.float32):
        key = (shape, dtype)
        if key not in self.pools:
            self.pools[key] = []
        
        if self.pools[key]:
            return self.pools[key].pop()
        else:
            return np.empty(shape, dtype=dtype)
    
    def return_array(self, array):
        key = (array.shape, array.dtype)
        self.pools[key].append(array)
```

### 9. Numba JIT Compilation
```python
# Use Numba for critical loops
from numba import jit

@jit(nopython=True, cache=True)
def compute_band_values_fast(spectrum, band_indices, num_bars):
    band_values = np.zeros(num_bars)
    for i in range(num_bars):
        start, end = band_indices[i]
        if end > start:
            band_values[i] = np.mean(spectrum[start:end])
    return band_values
```

### 10. Profile-Guided Optimization
```python
# Add profiling to identify actual bottlenecks
import cProfile
import pstats

def profile_frame():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Process one frame
    process_audio_spectrum()
    draw_everything()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 time consumers
```

## Implementation Priority

1. **High Impact, Low Effort**:
   - Pre-compute frequency mappings
   - Reduce panel update frequencies
   - Cache FFT results when audio is silent/static

2. **High Impact, Medium Effort**:
   - Implement array pooling
   - Optimize numpy operations
   - Batch drawing operations

3. **High Impact, High Effort**:
   - Threading for panel updates
   - Dirty rectangle optimization
   - Numba compilation for hot paths

## Expected Performance Gains

- Pre-computation: +5-10 FPS
- Reduced updates: +10-15 FPS
- Array pooling: +5-8 FPS
- Threading: +10-20 FPS
- **Total potential**: 60+ FPS at 512 bars

## Quick Wins to Implement First

1. Reduce genre classification to 10 FPS (every 6 frames)
2. Pre-compute all frequency mappings
3. Cache FFT results for silent periods
4. Use array pooling for temporary arrays
5. Batch rectangle drawing operations