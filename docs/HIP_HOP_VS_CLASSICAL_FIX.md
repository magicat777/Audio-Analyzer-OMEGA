# Hip-Hop vs Classical Misclassification Fix

## Problem
The integrated music analysis was incorrectly detecting hip-hop music as classical.

## Root Causes
1. **Overlapping feature ranges**: Classical and hip-hop had similar spectral centroid ranges (800-2000Hz vs 800-2500Hz)
2. **Missing feature checks**: Harmonic features weren't being used in classification
3. **No genre-specific penalties**: Classical wasn't penalized for having hip-hop characteristics
4. **Simple harmonic complexity**: Calculated as `len(harmonic_series) / 10.0` which didn't differentiate well

## Solutions Implemented

### 1. Added Harmonic Feature Checks
Now checking all harmonic features in the classification:
- `harmonic_complexity` (weight: 0.8)
- `chord_change_rate` (weight: 0.6)  
- `key_stability` (weight: 0.5)

### 2. Classical-Specific Penalties
Classical music is now penalized for having hip-hop characteristics:
- **High bass emphasis** (>0.5): Penalty proportional to bass level
- **High percussion** (>0.6): -0.6 penalty
- **Strong vocal presence** (>0.7): -0.5 penalty

### 3. Hip-Hop Enhancements
- **Increased bass emphasis weight** from 1.2 to 1.5
- **Synergy bonus**: +0.5 bonus when 3+ hip-hop features are present together
- **Feature counting**: Tracks bass emphasis, beat regularity, percussion, and key stability

### 4. Adjusted Feature Ranges
#### Classical (to reduce overlap with hip-hop):
- Spectral centroid: 800-2000 → **1200-3000** Hz (higher frequencies)
- Spectral rolloff: 2000-4000 → **3000-6000** Hz  
- Percussion strength: 0.0-0.3 → **0.0-0.2** (less percussion)
- Harmonic complexity: 0.3-0.7 → **0.4-0.8** (more complex)
- Pitch class concentration: 0.5-0.8 → **0.4-0.7** (more chromatic)

### 5. Improved Harmonic Complexity Calculation
Instead of simple division by 10, now uses stepped ranges:
- ≤3 harmonics: 0.1 (hip-hop/electronic)
- 4-5 harmonics: 0.3 (pop/rock)
- 6-8 harmonics: 0.5 (jazz/classical)
- >8 harmonics: 0.7 (classical/jazz)

## Expected Results

### Hip-Hop Will Score Higher When:
- Strong bass emphasis (>0.7)
- Regular beats (>0.7)
- High percussion (>0.85)
- Simple harmonics (<0.3)
- Static harmony (key stability >0.9)
- Multiple features present together (synergy bonus)

### Classical Will Score Lower When:
- Too much bass (penalized)
- Too much percussion (penalized)
- Pop/rap vocals present (penalized)
- Spectral content too low (outside new range)

## Testing
The system should now:
1. Correctly identify hip-hop by its bass-heavy, beat-driven nature
2. Not confuse hip-hop with classical due to separated frequency ranges
3. Apply appropriate penalties when genres have wrong characteristics
4. Give hip-hop priority through higher weights and synergy bonuses