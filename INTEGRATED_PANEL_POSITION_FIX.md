# Integrated Panel Position Fix Summary

## Problem
The integrated music panel was overlapping with the genre classification panel because both were being drawn at the same time.

## Root Cause
When the user pressed 'I' to show the integrated panel, the code was:
1. Setting `show_integrated_music = True`
2. Setting `show_chromagram = False` and `show_genre_classification = False`
3. BUT the drawing code was checking these flags independently, causing both individual panels AND the integrated panel to be drawn

## Solution
Modified the panel collection logic to ensure chromagram and genre panels are only added to `active_panels` when the integrated panel is OFF:

```python
if self.show_chromagram and not self.show_integrated_music:  # Don't show if integrated is on
    active_panels.append(('chromagram', self.chromagram_panel))
if self.show_genre_classification and not self.show_integrated_music:  # Don't show if integrated is on
    active_panels.append(('genre', self.genre_classification_panel))
```

## Result
Now the panel layout works correctly:

### When Integrated Panel is OFF (default):
- Row 0: Meters | Harmonic | Pitch | Chromagram
- Row 1: Genre | Room Modes

### When Integrated Panel is ON (press 'I'):
- Row 0: Meters | Harmonic | Pitch | Room Modes
- Row 1: Integrated Music Analysis

The integrated panel now appears to the right of room modes (row=1, col=0) without any overlap.

## Additional Improvements
1. Simplified the keyboard handler - no need to manually toggle chromagram/genre flags
2. Added automatic window adjustment when toggling the integrated panel
3. The logic is now clearer and prevents any possibility of overlap