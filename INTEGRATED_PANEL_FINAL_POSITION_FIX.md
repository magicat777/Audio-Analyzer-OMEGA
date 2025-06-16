# Integrated Panel Final Position and Height Fix

## Changes Made

### 1. Position Fix
The integrated music panel is now forced to appear in column 1 (under Harmonic Analysis) instead of column 0:

```python
if panel_type == 'integrated':
    # Force integrated panel to column 1 (under Harmonic Analysis)
    col = 1
    panel_x = grid_start_x + col * (panel_width + panel_padding)
```

### 2. Height Fix
The integrated panel now gets an extra 50px of height (310px total instead of 260px) to accommodate the chromagram display with labels:

```python
actual_panel_height = panel_height
if panel_type == 'integrated':
    # Increase height for integrated panel to fit chromagram
    actual_panel_height = panel_height + 50  # 310px total
```

### 3. Window Height Calculation
The window height calculation now accounts for the extra height of the integrated panel:

```python
# Add extra height if integrated panel is shown
if self.show_integrated_music:
    tech_panels_height += 50  # Extra 50px for integrated panel
```

## Result

When the integrated panel is active (press 'I'):

### Panel Layout:
```
Row 0: [Meters] [Harmonic] [Pitch] [Room Modes]
Row 1: [      ] [Integrated Music Analysis    ]
             ^-- Column 1 (under Harmonic)
```

### Panel Heights:
- Regular panels: 260px
- Integrated panel: 310px (extra 50px for chromagram with labels)

## Why These Changes Were Needed

1. **Position**: The integrated panel was appearing in the leftmost position (column 0) which didn't look as good visually. Placing it under Harmonic Analysis (column 1) creates better visual balance.

2. **Height**: The chromagram visualization includes note labels that are drawn 15px below the chromagram bars. With the original 260px height, these labels were being drawn outside the panel boundaries. The extra 50px ensures all content fits within the panel.

## Technical Details

- The integrated panel's column is forced to 1 regardless of its position in the active_panels array
- The `actual_panel_height` variable is used throughout drawing and frozen overlay to ensure consistent sizing
- The window automatically adjusts its height when the integrated panel is toggled on/off