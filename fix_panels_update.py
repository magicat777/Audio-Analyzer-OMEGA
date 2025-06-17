#!/usr/bin/env python3
"""
Fix VU meters and bass zoom panels update issue
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_panels.py', 'w') as f:
    f.write(content)

# Fix 1: Make sure all lambda functions in panel registration accept **kwargs
# This ensures compatibility with the batch update system

# Fix VU meters lambda
content = re.sub(
    r"lambda \*args, \*\*kwargs: self\.vu_meters_panel\.update\(\*args\) if not self\.frozen_vu_meters else None",
    "lambda *args, **kwargs: self.vu_meters_panel.update(*args, **kwargs) if not self.frozen_vu_meters else None",
    content
)

# Fix bass zoom lambda  
content = re.sub(
    r"lambda \*args, \*\*kwargs: self\.bass_zoom_panel\.update\(\*args\) if not self\.frozen_bass_zoom else None",
    "lambda *args, **kwargs: self.bass_zoom_panel.update(*args, **kwargs) if not self.frozen_bass_zoom else None",
    content
)

# Fix 2: Add direct updates for VU meters and bass zoom after batch updates
# Find the line after batch_update_panels and add direct updates

batch_update_pattern = r'(update_results = self\.panel_update_manager\.batch_update_panels\(panel_updates\)\s*\n)'

additional_updates = '''
        # Direct updates for non-canvas panels (VU meters and bass zoom)
        # These are critical panels that should always update when visible
        if 'vu_meters' in panel_updates and not self.frozen_vu_meters:
            try:
                args, kwargs = panel_updates['vu_meters']
                self.vu_meters_panel.update(*args, **kwargs)
                update_results['vu_meters'] = True
            except Exception as e:
                print(f"Error updating VU meters: {e}")
                
        if 'bass_zoom' in panel_updates and not self.frozen_bass_zoom:
            try:
                args, kwargs = panel_updates['bass_zoom']
                self.bass_zoom_panel.update(*args, **kwargs)
                update_results['bass_zoom'] = True
            except Exception as e:
                print(f"Error updating bass zoom: {e}")
'''

content = re.sub(batch_update_pattern, r'\1' + additional_updates + '\n', content)

# Write the fixed file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Fixed VU meters and bass zoom panel updates")
print("Changes made:")
print("1. Updated lambda functions to accept **kwargs")
print("2. Added direct update calls for VU meters and bass zoom panels")