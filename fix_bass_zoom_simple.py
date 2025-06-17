#!/usr/bin/env python3
"""
Simple fix for bass zoom - always add it to panel_updates when visible
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_bass_fix.py', 'w') as f:
    f.write(content)

# Replace the conditional check with a simpler one
# Change from checking if NOT in panel_updates to just adding it when show_bass_zoom is True

old_pattern = r'''            # Pass audio data and drum info \(only if not frozen and adaptive updater allows\)
            # Bass zoom is already in panel_updates if visible
            if self\.show_debug:
                print\(f"\[DEBUG\] Bass zoom check: show_bass_zoom=\{self\.show_bass_zoom\}, in panel_updates=\{'bass_zoom' in panel_updates\}"\)
            if 'bass_zoom' not in panel_updates:
                panel_updates\['bass_zoom'\] = \(\(audio_windowed, drum_info\), \{\}\)'''

new_pattern = '''            # Pass audio data and drum info (only if not frozen and adaptive updater allows)
            # Always add bass zoom to panel_updates when visible
            if self.show_debug:
                print(f"[DEBUG] Bass zoom check: show_bass_zoom={self.show_bass_zoom}, drum_info keys={list(drum_info.keys())}")
            panel_updates['bass_zoom'] = ((audio_windowed, drum_info), {})'''

content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)

# Also ensure bass zoom is activated in the panel update manager
# Add activation after the visible panels loop
activation_pattern = r'(# Activate panels that are visible\s+for panel_id in self\.panel_canvas\.get_visible_panels\(\):\s+self\.panel_update_manager\.activate_panel\(panel_id\))'

additional_activation = r'''\1
        
        # Also activate non-canvas panels that are always visible
        if self.show_bass_zoom:
            self.panel_update_manager.activate_panel('bass_zoom')
        if self.show_vu_meters:
            self.panel_update_manager.activate_panel('vu_meters')'''

content = re.sub(activation_pattern, additional_activation, content)

# Write the updated file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Fixed bass zoom panel updates:")
print("1. Removed conditional check - always add to panel_updates when show_bass_zoom is True")
print("2. Added activation for bass_zoom and vu_meters in panel update manager")