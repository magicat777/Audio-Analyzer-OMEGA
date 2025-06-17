#!/usr/bin/env python3
"""
Add debug output to understand why bass zoom isn't updating
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_bass_debug.py', 'w') as f:
    f.write(content)

# Add debug output before bass zoom update
debug_code = '''
            # Pass audio data and drum info (only if not frozen and adaptive updater allows)
            # Bass zoom is already in panel_updates if visible
            if self.show_debug:
                print(f"[DEBUG] Bass zoom check: show_bass_zoom={self.show_bass_zoom}, in panel_updates={'bass_zoom' in panel_updates}")
            if 'bass_zoom' not in panel_updates:'''

content = re.sub(
    r"(\s+# Pass audio data and drum info.*\n\s+# Bass zoom is already in panel_updates if visible\n\s+if 'bass_zoom' not in panel_updates:)",
    debug_code,
    content
)

# Add debug in the direct update section
content = re.sub(
    r"(if 'bass_zoom' in panel_updates and not self\.frozen_bass_zoom:)",
    r"\1\n            if self.show_debug:\n                print(f\"[DEBUG] Direct bass zoom update: panel_updates keys={list(panel_updates.keys())}\")",
    content
)

# Add debug in bass zoom update try block
content = re.sub(
    r"(args, kwargs = panel_updates\['bass_zoom'\])",
    r"\1\n                if self.show_debug:\n                    print(f\"[DEBUG] Bass zoom args: {len(args)} args, drum_info keys: {list(args[1].keys()) if len(args) > 1 and isinstance(args[1], dict) else 'No drum_info'}\")",
    content
)

# Write the updated file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Added debug output for bass zoom panel")
print("Press 'D' while running to enable debug mode and see why bass zoom isn't updating")