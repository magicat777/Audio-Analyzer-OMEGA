#!/usr/bin/env python3
"""
Clean up debug output to be less spammy
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_debug.py', 'w') as f:
    f.write(content)

# Remove or comment out the verbose debug statements that were added for troubleshooting

# 1. Remove the bass zoom debug statements that run every frame
debug_patterns = [
    # Bass zoom check debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Bass zoom check: show_bass_zoom=\{self\.show_bass_zoom\}, drum_info keys=\{list\(drum_info\.keys\(\)\)\}"\)',
    
    # Added to panel_updates debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Added bass_zoom to panel_updates\. Total panels to update: \{len\(panel_updates\)\}"\)',
    
    # Before direct updates debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Before direct updates: panel_updates contains: \{list\(panel_updates\.keys\(\)\)\}"\)',
    
    # Direct bass zoom update debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Direct bass zoom update: panel_updates keys=\{list\(panel_updates\.keys\(\)\)\}"\)',
    
    # Bass zoom args debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Bass zoom args: \{len\(args\)\} args, drum_info keys: \{list\(args\[1\]\.keys\(\)\) if len\(args\) > 1 and isinstance\(args\[1\], dict\) else \'No drum_info\'\}"\)',
    
    # Bass zoom updated successfully debug
    r'if self\.show_debug:\s*\n\s*print\(f"\[DEBUG\] Bass zoom updated successfully"\)',
]

for pattern in debug_patterns:
    content = re.sub(pattern, '', content)

# Write the cleaned file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Cleaned up debug output")
print("Removed verbose per-frame debug statements")
print("The [BASS_ZOOM] panel debug messages and debug snapshots will still appear when 'D' is pressed")