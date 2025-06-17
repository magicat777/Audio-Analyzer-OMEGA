#!/usr/bin/env python3
"""
Simple performance fix - add frame counters and skip updates for non-critical panels
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_perf.py', 'w') as f:
    f.write(content)

# Add frame counter after self.running = True
if 'self.frame_counter = 0' not in content:
    content = re.sub(
        r'(self\.running = True)',
        r'\1\n        self.frame_counter = 0  # For frame-based panel updates',
        content
    )

# Add simple update check method after __init__
simple_method = '''
    def _should_skip_panel_update(self, panel_name):
        """Skip updates for non-critical panels to improve performance"""
        # Skip intervals (update every N frames)
        skip_intervals = {
            'harmonic_analysis': 8,      # Update every 8 frames
            'genre_classification': 8,   
            'integrated_music': 16,      # Update every 16 frames
            'chromagram': 4,             # Update every 4 frames
            'pitch_detection': 4,
            'voice_detection': 4,
            'phase_correlation': 4,
        }
        
        if panel_name in skip_intervals:
            return self.frame_counter % skip_intervals[panel_name] != 0
        return False  # Don't skip critical panels
'''

# Insert the method after __init__
init_end = content.find('    def capture_audio(self):')
if init_end > 0 and '_should_skip_panel_update' not in content:
    content = content[:init_end] + simple_method + '\n' + content[init_end:]

# Wrap panel updates with skip checks
# For each panel update in the batch_update_panels section, add a skip check

# First, let's modify the panel_updates dictionary creation to check for skips
panel_patterns = [
    (r"(if 'harmonic_analysis' in self\.panel_canvas\.get_visible_panels\(\):)", 
     r"\1\n            if not self._should_skip_panel_update('harmonic_analysis'):"),
    (r"(if 'pitch_detection' in self\.panel_canvas\.get_visible_panels\(\):)",
     r"\1\n            if not self._should_skip_panel_update('pitch_detection'):"),
    (r"(if 'chromagram' in self\.panel_canvas\.get_visible_panels\(\):)",
     r"\1\n            if not self._should_skip_panel_update('chromagram'):"),
    (r"(if 'genre_classification' in self\.panel_canvas\.get_visible_panels\(\):)",
     r"\1\n            if not self._should_skip_panel_update('genre_classification'):"),
    (r"(if 'voice_detection' in self\.panel_canvas\.get_visible_panels\(\):)",
     r"\1\n            if not self._should_skip_panel_update('voice_detection'):"),
    (r"(if 'phase_correlation' in self\.panel_canvas\.get_visible_panels\(\):)",
     r"\1\n            if not self._should_skip_panel_update('phase_correlation'):"),
]

for pattern, replacement in panel_patterns:
    content = re.sub(pattern, replacement, content)

# Fix indentation for the panel_updates assignments
# Add extra indentation to the lines after our new if statements
content = re.sub(
    r"(if not self\._should_skip_panel_update\('.*?'\):\n)(\s+)(panel_updates\['.*?'\] = .*)",
    r'\1\2    \3',
    content
)

# Add frame counter increment at the end of the main loop
# Find the sleep time section and add after it
if 'self.frame_counter += 1' not in content:
    content = re.sub(
        r'(pygame\.time\.wait\(int\(sleep_time \* 1000\)\))',
        r'\1\n                \n                # Increment frame counter\n                self.frame_counter += 1',
        content
    )

# Write the updated file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Applied simple performance optimization with frame skipping")
print("Non-critical panels will update at reduced rates:")
print("- Chromagram, Pitch, Voice, Phase: 15 FPS (every 4 frames)")
print("- Harmonic, Genre Classification: 7.5 FPS (every 8 frames)") 
print("- Integrated Music: 3.75 FPS (every 16 frames)")