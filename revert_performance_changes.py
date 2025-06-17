#!/usr/bin/env python3
"""
Revert the performance optimization changes that reduced update frequency
"""

import re

# Read the current file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup current state
with open('omega4_main_with_optimization.py', 'w') as f:
    f.write(content)

# Remove the frame counter
content = re.sub(r'\s*self\.frame_counter = 0\s*#.*panel updates.*\n', '', content)

# Remove the _should_skip_panel_update method
content = re.sub(
    r'def _should_skip_panel_update\(self, panel_name\):.*?return False.*?\n',
    '',
    content,
    flags=re.DOTALL
)

# Remove all the skip checks from panel updates
skip_patterns = [
    r'if not self\._should_skip_panel_update\(\'harmonic_analysis\'\):\s*\n\s+',
    r'if not self\._should_skip_panel_update\(\'pitch_detection\'\):\s*\n\s+',
    r'if not self\._should_skip_panel_update\(\'chromagram\'\):\s*\n\s+',
    r'if not self\._should_skip_panel_update\(\'genre_classification\'\):\s*\n\s+',
    r'if not self\._should_skip_panel_update\(\'voice_detection\'\):\s*\n\s+',
    r'if not self\._should_skip_panel_update\(\'phase_correlation\'\):\s*\n\s+',
]

for pattern in skip_patterns:
    content = re.sub(pattern, '', content)

# Remove frame counter increment
content = re.sub(
    r'\s*# Increment frame counter\s*\n\s*self\.frame_counter \+= 1\s*\n',
    '',
    content
)

# Write the reverted file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Reverted performance optimization changes")
print("All panels will now update every frame as before")
print("Backup saved as omega4_main_with_optimization.py")