#!/usr/bin/env python3
"""
Remove Integrated Music Panel from the codebase
"""

import re

# Read the main file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_integrated.py', 'w') as f:
    f.write(content)

# Remove import
content = re.sub(r'from omega4\.panels\.integrated_music_panel import IntegratedMusicPanel\n', '', content)

# Remove initialization
content = re.sub(r'\s*self\.integrated_music_panel = IntegratedMusicPanel\(SAMPLE_RATE\)\n', '', content)

# Remove font setting
content = re.sub(r'\s*self\.integrated_music_panel\.set_fonts\(\{[^}]+\}\)\n', '', content, flags=re.DOTALL)

# Remove show flag
content = re.sub(r'\s*self\.show_integrated_music = False.*\n', '', content)

# Remove frozen flags and comments
content = re.sub(r'\s*self\.frozen_chromagram = False\s*# Not used - integrated panel preferred\n', '        self.frozen_chromagram = False\n', content)
content = re.sub(r'\s*self\.frozen_genre_classification = False\s*# Not used - integrated panel preferred\n', '        self.frozen_genre_classification = False\n', content)
content = re.sub(r'\s*self\.frozen_integrated_music = False\n', '', content)

# Remove panel registration
content = re.sub(r"\s*self\.panel_canvas\.register_panel\('integrated_music', self\.integrated_music_panel\)\n", '', content)

# Remove update manager registration
content = re.sub(r"\s*self\.panel_update_manager\.register_panel\('integrated_music',\s*\n\s*lambda.*?frozen_integrated_music else None\)\n", '', content, flags=re.DOTALL)

# Remove FFT preparation section
fft_section = r"# Integrated music panel FFT \(if active\)\s*\n\s*music_fft_id = None\s*\n\s*if \('integrated_music'.*?\n.*?music_fft_id = self\.batched_fft\.prepare_batch.*?\n"
content = re.sub(fft_section, '', content, flags=re.DOTALL)

# Remove the update section
update_section = r"# Update integrated music panel if active\s*\n\s*if \('integrated_music'.*?\n.*?self\.integrated_music_panel\.update\([^)]+\)\n"
content = re.sub(update_section, '', content, flags=re.DOTALL)

# Remove debug output section
debug_section = r"if 'integrated_music' in visible_panels:.*?\n.*?print\(f\"Key: \{self\.integrated_music_panel\.current_key\}\"\)"
content = re.sub(debug_section, '', content, flags=re.DOTALL)

# Remove keyboard handling for 'I' key
content = re.sub(r'elif event\.key == pygame\.K_i:.*?self\._toggle_panel\(\'integrated_music\', \'Integrated Music Analysis\'\)\n', '', content, flags=re.DOTALL)

# Write the updated file
with open('omega4_main.py', 'w') as f:
    f.write(content)

# Update the panels __init__.py
try:
    with open('omega4/panels/__init__.py', 'r') as f:
        init_content = f.read()
    
    # Remove import
    init_content = re.sub(r'from \.integrated_music_panel import IntegratedMusicPanel\n', '', init_content)
    
    # Remove from __all__
    init_content = re.sub(r"'IntegratedMusicPanel',\s*\n", '', init_content)
    
    with open('omega4/panels/__init__.py', 'w') as f:
        f.write(init_content)
    
    print("Updated omega4/panels/__init__.py")
except Exception as e:
    print(f"Could not update __init__.py: {e}")

print("Removed Integrated Music Panel references from omega4_main.py")
print("Backup saved as omega4_main_backup_integrated.py")