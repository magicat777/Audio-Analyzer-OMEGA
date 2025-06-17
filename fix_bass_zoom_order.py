#!/usr/bin/env python3
"""
Fix bass zoom by moving its update before batch_update_panels
"""

import re

# Read the file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Backup
with open('omega4_main_backup_order.py', 'w') as f:
    f.write(content)

# Find the bass zoom update section (it's currently after batch update)
bass_zoom_section = r'''        # Update bass zoom if active
        if self\.show_bass_zoom:
            # Extract drum info from drum events
            drum_info = \{\}
            if drum_events:
                for event in drum_events:
                    if isinstance\(event, dict\):
                        event_type = event\.get\('type', ''\)
                        magnitude = event\.get\('magnitude', 0\.0\)
                        if 'kick' in event_type\.lower\(\):
                            drum_info\['kick'\] = magnitude
                        elif 'sub' in event_type\.lower\(\):
                            drum_info\['sub'\] = magnitude
                        elif 'floor' in event_type\.lower\(\):
                            drum_info\['floor'\] = magnitude
                        elif 'bass' in event_type\.lower\(\):
                            drum_info\['bass'\] = magnitude
            
            # If no drum events, analyze bass frequencies directly
            if not drum_info and len\(spectrum_data\['spectrum'\]\) > 0:
                freq_bin_width = SAMPLE_RATE / \(2 \* len\(spectrum_data\['spectrum'\]\)\)
                
                # Analyze specific bass frequencies
                sub_idx = int\(40 / freq_bin_width\)
                kick_idx = int\(60 / freq_bin_width\)
                floor_idx = int\(80 / freq_bin_width\) 
                bass_idx = int\(110 / freq_bin_width\)
                
                if sub_idx < len\(spectrum_data\['spectrum'\]\):
                    drum_info\['sub'\] = spectrum_data\['spectrum'\]\[sub_idx\]
                if kick_idx < len\(spectrum_data\['spectrum'\]\):
                    drum_info\['kick'\] = spectrum_data\['spectrum'\]\[kick_idx\]
                if floor_idx < len\(spectrum_data\['spectrum'\]\):
                    drum_info\['floor'\] = spectrum_data\['spectrum'\]\[floor_idx\]
                if bass_idx < len\(spectrum_data\['spectrum'\]\):
                    drum_info\['bass'\] = spectrum_data\['spectrum'\]\[bass_idx\]
            # Pass audio data and drum info \(only if not frozen and adaptive updater allows\)
            # Always add bass zoom to panel_updates when visible
            if self\.show_debug:
                print\(f"\[DEBUG\] Bass zoom check: show_bass_zoom=\{self\.show_bass_zoom\}, drum_info keys=\{list\(drum_info\.keys\(\)\)\}"\)
            panel_updates\['bass_zoom'\] = \(\(audio_windowed, drum_info\), \{\}\)
            if self\.show_debug:
                print\(f"\[DEBUG\] Added bass_zoom to panel_updates\. Total panels to update: \{len\(panel_updates\)\}"\)'''

# Remove it from its current location
content = re.sub(bass_zoom_section, '', content, flags=re.DOTALL)

# Find where to insert it (before "Batch update all panels")
insert_location = r'(# Transient detection \(high priority\)\s+if \'transient_detection\' in self\.panel_canvas\.get_visible_panels\(\):\s+panel_updates\[\'transient_detection\'\] = \(\(audio_windowed, spectrum_data\[\'spectrum\'\]\), \{\}\)\s*\n)'

# Prepare the bass zoom code to insert
bass_zoom_to_insert = '''
        # Update bass zoom if active (needs to be before batch update)
        if self.show_bass_zoom:
            # Extract drum info from drum events
            drum_info = {}
            if drum_events:
                for event in drum_events:
                    if isinstance(event, dict):
                        event_type = event.get('type', '')
                        magnitude = event.get('magnitude', 0.0)
                        if 'kick' in event_type.lower():
                            drum_info['kick'] = magnitude
                        elif 'sub' in event_type.lower():
                            drum_info['sub'] = magnitude
                        elif 'floor' in event_type.lower():
                            drum_info['floor'] = magnitude
                        elif 'bass' in event_type.lower():
                            drum_info['bass'] = magnitude
            
            # If no drum events, analyze bass frequencies directly
            if not drum_info and len(spectrum_data['spectrum']) > 0:
                freq_bin_width = SAMPLE_RATE / (2 * len(spectrum_data['spectrum']))
                
                # Analyze specific bass frequencies
                sub_idx = int(40 / freq_bin_width)
                kick_idx = int(60 / freq_bin_width)
                floor_idx = int(80 / freq_bin_width) 
                bass_idx = int(110 / freq_bin_width)
                
                if sub_idx < len(spectrum_data['spectrum']):
                    drum_info['sub'] = spectrum_data['spectrum'][sub_idx]
                if kick_idx < len(spectrum_data['spectrum']):
                    drum_info['kick'] = spectrum_data['spectrum'][kick_idx]
                if floor_idx < len(spectrum_data['spectrum']):
                    drum_info['floor'] = spectrum_data['spectrum'][floor_idx]
                if bass_idx < len(spectrum_data['spectrum']):
                    drum_info['bass'] = spectrum_data['spectrum'][bass_idx]
            # Pass audio data and drum info
            if self.show_debug:
                print(f"[DEBUG] Bass zoom check: show_bass_zoom={self.show_bass_zoom}, drum_info keys={list(drum_info.keys())}")
            panel_updates['bass_zoom'] = ((audio_windowed, drum_info), {})
            if self.show_debug:
                print(f"[DEBUG] Added bass_zoom to panel_updates. Total panels to update: {len(panel_updates)}")
        
'''

# Insert the bass zoom code before batch update
content = re.sub(insert_location, r'\1' + bass_zoom_to_insert, content)

# Write the updated file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Fixed bass zoom panel update order")
print("Bass zoom is now added to panel_updates BEFORE batch_update_panels is called")