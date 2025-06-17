#!/usr/bin/env python3
"""
Simple frame-skipping optimization script
"""

import re

# Read the current file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Create backup
with open('omega4_main_pre_simple_opt.py', 'w') as f:
    f.write(content)

# Remove the complex optimization imports and initializations
content = re.sub(r'from omega4\.optimization\.panel_update_manager import.*\n', '', content)
content = re.sub(r'from omega4\.optimization\.render_optimization import.*\n', '', content)
content = re.sub(r'^\s*self\.panel_update_manager = .*\n', '', content, flags=re.MULTILINE)
content = re.sub(r'^\s*self\.render_optimizer = .*\n', '', content, flags=re.MULTILINE)

# Remove the _register_panel_updates method
content = re.sub(r'def _register_panel_updates\(self\):.*?(?=def\s|\Z)', '', content, flags=re.DOTALL)

# Add frame counter after self.running = True
content = re.sub(
    r'(self\.running = True)',
    r'\1\n        self.frame_counter = 0  # For simple frame-based panel updates',
    content
)

# Add simple panel update scheduling method
simple_update_method = '''
    def _should_update_panel(self, panel_name):
        """Simple frame-based update scheduling for performance"""
        # Update intervals (in frames)
        intervals = {
            'vu_meters': 1,            # Every frame (60 FPS)
            'professional_meters': 2,   # Every 2 frames (30 FPS)
            'bass_zoom': 2,            
            'transient_detection': 2,   
            'chromagram': 4,           # Every 4 frames (15 FPS)
            'pitch_detection': 4,      
            'voice_detection': 4,      
            'phase_correlation': 4,    
            'harmonic_analysis': 8,    # Every 8 frames (7.5 FPS)
            'genre_classification': 8, 
            'integrated_music': 16,    # Every 16 frames (3.75 FPS)
            'room_analysis': 16,       
        }
        
        interval = intervals.get(panel_name, 4)
        return self.frame_counter % interval == 0
'''

# Add the method after capture_audio
content = re.sub(
    r'(def capture_audio\(self\):.*?(?=def\s))',
    r'\1' + simple_update_method + '\n    ',
    content,
    flags=re.DOTALL
)

# Now modify the panel updates to use simple frame skipping
# Replace the batch_update_panels call with direct updates

# Find and replace the batch update section
batch_update_pattern = r'# Batch update all panels based on priority\s*\n\s*update_results = self\.panel_update_manager\.batch_update_panels\(panel_updates\)'

replacement = '''# Simple frame-based panel updates
        update_results = {}
        
        # Update panels based on frame intervals
        for panel_id, (args, kwargs) in panel_updates.items():
            if self._should_update_panel(panel_id):
                try:
                    if panel_id == 'vu_meters' and not self.frozen_vu_meters:
                        self.vu_meters_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'professional_meters' and not self.frozen_meters:
                        self.professional_meters_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'bass_zoom' and not self.frozen_bass_zoom:
                        self.bass_zoom_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'harmonic_analysis' and not self.frozen_harmonic:
                        self.harmonic_analysis_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'pitch_detection' and not self.frozen_pitch_detection:
                        self.pitch_detection_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'chromagram' and not self.frozen_chromagram:
                        self.chromagram_panel.update(*args, **kwargs)
                        update_results[panel_id] = True
                    elif panel_id == 'genre_classification' and not self.frozen_genre_classification:
                        self.genre_classification_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'voice_detection' and not self.frozen_voice_detection:
                        self.voice_detection_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'phase_correlation' and not self.frozen_phase_correlation:
                        self.phase_correlation_panel.update(*args)
                        update_results[panel_id] = True
                    elif panel_id == 'transient_detection' and not self.frozen_transient_detection:
                        self.transient_detection_panel.update(*args)
                        update_results[panel_id] = True
                except Exception as e:
                    print(f"Error updating panel {panel_id}: {e}")'''

content = re.sub(batch_update_pattern, replacement, content)

# Remove calls to activate_panel
content = re.sub(r'self\.panel_update_manager\.activate_panel\(.*?\)\n', '', content)

# Increment frame counter at the end of the main loop
# Find the main loop and add frame counter increment
main_loop_pattern = r'(# Force minimum frame time.*?pygame\.time\.wait\(int\(sleep_time \* 1000\)\)\s*\n)'
content = re.sub(
    main_loop_pattern,
    r'\1            \n            # Increment frame counter\n            self.frame_counter += 1\n',
    content,
    flags=re.DOTALL
)

# Write the optimized file
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Applied simple frame-skipping optimization")
print("Backup saved as omega4_main_pre_simple_opt.py")