#!/usr/bin/env python3
"""
Simple optimization approach for omega4_main.py
This version implements frame skipping for non-critical panels without complex optimization systems
"""

import shutil
import os

# Create backup
shutil.copy('omega4_main.py', 'omega4_main_backup_complex_opt.py')

# Read the original file
with open('omega4_main.py', 'r') as f:
    content = f.read()

# Define the simple optimization approach
simple_optimization = '''
    def _should_update_panel(self, panel_name, frame_count):
        """Simple frame-based update scheduling"""
        # Define update intervals for different panels
        update_intervals = {
            # Visual panels - update frequently
            'vu_meters': 1,            # Every frame
            'professional_meters': 2,   # Every 2 frames (30 FPS)
            'bass_zoom': 2,            # Every 2 frames
            'transient_detection': 2,   # Every 2 frames
            
            # Analysis panels - update less frequently
            'chromagram': 4,           # Every 4 frames (15 FPS)
            'pitch_detection': 4,      # Every 4 frames
            'voice_detection': 4,      # Every 4 frames
            'phase_correlation': 4,    # Every 4 frames
            
            # Heavy computation panels - update rarely
            'harmonic_analysis': 8,    # Every 8 frames (7.5 FPS)
            'genre_classification': 8, # Every 8 frames
            'integrated_music': 16,    # Every 16 frames (3.75 FPS)
            'room_analysis': 16,       # Every 16 frames
        }
        
        # Get interval for this panel
        interval = update_intervals.get(panel_name, 4)  # Default to 4 frames
        
        # Check if frozen
        frozen_attr = f'frozen_{panel_name}'
        if hasattr(self, frozen_attr) and getattr(self, frozen_attr):
            return False
            
        # Simple modulo check
        return frame_count % interval == 0
'''

# Remove complex optimization imports
content = content.replace('from omega4.optimization.panel_update_manager import PanelUpdateManager, UpdatePriority', '')
content = content.replace('from omega4.optimization.render_optimization import RenderOptimizer', '')

# Remove panel update manager initialization
content = content.replace('        self.panel_update_manager = PanelUpdateManager(TARGET_FPS)', '')
content = content.replace('        self.render_optimizer = RenderOptimizer()', '')

# Remove the _register_panel_updates method by replacing it with pass
import re
register_pattern = r'def _register_panel_updates\(self\):.*?(?=\n    def|\n\nclass|\Z)'
content = re.sub(register_pattern, 'def _register_panel_updates(self):\n        """Not needed in simple optimization"""\n        pass', content, flags=re.DOTALL)

# Add frame counter initialization
init_addition = '''
        # Simple frame counter for update scheduling
        self.frame_counter = 0
'''

# Find where to add it (after self.running = True)
running_pattern = r'(self\.running = True)'
content = re.sub(running_pattern, r'\1' + init_addition, content)

# Add the simple optimization method after capture_audio method
capture_pattern = r'(def capture_audio\(self\):.*?(?=\n    def))'
content = re.sub(capture_pattern, r'\1\n' + simple_optimization + '\n', content, flags=re.DOTALL)

# Modify the main update loop to use simple frame skipping
# We need to find where panels are updated and wrap them with our check

# Write the modified content
with open('omega4_main.py', 'w') as f:
    f.write(content)

print("Created omega4_main.py with simple optimization approach")
print("Backup saved as omega4_main_backup_complex_opt.py")