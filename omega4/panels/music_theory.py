"""
Music Theory Utilities for OMEGA-4 Audio Analyzer
Provides musical analysis helpers for chromagram and harmonic panels
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MusicTheory:
    """Musical theory utilities and constants"""
    
    # Circle of fifths ordering
    CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    CIRCLE_OF_FIFTHS_MINOR = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Fm', 'Cm', 'Gm', 'Dm']
    
    # Note names with enharmonic equivalents
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    
    # Modal scales (intervals from root)
    MODES = {
        'Ionian (Major)': [0, 2, 4, 5, 7, 9, 11],      # Major scale
        'Dorian': [0, 2, 3, 5, 7, 9, 10],              # Minor with raised 6th
        'Phrygian': [0, 1, 3, 5, 7, 8, 10],            # Minor with lowered 2nd
        'Lydian': [0, 2, 4, 6, 7, 9, 11],              # Major with raised 4th
        'Mixolydian': [0, 2, 4, 5, 7, 9, 10],          # Major with lowered 7th
        'Aeolian (Natural Minor)': [0, 2, 3, 5, 7, 8, 10],  # Natural minor
        'Locrian': [0, 1, 3, 5, 6, 8, 10]              # Diminished scale
    }
    
    # Roman numeral notation for scale degrees
    ROMAN_NUMERALS_MAJOR = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
    ROMAN_NUMERALS_MINOR = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']
    
    # Common chord progressions
    PROGRESSIONS = {
        'I-V-vi-IV': ['I', 'V', 'vi', 'IV'],           # Pop progression
        'I-vi-IV-V': ['I', 'vi', 'IV', 'V'],           # 50s progression
        'ii-V-I': ['ii', 'V', 'I'],                    # Jazz cadence
        'I-IV-V': ['I', 'IV', 'V'],                    # Blues/Rock
        'vi-IV-I-V': ['vi', 'IV', 'I', 'V'],           # Alternative pop
        'I-V-vi-iii-IV-I-IV-V': ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],  # Canon
        'i-VII-VI-V': ['i', 'VII', 'VI', 'V'],         # Andalusian cadence
        'i-iv-V': ['i', 'iv', 'V'],                    # Minor blues
    }
    
    # Chord functions
    CHORD_FUNCTIONS = {
        'I': 'Tonic',
        'ii': 'Supertonic',
        'iii': 'Mediant',
        'IV': 'Subdominant',
        'V': 'Dominant',
        'vi': 'Submediant',
        'vii°': 'Leading Tone',
        'i': 'Tonic',
        'ii°': 'Supertonic',
        'III': 'Mediant',
        'iv': 'Subdominant',
        'v': 'Dominant',
        'VI': 'Submediant',
        'VII': 'Subtonic'
    }
    
    # Tonic/Subdominant/Dominant groupings
    TSD_GROUPS = {
        'Tonic': ['I', 'vi', 'iii', 'i', 'VI', 'III'],
        'Subdominant': ['IV', 'ii', 'iv', 'ii°'],
        'Dominant': ['V', 'vii°', 'v', 'VII']
    }
    
    @staticmethod
    def get_circle_position(key: str) -> Tuple[int, bool]:
        """Get position on circle of fifths (0-11) and whether it's major"""
        # Extract root and quality
        if ' Major' in key:
            root = key.replace(' Major', '')
            is_major = True
        elif ' Minor' in key:
            root = key.replace(' Minor', '')
            is_major = False
        elif 'm' in key and len(key) <= 3:  # e.g., "Am", "C#m"
            root = key.replace('m', '')
            is_major = False
        else:
            root = key
            is_major = True
        
        # Handle enharmonic equivalents
        enharmonics = {'Gb': 'F#', 'Db': 'C#', 'Ab': 'G#', 'Eb': 'D#', 'Bb': 'A#'}
        if root in enharmonics:
            root = enharmonics[root]
        
        # Find position
        try:
            if is_major:
                position = MusicTheory.CIRCLE_OF_FIFTHS.index(root)
            else:
                # Convert to relative major for position
                major_root = MusicTheory.relative_major(root)
                position = MusicTheory.CIRCLE_OF_FIFTHS.index(major_root)
            return position, is_major
        except ValueError:
            return 0, True
    
    @staticmethod
    def relative_major(minor_root: str) -> str:
        """Get relative major of a minor key"""
        try:
            idx = MusicTheory.NOTE_NAMES.index(minor_root)
            major_idx = (idx + 3) % 12
            return MusicTheory.NOTE_NAMES[major_idx]
        except ValueError:
            return 'C'
    
    @staticmethod
    def relative_minor(major_root: str) -> str:
        """Get relative minor of a major key"""
        try:
            idx = MusicTheory.NOTE_NAMES.index(major_root)
            minor_idx = (idx - 3) % 12
            return MusicTheory.NOTE_NAMES[minor_idx]
        except ValueError:
            return 'A'
    
    @staticmethod
    def chord_to_roman(chord: str, key: str) -> str:
        """Convert chord name to Roman numeral in given key"""
        # Extract root from chord
        if len(chord) >= 2 and chord[1] in ['#', 'b']:
            chord_root = chord[:2]
            chord_quality = chord[2:]
        else:
            chord_root = chord[0]
            chord_quality = chord[1:]
        
        # Extract key root
        key_parts = key.split()
        key_root = key_parts[0]
        is_major_key = len(key_parts) > 1 and key_parts[1] == 'Major'
        
        try:
            # Get scale degree
            key_idx = MusicTheory.NOTE_NAMES.index(key_root)
            chord_idx = MusicTheory.NOTE_NAMES.index(chord_root)
            degree = ((chord_idx - key_idx) % 12)
            
            # Map to scale position (0-indexed to 1-indexed)
            scale_degrees = [0, 2, 4, 5, 7, 9, 11] if is_major_key else [0, 2, 3, 5, 7, 8, 10]
            
            # Find which scale degree this is
            for i, sd in enumerate(scale_degrees):
                if degree == sd:
                    if is_major_key:
                        roman = MusicTheory.ROMAN_NUMERALS_MAJOR[i]
                    else:
                        roman = MusicTheory.ROMAN_NUMERALS_MINOR[i]
                    
                    # Add extensions
                    if '7' in chord_quality:
                        roman += '7'
                    elif '9' in chord_quality:
                        roman += '9'
                    
                    return roman
            
            # Not in scale - return with accidentals
            return f"♭{chord_root}" if degree - 1 in scale_degrees else f"♯{chord_root}"
            
        except ValueError:
            return chord
    
    @staticmethod
    def analyze_mode(chroma: np.ndarray, root_idx: int) -> Tuple[str, float]:
        """Analyze which mode best fits the chromagram"""
        best_mode = 'Ionian (Major)'
        best_score = 0.0
        
        # Rotate chromagram to put root at position 0
        rotated_chroma = np.roll(chroma, -root_idx)
        
        for mode_name, intervals in MusicTheory.MODES.items():
            # Create mode template
            template = np.zeros(12)
            for interval in intervals:
                template[interval] = 1.0
            
            # Normalize template
            template = template / np.sum(template)
            
            # Compare with chromagram
            if np.std(rotated_chroma) > 0 and np.std(template) > 0:
                correlation = np.corrcoef(rotated_chroma, template)[0, 1]
                
                if correlation > best_score:
                    best_score = correlation
                    best_mode = mode_name
        
        return best_mode, best_score
    
    @staticmethod
    def get_chord_function(roman: str) -> str:
        """Get functional description of a chord"""
        # Remove extensions for lookup
        base_roman = roman.rstrip('7965')
        return MusicTheory.CHORD_FUNCTIONS.get(base_roman, 'Unknown')
    
    @staticmethod
    def get_tsd_group(roman: str) -> str:
        """Get Tonic/Subdominant/Dominant group for a chord"""
        base_roman = roman.rstrip('7965')
        
        for group, members in MusicTheory.TSD_GROUPS.items():
            if base_roman in members:
                return group
        
        return 'Other'
    
    @staticmethod
    def detect_key_modulation(prev_key: str, current_key: str) -> Optional[str]:
        """Detect type of key modulation"""
        if prev_key == current_key:
            return None
        
        # Extract roots
        prev_root = prev_key.split()[0]
        curr_root = current_key.split()[0]
        prev_is_major = 'Major' in prev_key
        curr_is_major = 'Major' in current_key
        
        # Check for parallel modulation (same root, different mode)
        if prev_root == curr_root:
            return "Parallel"
        
        # Check for relative modulation
        if prev_is_major and not curr_is_major:
            if MusicTheory.relative_minor(prev_root) == curr_root:
                return "Relative Minor"
        elif not prev_is_major and curr_is_major:
            if MusicTheory.relative_major(prev_root) == curr_root:
                return "Relative Major"
        
        # Check circle of fifths distance
        try:
            prev_pos = MusicTheory.CIRCLE_OF_FIFTHS.index(prev_root)
            curr_pos = MusicTheory.CIRCLE_OF_FIFTHS.index(curr_root)
            
            # Calculate shortest distance around circle
            dist = min(abs(curr_pos - prev_pos), 12 - abs(curr_pos - prev_pos))
            
            if dist == 1:
                if curr_pos == (prev_pos + 1) % 12:
                    return "Dominant"
                else:
                    return "Subdominant"
            elif dist == 2:
                return "Supertonic"
            elif dist == 3:
                return "Mediant"
            elif dist == 6:
                return "Tritone"
        except ValueError:
            pass
        
        return "Chromatic"
    
    @staticmethod
    def create_chord_transition_matrix(chord_sequence: List[str]) -> Dict[str, Dict[str, float]]:
        """Create transition probability matrix from chord sequence"""
        transitions = {}
        
        # Count transitions
        for i in range(len(chord_sequence) - 1):
            curr = chord_sequence[i]
            next_chord = chord_sequence[i + 1]
            
            if curr not in transitions:
                transitions[curr] = {}
            
            if next_chord not in transitions[curr]:
                transitions[curr][next_chord] = 0
            
            transitions[curr][next_chord] += 1
        
        # Normalize to probabilities
        for chord in transitions:
            total = sum(transitions[chord].values())
            if total > 0:
                for next_chord in transitions[chord]:
                    transitions[chord][next_chord] /= total
        
        return transitions