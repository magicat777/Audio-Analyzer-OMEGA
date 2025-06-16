"""
Debug utilities for music analysis - helps identify why values are stuck
"""

import numpy as np
from typing import Dict, Any

def debug_genre_features(features: Dict[str, float], genre_name: str) -> str:
    """Debug why a genre is getting a certain score"""
    debug_lines = [f"\n=== Debug Genre: {genre_name} ==="]
    
    # List all features with their values
    for feature, value in features.items():
        debug_lines.append(f"{feature}: {value:.3f}")
    
    # Check for stuck values
    stuck_features = []
    for feature, value in features.items():
        if 0.12 < value < 0.14:  # Around 13%
            stuck_features.append(f"{feature} (stuck at ~13%)")
        elif 0.49 < value < 0.51:  # Around 50%
            stuck_features.append(f"{feature} (stuck at ~50%)")
        elif 0.33 < value < 0.35:  # Around 34%
            stuck_features.append(f"{feature} (stuck at ~34%)")
    
    if stuck_features:
        debug_lines.append(f"\nSTUCK FEATURES: {', '.join(stuck_features)}")
    
    return '\n'.join(debug_lines)

def debug_confidence_calculation(harmonic_conf: float, genre_conf: float, rhythmic_conf: float) -> str:
    """Debug confidence fusion calculation"""
    # Original weights from config
    harmonic_weight = 0.4
    timbral_weight = 0.3
    rhythmic_weight = 0.3
    
    # Current calculation
    weighted_sum = (harmonic_conf * harmonic_weight + 
                   genre_conf * timbral_weight + 
                   rhythmic_conf * rhythmic_weight)
    
    debug = f"""
=== Confidence Debug ===
Harmonic: {harmonic_conf:.3f} * {harmonic_weight} = {harmonic_conf * harmonic_weight:.3f}
Genre/Timbral: {genre_conf:.3f} * {timbral_weight} = {genre_conf * timbral_weight:.3f}
Rhythmic: {rhythmic_conf:.3f} * {rhythmic_weight} = {rhythmic_conf * rhythmic_weight:.3f}
Weighted Sum: {weighted_sum:.3f}
Max Possible: {harmonic_weight + timbral_weight + rhythmic_weight:.3f}

Issue: If all confidences are ~0.5, result will be ~0.5
Need: Dynamic range expansion or different fusion method
"""
    return debug

def debug_chromagram(chromagram: np.ndarray) -> str:
    """Debug chromagram values"""
    if chromagram is None or len(chromagram) != 12:
        return "Invalid chromagram"
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    debug_lines = ["\n=== Chromagram Debug ==="]
    
    # Find strongest notes
    sorted_indices = np.argsort(chromagram)[::-1]
    
    for i in range(12):
        idx = sorted_indices[i]
        value = chromagram[idx]
        if value > 0.01:  # Only show significant values
            debug_lines.append(f"{notes[idx]}: {value:.3f} {'*' * int(value * 20)}")
    
    # Check if stuck on F#
    if chromagram[6] > 0.3:  # F# is index 6
        debug_lines.append("\nWARNING: F# dominance detected - may be noise or calibration issue")
    
    return '\n'.join(debug_lines)

def debug_hip_hop_detection(features: Dict[str, float], hip_hop_analysis: Dict[str, Any]) -> str:
    """Debug why hip-hop isn't being detected"""
    debug_lines = ["\n=== Hip-Hop Detection Debug ==="]
    
    # Check each hip-hop feature
    debug_lines.append(f"Hip-Hop Confidence: {hip_hop_analysis.get('confidence', 0):.3f}")
    debug_lines.append(f"Is Hip-Hop: {hip_hop_analysis.get('is_hip_hop', False)}")
    debug_lines.append(f"Subgenre: {hip_hop_analysis.get('subgenre', 'unknown')}")
    
    hip_hop_features = hip_hop_analysis.get('features', {})
    debug_lines.append("\nHip-Hop Features:")
    debug_lines.append(f"  Sub-bass (808s): {hip_hop_features.get('sub_bass_presence', 0):.3f}")
    debug_lines.append(f"  Kick Pattern: {hip_hop_features.get('kick_pattern_score', 0):.3f}")
    debug_lines.append(f"  Hi-hat Density: {hip_hop_features.get('hihat_density', 0):.3f}")
    debug_lines.append(f"  Spectral Tilt: {hip_hop_features.get('spectral_tilt', 0):.3f}")
    debug_lines.append(f"  Vocal Presence: {hip_hop_features.get('vocal_presence', 0):.3f}")
    
    # Check why it might be detected as Country instead
    debug_lines.append("\nCountry vs Hip-Hop Features:")
    debug_lines.append(f"  Percussion Strength: {features.get('percussion_strength', 0):.3f}")
    debug_lines.append(f"  Bass Emphasis: {features.get('bass_emphasis', 0):.3f}")
    debug_lines.append(f"  Harmonic Complexity: {features.get('harmonic_complexity', 0):.3f}")
    
    return '\n'.join(debug_lines)