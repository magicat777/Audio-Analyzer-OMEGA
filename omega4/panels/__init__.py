"""
OMEGA-4 Audio Analyzer Panels
Phase 3: Complex panel extraction as self-contained modules
"""

# Core panels
from .professional_meters import ProfessionalMetersPanel
from .vu_meters import VUMetersPanel
from .bass_zoom import BassZoomPanel
from .harmonic_analysis import HarmonicAnalysisPanel
from .pitch_detection import PitchDetectionPanel
from .chromagram import ChromagramPanel
from .genre_classification import GenreClassificationPanel
from .integrated_music_panel import IntegratedMusicPanel

# New panels
from .voice_detection import VoiceDetectionPanel
from .phase_correlation import PhaseCorrelationPanel
from .transient_detection import TransientDetectionPanel

__all__ = [
    'ProfessionalMetersPanel',
    'VUMetersPanel',
    'BassZoomPanel',
    'HarmonicAnalysisPanel',
    'PitchDetectionPanel',
    'ChromagramPanel',
    'GenreClassificationPanel',
    'IntegratedMusicPanel',
    'VoiceDetectionPanel',
    'PhaseCorrelationPanel',
    'TransientDetectionPanel'
]