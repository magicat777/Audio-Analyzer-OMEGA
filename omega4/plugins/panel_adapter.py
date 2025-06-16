"""
Panel Adapter for OMEGA-4
Converts existing panels to plugin format
"""

from typing import Dict, Any, Optional
import numpy as np

from omega4.plugins.base import PanelPlugin, PluginMetadata, PluginType


class PanelAdapter(PanelPlugin):
    """Adapter to convert existing panel classes to plugins"""
    
    def __init__(self, panel_class, metadata: PluginMetadata):
        super().__init__()
        self.panel_class = panel_class
        self._metadata = metadata
        self.panel_instance = None
        self.sample_rate = 48000
        
    def get_metadata(self) -> PluginMetadata:
        return self._metadata
        
    def initialize(self, config: Dict = None) -> bool:
        """Initialize the adapted panel"""
        if not super().initialize(config):
            return False
            
        try:
            # Create panel instance
            self.panel_instance = self.panel_class(self.sample_rate)
            
            # Set fonts if panel has the method
            if hasattr(self.panel_instance, 'set_fonts') and self._font_cache:
                self.panel_instance.set_fonts(self._font_cache)
                
            return True
        except Exception as e:
            print(f"Failed to initialize panel adapter: {e}")
            return False
            
    def update(self, data: Dict[str, Any]):
        """Update the adapted panel"""
        if not self._enabled or not self.panel_instance:
            return
            
        # Extract data that panels typically need
        audio_data = data.get("audio_data")
        fft_data = data.get("fft_data")
        band_values = data.get("band_values")
        
        # Call panel's update method based on what it expects
        if hasattr(self.panel_instance, 'update'):
            # Generic update method
            self.panel_instance.update(data)
        elif hasattr(self.panel_instance, 'update_spectrum'):
            # Spectrum-based panels
            if band_values is not None:
                self.panel_instance.update_spectrum(band_values)
        elif hasattr(self.panel_instance, 'update_audio'):
            # Audio-based panels
            if audio_data is not None:
                self.panel_instance.update_audio(audio_data)
                
        # Handle specific panel types
        panel_name = self._metadata.name.lower()
        
        if "meter" in panel_name and hasattr(self.panel_instance, 'update_audio'):
            if audio_data is not None:
                self.panel_instance.update_audio(audio_data)
                
        elif "bass" in panel_name and hasattr(self.panel_instance, 'update_async'):
            if fft_data is not None:
                freqs = data.get("frequencies", np.fft.rfftfreq(len(fft_data) * 2 - 1, 1 / self.sample_rate))
                self.panel_instance.update_async(fft_data, freqs)
                
        elif "harmonic" in panel_name and hasattr(self.panel_instance, 'update_spectrum'):
            if fft_data is not None:
                freqs = data.get("frequencies", np.fft.rfftfreq(len(fft_data) * 2 - 1, 1 / self.sample_rate))
                harmonic_info = self.panel_instance.analyzer.detect_harmonic_series(fft_data, freqs)
                self.panel_instance.harmonic_info = harmonic_info
                
        elif "pitch" in panel_name and hasattr(self.panel_instance, 'update_pitch'):
            pitch_info = data.get("pitch_info", {})
            self.panel_instance.update_pitch(pitch_info)
            
        elif "chromagram" in panel_name and hasattr(self.panel_instance, 'update_chromagram'):
            if fft_data is not None:
                freqs = data.get("frequencies", np.fft.rfftfreq(len(fft_data) * 2 - 1, 1 / self.sample_rate))
                self.panel_instance.update_chromagram(fft_data, freqs)
                
        elif "genre" in panel_name and hasattr(self.panel_instance, 'update_spectrum'):
            if fft_data is not None:
                self.panel_instance.update_spectrum(fft_data)
                
    def draw(self, screen, x: int, y: int, width: int, height: int):
        """Draw the adapted panel"""
        if not self._enabled or not self._visible or not self.panel_instance:
            return
            
        # Call panel's draw method
        if hasattr(self.panel_instance, 'draw'):
            self.panel_instance.draw(screen, x, y, width, height)
            
    def set_fonts(self, fonts: Dict[str, Any]):
        """Set fonts for the panel"""
        super().set_fonts(fonts)
        
        if self.panel_instance and hasattr(self.panel_instance, 'set_fonts'):
            self.panel_instance.set_fonts(fonts)
            
    def shutdown(self):
        """Clean up resources"""
        if self.panel_instance and hasattr(self.panel_instance, 'cleanup'):
            self.panel_instance.cleanup()
        super().shutdown()


def create_panel_plugin(panel_class, name: str, version: str = "1.0.0", 
                       author: str = "OMEGA-4", description: str = "") -> PanelAdapter:
    """Factory function to create panel plugins from existing classes"""
    
    metadata = PluginMetadata(
        name=name,
        version=version,
        author=author,
        description=description or f"{name} panel",
        plugin_type=PluginType.PANEL
    )
    
    return PanelAdapter(panel_class, metadata)


# Pre-defined adapters for existing panels
def create_professional_meters_plugin():
    """Create plugin for professional meters panel"""
    from omega4.panels.professional_meters import ProfessionalMetersPanel
    
    return create_panel_plugin(
        ProfessionalMetersPanel,
        "Professional Meters",
        description="LUFS metering with K-weighting and true peak detection"
    )


def create_vu_meters_plugin():
    """Create plugin for VU meters panel"""
    from omega4.panels.vu_meters import VUMetersPanel
    
    return create_panel_plugin(
        VUMetersPanel,
        "VU Meters",
        description="Classic VU meters with proper ballistics"
    )


def create_bass_zoom_plugin():
    """Create plugin for bass zoom panel"""
    from omega4.panels.bass_zoom import BassZoomPanel
    
    return create_panel_plugin(
        BassZoomPanel,
        "Bass Zoom",
        description="Detailed low-frequency analysis window"
    )


def create_harmonic_analysis_plugin():
    """Create plugin for harmonic analysis panel"""
    from omega4.panels.harmonic_analysis import HarmonicAnalysisPanel
    
    return create_panel_plugin(
        HarmonicAnalysisPanel,
        "Harmonic Analysis",
        description="Harmonic series detection and instrument identification"
    )


def create_pitch_detection_plugin():
    """Create plugin for pitch detection panel"""
    from omega4.panels.pitch_detection import PitchDetectionPanel
    
    return create_panel_plugin(
        PitchDetectionPanel,
        "Pitch Detection",
        description="Advanced pitch detection with multiple algorithms"
    )


def create_chromagram_plugin():
    """Create plugin for chromagram panel"""
    from omega4.panels.chromagram import ChromagramPanel
    
    return create_panel_plugin(
        ChromagramPanel,
        "Chromagram",
        description="Musical key detection and chromagram visualization"
    )


def create_genre_classification_plugin():
    """Create plugin for genre classification panel"""
    from omega4.panels.genre_classification import GenreClassificationPanel
    
    return create_panel_plugin(
        GenreClassificationPanel,
        "Genre Classification",
        description="Real-time music genre classification"
    )