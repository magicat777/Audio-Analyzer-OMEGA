"""
Preset Manager for OMEGA-4 Audio Analyzer
Phase 7: Handle user presets for different use cases
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .schema import Configuration


@dataclass
class PresetMetadata:
    """Metadata for a preset"""
    name: str
    description: str
    category: str
    author: str = "User"
    created: str = ""
    modified: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.modified:
            self.modified = self.created
        if self.tags is None:
            self.tags = []


@dataclass
class Preset:
    """A saved configuration preset"""
    metadata: PresetMetadata
    configuration: Configuration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metadata": asdict(self.metadata),
            "configuration": self.configuration.to_dict()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Preset':
        """Create from dictionary"""
        metadata = PresetMetadata(**data["metadata"])
        configuration = Configuration.from_dict(data["configuration"])
        return cls(metadata=metadata, configuration=configuration)


class PresetManager:
    """Manages user presets"""
    
    # Preset categories
    CATEGORIES = {
        "general": "General Purpose",
        "music": "Music Analysis",
        "production": "Music Production",
        "live": "Live Performance",
        "gaming": "Gaming",
        "podcast": "Podcast/Voice",
        "custom": "Custom"
    }
    
    def __init__(self, preset_dir: str = None):
        # Default to user preset directory
        if preset_dir is None:
            preset_dir = os.path.join(os.path.expanduser("~"), ".omega4", "presets")
            
        self.preset_dir = Path(preset_dir)
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category subdirectories
        for category in self.CATEGORIES:
            (self.preset_dir / category).mkdir(exist_ok=True)
            
        # Cache of loaded presets
        self._preset_cache: Dict[str, Preset] = {}
        
        # Built-in presets
        self._builtin_presets = self._create_builtin_presets()
        
    def _create_builtin_presets(self) -> Dict[str, Preset]:
        """Create built-in presets"""
        presets = {}
        
        # Music Production Preset
        music_config = Configuration()
        music_config.audio.input_gain = 2.0
        music_config.audio.target_lufs = -14.0
        music_config.analysis.freq_compensation = True
        music_config.analysis.psychoacoustic_weighting = True
        music_config.display.grid_enabled = True
        
        presets["music_production"] = Preset(
            metadata=PresetMetadata(
                name="Music Production",
                description="Optimized for music production and mixing",
                category="production",
                author="OMEGA-4",
                tags=["music", "studio", "mixing"]
            ),
            configuration=music_config
        )
        
        # Live Performance Preset
        live_config = Configuration()
        live_config.display.show_fps = False
        live_config.display.grid_enabled = False
        live_config.analysis.drum_detection = True
        live_config.analysis.drum_sensitivity = 1.5
        live_config.analysis.smoothing_factor = 0.85
        
        presets["live_performance"] = Preset(
            metadata=PresetMetadata(
                name="Live Performance",
                description="Optimized for live performance visualization",
                category="live",
                author="OMEGA-4",
                tags=["live", "performance", "visual"]
            ),
            configuration=live_config
        )
        
        # Podcast/Voice Preset
        voice_config = Configuration()
        voice_config.audio.input_gain = 6.0
        voice_config.analysis.voice_detection = True
        voice_config.analysis.voice_threshold = 0.3
        voice_config.analysis.genre_classification = False
        voice_config.analysis.drum_detection = False
        
        presets["podcast"] = Preset(
            metadata=PresetMetadata(
                name="Podcast/Voice",
                description="Optimized for voice and podcast recording",
                category="podcast",
                author="OMEGA-4",
                tags=["voice", "podcast", "speech"]
            ),
            configuration=voice_config
        )
        
        # Gaming Preset
        gaming_config = Configuration()
        gaming_config.display.target_fps = 144
        gaming_config.audio.chunk_size = 256  # Low latency
        gaming_config.analysis.fft_size = 2048
        gaming_config.analysis.num_bands = 512
        
        presets["gaming"] = Preset(
            metadata=PresetMetadata(
                name="Gaming",
                description="Low latency for gaming with audio visualization",
                category="gaming",
                author="OMEGA-4",
                tags=["gaming", "low-latency", "performance"]
            ),
            configuration=gaming_config
        )
        
        return presets
        
    def list_presets(self, category: Optional[str] = None) -> List[PresetMetadata]:
        """List available presets
        
        Args:
            category: Optional category filter
            
        Returns:
            List of preset metadata
        """
        presets = []
        
        # Add built-in presets
        for preset in self._builtin_presets.values():
            if category is None or preset.metadata.category == category:
                presets.append(preset.metadata)
                
        # Scan preset directories
        search_dirs = [self.preset_dir / category] if category else [
            self.preset_dir / cat for cat in self.CATEGORIES
        ]
        
        for preset_dir in search_dirs:
            if not preset_dir.exists():
                continue
                
            for preset_file in preset_dir.glob("*.yaml"):
                try:
                    # Load metadata only
                    with open(preset_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if "metadata" in data:
                            metadata = PresetMetadata(**data["metadata"])
                            presets.append(metadata)
                except:
                    pass
                    
        return presets
        
    def load_preset(self, preset_name: str) -> Optional[Preset]:
        """Load a preset by name
        
        Args:
            preset_name: Name of preset to load
            
        Returns:
            Loaded preset or None if not found
        """
        # Check cache
        if preset_name in self._preset_cache:
            return self._preset_cache[preset_name]
            
        # Check built-in presets
        if preset_name in self._builtin_presets:
            return self._builtin_presets[preset_name]
            
        # Search files
        for category in self.CATEGORIES:
            preset_file = self.preset_dir / category / f"{preset_name}.yaml"
            if preset_file.exists():
                try:
                    with open(preset_file, 'r') as f:
                        data = yaml.safe_load(f)
                        preset = Preset.from_dict(data)
                        self._preset_cache[preset_name] = preset
                        return preset
                except Exception as e:
                    print(f"Failed to load preset {preset_name}: {e}")
                    
        return None
        
    def save_preset(self, preset: Preset, overwrite: bool = False) -> bool:
        """Save a preset
        
        Args:
            preset: Preset to save
            overwrite: Whether to overwrite existing preset
            
        Returns:
            True if successful
        """
        # Determine file path
        category = preset.metadata.category
        if category not in self.CATEGORIES:
            category = "custom"
            
        preset_file = self.preset_dir / category / f"{preset.metadata.name}.yaml"
        
        # Check if exists
        if preset_file.exists() and not overwrite:
            print(f"Preset {preset.metadata.name} already exists")
            return False
            
        try:
            # Update modified time
            preset.metadata.modified = datetime.now().isoformat()
            
            # Save to file
            with open(preset_file, 'w') as f:
                yaml.dump(preset.to_dict(), f, default_flow_style=False)
                
            # Update cache
            self._preset_cache[preset.metadata.name] = preset
            
            print(f"✓ Preset saved: {preset.metadata.name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save preset: {e}")
            return False
            
    def save_current_as_preset(self, config: Configuration, name: str, 
                             description: str = "", category: str = "custom",
                             tags: List[str] = None) -> bool:
        """Save current configuration as a new preset
        
        Args:
            config: Current configuration
            name: Preset name
            description: Preset description
            category: Preset category
            tags: Optional tags
            
        Returns:
            True if successful
        """
        metadata = PresetMetadata(
            name=name,
            description=description,
            category=category,
            tags=tags or []
        )
        
        preset = Preset(metadata=metadata, configuration=config)
        return self.save_preset(preset)
        
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a preset
        
        Args:
            preset_name: Name of preset to delete
            
        Returns:
            True if successful
        """
        # Cannot delete built-in presets
        if preset_name in self._builtin_presets:
            print("Cannot delete built-in preset")
            return False
            
        # Search for preset file
        for category in self.CATEGORIES:
            preset_file = self.preset_dir / category / f"{preset_name}.yaml"
            if preset_file.exists():
                try:
                    preset_file.unlink()
                    
                    # Remove from cache
                    if preset_name in self._preset_cache:
                        del self._preset_cache[preset_name]
                        
                    print(f"✓ Preset deleted: {preset_name}")
                    return True
                    
                except Exception as e:
                    print(f"✗ Failed to delete preset: {e}")
                    
        return False
        
    def export_preset(self, preset_name: str, export_path: str) -> bool:
        """Export a preset to file
        
        Args:
            preset_name: Name of preset to export
            export_path: Path to export to
            
        Returns:
            True if successful
        """
        preset = self.load_preset(preset_name)
        if not preset:
            print(f"Preset not found: {preset_name}")
            return False
            
        try:
            with open(export_path, 'w') as f:
                yaml.dump(preset.to_dict(), f, default_flow_style=False)
                
            print(f"✓ Preset exported to {export_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to export preset: {e}")
            return False
            
    def import_preset(self, import_path: str, overwrite: bool = False) -> Optional[str]:
        """Import a preset from file
        
        Args:
            import_path: Path to import from
            overwrite: Whether to overwrite existing preset
            
        Returns:
            Name of imported preset or None if failed
        """
        try:
            with open(import_path, 'r') as f:
                data = yaml.safe_load(f)
                
            preset = Preset.from_dict(data)
            
            if self.save_preset(preset, overwrite):
                return preset.metadata.name
                
        except Exception as e:
            print(f"✗ Failed to import preset: {e}")
            
        return None
        
    def search_presets(self, query: str) -> List[PresetMetadata]:
        """Search presets by name, description, or tags
        
        Args:
            query: Search query
            
        Returns:
            List of matching preset metadata
        """
        query_lower = query.lower()
        matches = []
        
        # Search all presets
        all_presets = self.list_presets()
        
        for metadata in all_presets:
            # Check name
            if query_lower in metadata.name.lower():
                matches.append(metadata)
                continue
                
            # Check description
            if query_lower in metadata.description.lower():
                matches.append(metadata)
                continue
                
            # Check tags
            for tag in metadata.tags:
                if query_lower in tag.lower():
                    matches.append(metadata)
                    break
                    
        return matches
        
    def get_preset_by_category(self, category: str) -> List[PresetMetadata]:
        """Get all presets in a category
        
        Args:
            category: Category name
            
        Returns:
            List of preset metadata in category
        """
        return self.list_presets(category)
        
    def clear_cache(self):
        """Clear preset cache"""
        self._preset_cache.clear()