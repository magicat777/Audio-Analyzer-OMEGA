"""
Configuration Manager for OMEGA-4 Audio Analyzer
Phase 7: Handle configuration loading, saving, and migration
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from .schema import Configuration, DEFAULT_CONFIGS


class ConfigurationManager:
    """Manages application configuration with persistence"""
    
    def __init__(self, config_dir: str = None):
        # Default to user config directory
        if config_dir is None:
            config_dir = os.path.join(os.path.expanduser("~"), ".omega4")
            
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.yaml"
        self.backup_dir = self.config_dir / "backups"
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Current configuration
        self.config: Configuration = None
        self.config_loaded = False
        
    def load_config(self, path: Optional[str] = None) -> Configuration:
        """Load configuration from file
        
        Args:
            path: Optional path to config file. Uses default if not provided.
            
        Returns:
            Loaded configuration
        """
        config_path = Path(path) if path else self.config_file
        
        if config_path.exists():
            try:
                # Load from file
                with open(config_path, 'r') as f:
                    if config_path.suffix in ['.yaml', '.yml']:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                # Check if migration is needed
                if self._needs_migration(data):
                    data = self.migrate_config(data)
                    
                # Create configuration object
                self.config = Configuration.from_dict(data)
                
                # Validate
                errors = self.config.validate()
                if errors:
                    print(f"Configuration validation errors: {errors}")
                    # Continue with invalid config but log errors
                    
                self.config_loaded = True
                print(f"✓ Configuration loaded from {config_path}")
                
            except Exception as e:
                print(f"✗ Failed to load configuration: {e}")
                # Fall back to default
                self.config = self.get_default_config()
                self.config_loaded = False
                
        else:
            # No config file, use default
            print("No configuration file found, using defaults")
            self.config = self.get_default_config()
            self.config_loaded = False
            
        return self.config
        
    def save_config(self, config: Optional[Configuration] = None, 
                   path: Optional[str] = None) -> bool:
        """Save configuration to file
        
        Args:
            config: Configuration to save (uses current if not provided)
            path: Optional path to save to
            
        Returns:
            True if successful
        """
        if config is None:
            config = self.config
            
        if config is None:
            print("No configuration to save")
            return False
            
        config_path = Path(path) if path else self.config_file
        
        try:
            # Create backup if file exists
            if config_path.exists():
                self._create_backup(config_path)
                
            # Convert to dict
            data = config.to_dict()
            
            # Save based on extension
            with open(config_path, 'w') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(data, f, indent=2)
                    
            print(f"✓ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save configuration: {e}")
            return False
            
    def validate_config(self, config: Optional[Configuration] = None) -> List[str]:
        """Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        if config is None:
            config = self.config
            
        if config is None:
            return ["No configuration to validate"]
            
        return config.validate()
        
    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration format to current version
        
        Args:
            old_config: Old configuration data
            
        Returns:
            Migrated configuration data
        """
        print("Migrating configuration...")
        
        # Get version
        old_version = old_config.get("version", "0.0.0")
        current_version = Configuration().version
        
        if old_version == current_version:
            return old_config
            
        # Make a copy
        migrated = old_config.copy()
        
        # Version-specific migrations
        if old_version < "1.0.0":
            # Example migration from pre-1.0
            print(f"Migrating from version {old_version} to {current_version}")
            
            # Rename old fields
            if "bars" in migrated:
                if "analysis" not in migrated:
                    migrated["analysis"] = {}
                migrated["analysis"]["num_bands"] = migrated.pop("bars")
                
            # Add new required fields with defaults
            default = Configuration()
            
            if "audio" not in migrated:
                migrated["audio"] = default.audio.to_dict()
                
            if "display" not in migrated:
                migrated["display"] = default.display.to_dict()
                
        # Update version
        migrated["version"] = current_version
        
        print(f"✓ Configuration migrated to version {current_version}")
        return migrated
        
    def get_default_config(self, preset: str = "default") -> Configuration:
        """Get default configuration
        
        Args:
            preset: Name of default preset to use
            
        Returns:
            Default configuration
        """
        if preset in DEFAULT_CONFIGS:
            return DEFAULT_CONFIGS[preset]
        return Configuration()
        
    def reset_to_default(self, preset: str = "default") -> Configuration:
        """Reset configuration to defaults
        
        Args:
            preset: Default preset to use
            
        Returns:
            New default configuration
        """
        self.config = self.get_default_config(preset)
        return self.config
        
    def update(self, updates: Dict[str, Any]):
        """Update configuration with partial data
        
        Args:
            updates: Dictionary with updates (can be nested)
        """
        if self.config is None:
            self.config = Configuration()
            
        # Apply updates recursively
        self._apply_updates(self.config, updates)
        
    def _apply_updates(self, obj: Any, updates: Dict[str, Any]):
        """Recursively apply updates to configuration object"""
        for key, value in updates.items():
            if hasattr(obj, key):
                current = getattr(obj, key)
                
                if isinstance(value, dict) and hasattr(current, '__dict__'):
                    # Recursively update nested objects
                    self._apply_updates(current, value)
                else:
                    # Direct update
                    setattr(obj, key, value)
                    
    def export_config(self, path: str):
        """Export configuration to file
        
        Args:
            path: Path to export to
        """
        if self.config:
            self.save_config(self.config, path)
            
    def import_config(self, path: str) -> Configuration:
        """Import configuration from file
        
        Args:
            path: Path to import from
            
        Returns:
            Imported configuration
        """
        imported = self.load_config(path)
        self.config = imported
        return imported
        
    def _create_backup(self, config_path: Path):
        """Create backup of configuration file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"config_backup_{timestamp}{config_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(config_path, backup_path)
            print(f"✓ Backup created: {backup_path}")
            
            # Keep only last 10 backups
            self._cleanup_old_backups()
            
        except Exception as e:
            print(f"✗ Failed to create backup: {e}")
            
    def _cleanup_old_backups(self, keep_count: int = 10):
        """Remove old backup files"""
        backups = sorted(self.backup_dir.glob("config_backup_*"))
        
        if len(backups) > keep_count:
            for backup in backups[:-keep_count]:
                try:
                    backup.unlink()
                except:
                    pass
                    
    def _needs_migration(self, config_data: Dict[str, Any]) -> bool:
        """Check if configuration needs migration"""
        current_version = Configuration().version
        file_version = config_data.get("version", "0.0.0")
        
        return file_version < current_version
        
    def get_audio_devices(self) -> List[str]:
        """Get list of available audio devices"""
        # This would interface with the audio system
        # For now, return empty list
        return []
        
    def apply_color_scheme(self, scheme_name: str):
        """Apply a color scheme to the configuration"""
        from .schema import ColorScheme
        
        try:
            scheme = ColorScheme(scheme_name)
            if self.config:
                self.config.display.color_scheme = scheme
        except ValueError:
            print(f"Invalid color scheme: {scheme_name}")
            
    def get_panel_layout(self) -> Dict[str, Dict[str, Any]]:
        """Get current panel layout configuration"""
        if self.config and self.config.layout:
            layout = {}
            for panel_name, panel_config in self.config.layout.panels.items():
                layout[panel_name] = {
                    "visible": panel_config.visible,
                    "position": panel_config.position,
                    "size": panel_config.size
                }
            return layout
        return {}
        
    def save_panel_layout(self, layout_name: str):
        """Save current panel layout with a name"""
        if self.config:
            self.config.layout.layout_name = layout_name
            # Layout is already part of config, just save
            self.save_config()