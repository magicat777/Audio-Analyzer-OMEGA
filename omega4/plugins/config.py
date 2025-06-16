"""
Plugin Configuration Management for OMEGA-4
Phase 6: Handle plugin settings and persistence
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class PluginConfig:
    """Configuration for a single plugin"""
    enabled: bool = True
    settings: Dict[str, Any] = None
    position: Optional[Dict[str, int]] = None  # For panel plugins
    size: Optional[Dict[str, int]] = None      # For panel plugins
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {}
            
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'PluginConfig':
        """Create from dictionary"""
        return cls(**data)


class PluginConfigManager:
    """Manages plugin configurations"""
    
    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir or os.path.join(os.path.expanduser("~"), ".omega4", "plugins")
        self.configs: Dict[str, PluginConfig] = {}
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Ensure configuration directory exists"""
        os.makedirs(self.config_dir, exist_ok=True)
        
    def load_all_configs(self) -> Dict[str, PluginConfig]:
        """Load all plugin configurations"""
        self.configs.clear()
        
        # Load from individual plugin config files
        for file in Path(self.config_dir).glob("*.yaml"):
            plugin_name = file.stem
            try:
                config = self.load_plugin_config(plugin_name)
                if config:
                    self.configs[plugin_name] = config
            except Exception as e:
                print(f"Failed to load config for {plugin_name}: {e}")
                
        # Load from global config file
        global_config_path = os.path.join(self.config_dir, "plugins.yaml")
        if os.path.exists(global_config_path):
            try:
                with open(global_config_path, 'r') as f:
                    global_configs = yaml.safe_load(f) or {}
                    
                for plugin_name, config_data in global_configs.items():
                    if plugin_name not in self.configs:
                        self.configs[plugin_name] = PluginConfig.from_dict(config_data)
            except Exception as e:
                print(f"Failed to load global plugin config: {e}")
                
        return self.configs
        
    def load_plugin_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Load configuration for a specific plugin"""
        config_path = os.path.join(self.config_dir, f"{plugin_name}.yaml")
        
        if not os.path.exists(config_path):
            # Try JSON format
            config_path = os.path.join(self.config_dir, f"{plugin_name}.json")
            
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml'):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                        
                return PluginConfig.from_dict(data)
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
                
        return None
        
    def save_plugin_config(self, plugin_name: str, config: PluginConfig):
        """Save configuration for a specific plugin"""
        self.configs[plugin_name] = config
        
        config_path = os.path.join(self.config_dir, f"{plugin_name}.yaml")
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")
            
    def save_all_configs(self):
        """Save all plugin configurations to global file"""
        global_config_path = os.path.join(self.config_dir, "plugins.yaml")
        
        all_configs = {
            name: config.to_dict() 
            for name, config in self.configs.items()
        }
        
        try:
            with open(global_config_path, 'w') as f:
                yaml.dump(all_configs, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving global plugin config: {e}")
            
    def get_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get configuration for a plugin"""
        return self.configs.get(plugin_name)
        
    def set_config(self, plugin_name: str, config: PluginConfig):
        """Set configuration for a plugin"""
        self.configs[plugin_name] = config
        
    def update_setting(self, plugin_name: str, key: str, value: Any):
        """Update a specific setting for a plugin"""
        if plugin_name not in self.configs:
            self.configs[plugin_name] = PluginConfig()
            
        self.configs[plugin_name].settings[key] = value
        
    def get_setting(self, plugin_name: str, key: str, default: Any = None) -> Any:
        """Get a specific setting for a plugin"""
        if plugin_name in self.configs:
            return self.configs[plugin_name].settings.get(key, default)
        return default
        
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        if plugin_name not in self.configs:
            self.configs[plugin_name] = PluginConfig()
        self.configs[plugin_name].enabled = True
        
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        if plugin_name not in self.configs:
            self.configs[plugin_name] = PluginConfig()
        self.configs[plugin_name].enabled = False
        
    def is_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled"""
        if plugin_name in self.configs:
            return self.configs[plugin_name].enabled
        return True  # Default to enabled
        
    def get_panel_layout(self) -> Dict[str, Dict[str, int]]:
        """Get layout information for all panel plugins"""
        layout = {}
        
        for plugin_name, config in self.configs.items():
            if config.position or config.size:
                layout[plugin_name] = {}
                if config.position:
                    layout[plugin_name]['position'] = config.position
                if config.size:
                    layout[plugin_name]['size'] = config.size
                    
        return layout
        
    def set_panel_position(self, plugin_name: str, x: int, y: int):
        """Set position for a panel plugin"""
        if plugin_name not in self.configs:
            self.configs[plugin_name] = PluginConfig()
        self.configs[plugin_name].position = {'x': x, 'y': y}
        
    def set_panel_size(self, plugin_name: str, width: int, height: int):
        """Set size for a panel plugin"""
        if plugin_name not in self.configs:
            self.configs[plugin_name] = PluginConfig()
        self.configs[plugin_name].size = {'width': width, 'height': height}
        
    def export_config(self, filepath: str):
        """Export all configurations to a file"""
        all_configs = {
            name: config.to_dict() 
            for name, config in self.configs.items()
        }
        
        try:
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml'):
                    yaml.dump(all_configs, f, default_flow_style=False)
                else:
                    json.dump(all_configs, f, indent=2)
        except Exception as e:
            print(f"Error exporting config: {e}")
            
    def import_config(self, filepath: str):
        """Import configurations from a file"""
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
                    
            for plugin_name, config_data in data.items():
                self.configs[plugin_name] = PluginConfig.from_dict(config_data)
                
        except Exception as e:
            print(f"Error importing config: {e}")