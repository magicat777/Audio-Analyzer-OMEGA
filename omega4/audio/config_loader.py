"""
Configuration loading and management for OMEGA-4 Audio Analyzer
Handles loading from files, environment variables, and saving configurations
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import dataclasses

from .audio_config import PipelineConfig, LinuxAudioConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading, saving, and merging configurations"""
    
    DEFAULT_CONFIG_PATH = Path.home() / ".config" / "omega4" / "config.json"
    
    @staticmethod
    def load_from_file(config_path: Optional[str] = None) -> PipelineConfig:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to config file. If None, uses default location
            
        Returns:
            PipelineConfig instance
        """
        if config_path is None:
            config_path = ConfigManager.DEFAULT_CONFIG_PATH
        
        path = Path(config_path)
        
        if not path.exists():
            logger.info(f"Config file not found at {path}, using defaults")
            return PipelineConfig()
            
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            
            # Validate and create config
            return PipelineConfig(**config_dict)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return PipelineConfig()
        except TypeError as e:
            logger.error(f"Invalid configuration parameters: {e}")
            return PipelineConfig()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return PipelineConfig()
    
    @staticmethod
    def save_to_file(config: PipelineConfig, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file
        
        Args:
            config: PipelineConfig to save
            config_path: Path to save to. If None, uses default location
            
        Returns:
            True if successful, False otherwise
        """
        if config_path is None:
            config_path = ConfigManager.DEFAULT_CONFIG_PATH
            
        path = Path(config_path)
        
        try:
            # Create directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict
            config_dict = dataclasses.asdict(config)
            
            # Save with pretty printing
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def load_linux_config(config_path: Optional[str] = None) -> LinuxAudioConfig:
        """Load Linux-specific audio configuration"""
        if config_path is None:
            config_path = Path.home() / ".config" / "omega4" / "linux_audio.json"
        
        path = Path(config_path)
        
        if not path.exists():
            return LinuxAudioConfig()
            
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return LinuxAudioConfig(**config_dict)
        except Exception as e:
            logger.error(f"Error loading Linux audio config: {e}")
            return LinuxAudioConfig()
    
    @staticmethod
    def merge_with_env(config: PipelineConfig) -> PipelineConfig:
        """
        Merge configuration with environment variables
        
        Environment variables override file configuration
        Format: OMEGA4_<PARAMETER_NAME> (uppercase)
        """
        config_dict = dataclasses.asdict(config)
        
        # Check for environment overrides
        for field in dataclasses.fields(PipelineConfig):
            env_name = f"OMEGA4_{field.name.upper()}"
            env_value = os.environ.get(env_name)
            
            if env_value is not None:
                try:
                    # Convert based on type
                    if field.type == int:
                        config_dict[field.name] = int(env_value)
                    elif field.type == float:
                        config_dict[field.name] = float(env_value)
                    elif field.type == bool:
                        config_dict[field.name] = env_value.lower() in ('true', '1', 'yes')
                    else:
                        config_dict[field.name] = env_value
                        
                    logger.info(f"Override {field.name} from environment: {env_value}")
                    
                except ValueError as e:
                    logger.warning(f"Invalid env value for {env_name}: {e}")
        
        return PipelineConfig(**config_dict)
    
    @staticmethod
    def create_default_config_files():
        """Create default configuration files in user's config directory"""
        config_dir = Path.home() / ".config" / "omega4"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Save default pipeline config
        default_pipeline = PipelineConfig()
        ConfigManager.save_to_file(default_pipeline, config_dir / "config.json")
        
        # Save default Linux audio config
        default_linux = LinuxAudioConfig()
        linux_path = config_dir / "linux_audio.json"
        
        try:
            with open(linux_path, 'w') as f:
                json.dump(dataclasses.asdict(default_linux), f, indent=2)
            logger.info(f"Created default configs in {config_dir}")
            return True
        except Exception as e:
            logger.error(f"Error creating default configs: {e}")
            return False
    
    @staticmethod
    def validate_system_requirements(config: PipelineConfig) -> Dict[str, bool]:
        """
        Validate that system meets configuration requirements
        
        Returns:
            Dict with validation results
        """
        results = {}
        
        # Check CPU cores for threading
        import os
        cpu_count = os.cpu_count() or 1
        results['sufficient_cores'] = cpu_count >= config.max_worker_threads
        
        # Check available memory
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            results['sufficient_memory'] = available_mb >= config.max_memory_usage_mb
        except ImportError:
            results['sufficient_memory'] = True  # Assume OK if can't check
        
        # Check sample rate support (basic check)
        valid_rates = [44100, 48000, 96000, 192000]
        results['valid_sample_rate'] = config.sample_rate in valid_rates
        
        return results


# Convenience functions
def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load and merge configuration from file and environment"""
    config = ConfigManager.load_from_file(path)
    return ConfigManager.merge_with_env(config)


def save_config(config: PipelineConfig, path: Optional[str] = None) -> bool:
    """Save configuration to file"""
    return ConfigManager.save_to_file(config, path)