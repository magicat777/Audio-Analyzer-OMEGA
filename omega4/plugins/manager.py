"""
Plugin Manager for OMEGA-4 Audio Analyzer
Phase 6: Dynamic plugin loading and management
"""

import os
import sys
import importlib
import importlib.util
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import traceback
# Optional hot-reload support
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # Dummy base class

from .base import Plugin, PluginRegistry, PluginMetadata, PluginType


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails"""
    pass


class PluginFileHandler(FileSystemEventHandler):
    """Handle plugin file changes for hot-reload"""
    
    def __init__(self, plugin_manager):
        self.plugin_manager = plugin_manager
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if event.src_path.endswith('.py'):
            # Extract plugin name from path
            plugin_path = Path(event.src_path)
            if plugin_path.stem != '__init__':
                print(f"Plugin file modified: {plugin_path}")
                self.plugin_manager.reload_plugin(str(plugin_path))


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle"""
    
    def __init__(self, plugin_dirs: List[str] = None):
        self.registry = PluginRegistry()
        self.plugin_dirs = plugin_dirs or []
        self._loaded_modules: Dict[str, Any] = {}
        self._plugin_configs: Dict[str, Dict] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._file_observer = None
        self._hot_reload_enabled = False
        
    def add_plugin_dir(self, directory: str):
        """Add a directory to search for plugins"""
        if directory not in self.plugin_dirs:
            self.plugin_dirs.append(directory)
            
    def enable_hot_reload(self):
        """Enable hot-reload for plugin changes"""
        if not WATCHDOG_AVAILABLE:
            print("Hot-reload not available - watchdog package not installed")
            return
            
        if self._hot_reload_enabled:
            return
            
        self._hot_reload_enabled = True
        self._file_observer = Observer()
        handler = PluginFileHandler(self)
        
        for plugin_dir in self.plugin_dirs:
            if os.path.exists(plugin_dir):
                self._file_observer.schedule(handler, plugin_dir, recursive=True)
                
        self._file_observer.start()
        print("Hot-reload enabled for plugins")
        
    def disable_hot_reload(self):
        """Disable hot-reload"""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self._file_observer = None
            self._hot_reload_enabled = False
            print("Hot-reload disabled")
            
    def discover_plugins(self) -> List[str]:
        """Discover all available plugins
        
        Returns:
            List of plugin file paths
        """
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                continue
                
            # Look for Python files
            for root, dirs, files in os.walk(plugin_dir):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if file.endswith('.py') and file != '__init__.py':
                        plugin_path = os.path.join(root, file)
                        discovered.append(plugin_path)
                        
        return discovered
        
    def load_plugin(self, plugin_path: str, config: Dict = None) -> Optional[Plugin]:
        """Load a single plugin
        
        Args:
            plugin_path: Path to plugin file
            config: Optional configuration for the plugin
            
        Returns:
            Loaded plugin instance or None if loading failed
        """
        try:
            # Generate module name from path
            module_name = self._get_module_name(plugin_path)
            
            # Load or reload module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                raise PluginLoadError(f"Cannot load plugin spec from {plugin_path}")
                
            module = importlib.util.module_from_spec(spec)
            
            # Store in sys.modules for proper imports
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                raise PluginLoadError(f"No plugin class found in {plugin_path}")
                
            # Instantiate plugin
            plugin = plugin_class()
            
            # Initialize with config
            if config is None:
                config = self._plugin_configs.get(module_name, {})
                
            if not plugin.initialize(config):
                raise PluginLoadError(f"Plugin initialization failed: {module_name}")
                
            # Check dependencies
            metadata = plugin.get_metadata()
            if not self._check_dependencies(metadata):
                raise PluginLoadError(f"Missing dependencies for {metadata.name}")
                
            # Register plugin
            if self.registry.register(plugin):
                self._loaded_modules[module_name] = module
                self._update_dependency_graph(metadata)
                print(f"✓ Loaded plugin: {metadata.name} v{metadata.version}")
                return plugin
            else:
                raise PluginLoadError(f"Plugin already registered: {metadata.name}")
                
        except Exception as e:
            print(f"✗ Failed to load plugin {plugin_path}: {e}")
            traceback.print_exc()
            return None
            
    def reload_plugin(self, plugin_path: str) -> bool:
        """Reload a plugin (for hot-reload)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            module_name = self._get_module_name(plugin_path)
            
            # Find existing plugin
            old_plugin = None
            for name, plugin in self.registry.get_all().items():
                if hasattr(plugin, '__module__') and plugin.__module__ == module_name:
                    old_plugin = plugin
                    break
                    
            if old_plugin:
                # Save configuration
                config = old_plugin.get_config()
                metadata = old_plugin.get_metadata()
                
                # Shutdown old plugin
                old_plugin.shutdown()
                self.registry.unregister(metadata.name)
                
                # Remove from sys.modules to force reload
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
            # Load new version
            new_plugin = self.load_plugin(plugin_path, config)
            return new_plugin is not None
            
        except Exception as e:
            print(f"✗ Failed to reload plugin {plugin_path}: {e}")
            return False
            
    def load_all_plugins(self) -> int:
        """Load all discovered plugins
        
        Returns:
            Number of successfully loaded plugins
        """
        discovered = self.discover_plugins()
        loaded = 0
        
        print(f"Discovered {len(discovered)} plugin(s)")
        
        # Load plugin configurations first
        self._load_plugin_configs()
        
        # Sort by dependencies (simple topological sort)
        sorted_plugins = self._sort_by_dependencies(discovered)
        
        for plugin_path in sorted_plugins:
            if self.load_plugin(plugin_path):
                loaded += 1
                
        print(f"Loaded {loaded}/{len(discovered)} plugins successfully")
        return loaded
        
    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin
        
        Returns:
            True if successful, False otherwise
        """
        plugin = self.registry.get(name)
        if not plugin:
            return False
            
        try:
            # Check if other plugins depend on this one
            dependents = self._get_dependents(name)
            if dependents:
                print(f"Cannot unload {name}: required by {', '.join(dependents)}")
                return False
                
            # Shutdown plugin
            plugin.shutdown()
            
            # Unregister
            self.registry.unregister(name)
            
            # Clean up module
            if hasattr(plugin, '__module__'):
                module_name = plugin.__module__
                if module_name in self._loaded_modules:
                    del self._loaded_modules[module_name]
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
            print(f"✓ Unloaded plugin: {name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to unload plugin {name}: {e}")
            return False
            
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name"""
        return self.registry.get(name)
        
    def get_all_plugins(self) -> Dict[str, Plugin]:
        """Get all loaded plugins"""
        return self.registry.get_all()
        
    def save_plugin_config(self, name: str, config: Dict):
        """Save plugin configuration"""
        plugin = self.registry.get(name)
        if plugin:
            plugin.set_config(config)
            self._plugin_configs[name] = config
            # TODO: Persist to file
            
    def _get_module_name(self, plugin_path: str) -> str:
        """Generate module name from plugin path"""
        # Convert path to module name
        path = Path(plugin_path)
        parts = []
        
        # Walk up to find plugin directory
        current = path.parent
        while current not in [Path(d) for d in self.plugin_dirs]:
            parts.insert(0, current.name)
            current = current.parent
            if current == current.parent:  # Reached root
                break
                
        parts.append(path.stem)
        return '.'.join(['omega4_plugins'] + parts)
        
    def _find_plugin_class(self, module) -> Optional[type]:
        """Find the plugin class in a module"""
        from .base import Plugin, PanelPlugin, AnalyzerPlugin, EffectPlugin, InputPlugin, OutputPlugin
        
        # Base classes to exclude
        base_classes = {Plugin, PanelPlugin, AnalyzerPlugin, EffectPlugin, InputPlugin, OutputPlugin}
        
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, Plugin) and 
                obj not in base_classes and
                not name.startswith('_') and
                hasattr(obj, '__module__') and
                obj.__module__ == module.__name__):
                return obj
                
        return None
        
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied"""
        for dep in metadata.dependencies:
            if not self.registry.get(dep):
                print(f"Missing dependency: {dep}")
                return False
        return True
        
    def _update_dependency_graph(self, metadata: PluginMetadata):
        """Update dependency graph"""
        self._dependency_graph[metadata.name] = set(metadata.dependencies)
        
    def _get_dependents(self, name: str) -> List[str]:
        """Get plugins that depend on this one"""
        dependents = []
        for plugin_name, deps in self._dependency_graph.items():
            if name in deps:
                dependents.append(plugin_name)
        return dependents
        
    def _sort_by_dependencies(self, plugin_paths: List[str]) -> List[str]:
        """Sort plugins by dependencies (simple implementation)"""
        # For now, just return as-is
        # TODO: Implement proper topological sort
        return plugin_paths
        
    def _load_plugin_configs(self):
        """Load plugin configurations from files"""
        # TODO: Implement configuration loading from YAML/JSON files
        pass
        
    def shutdown(self):
        """Shutdown plugin manager and all plugins"""
        # Disable hot-reload
        self.disable_hot_reload()
        
        # Shutdown all plugins
        for plugin in self.registry.get_all().values():
            try:
                plugin.shutdown()
            except:
                pass