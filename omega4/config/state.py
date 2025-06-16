"""
Persistent State Manager for OMEGA-4 Audio Analyzer
Phase 7: Handle application state persistence between sessions
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class WindowState:
    """Window position and size state"""
    x: int = 100
    y: int = 100
    width: int = 2000
    height: int = 1080
    maximized: bool = False
    fullscreen: bool = False
    monitor: int = 0


@dataclass
class SessionState:
    """Session-specific state"""
    last_preset: Optional[str] = None
    last_audio_device: Optional[str] = None
    input_gain: float = 4.0
    panel_visibility: Dict[str, bool] = field(default_factory=dict)
    plugin_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recent_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Default panel visibility
        default_panels = {
            "professional_meters": True,
            "vu_meters": True,
            "bass_zoom": True,
            "harmonic_analysis": True,
            "pitch_detection": True,
            "chromagram": True,
            "genre_classification": True,
            "spectrogram": False,
            "waterfall": False
        }
        
        # Apply defaults for missing panels
        for panel, visible in default_panels.items():
            if panel not in self.panel_visibility:
                self.panel_visibility[panel] = visible


@dataclass
class AppState:
    """Complete application state"""
    version: str = "1.0.0"
    window: WindowState = field(default_factory=WindowState)
    session: SessionState = field(default_factory=SessionState)
    last_session: str = ""
    total_runtime: float = 0.0  # Total runtime in seconds
    session_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppState':
        """Create from dictionary"""
        # Handle nested dataclasses
        if "window" in data and isinstance(data["window"], dict):
            data["window"] = WindowState(**data["window"])
            
        if "session" in data and isinstance(data["session"], dict):
            data["session"] = SessionState(**data["session"])
            
        return cls(**data)


class StateManager:
    """Manages persistent application state"""
    
    def __init__(self, state_dir: str = None):
        # Default to user state directory
        if state_dir is None:
            state_dir = os.path.join(os.path.expanduser("~"), ".omega4")
            
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / "state.json"
        
        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Current state
        self.state: AppState = None
        self.session_start_time = datetime.now()
        
        # Auto-save timer
        self._last_save_time = 0
        self._auto_save_interval = 60  # Save every 60 seconds
        
    def load_state(self) -> AppState:
        """Load application state from file
        
        Returns:
            Loaded state or default if not found
        """
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    
                self.state = AppState.from_dict(data)
                
                # Update session info
                self.state.session_count += 1
                self.state.last_session = datetime.now().isoformat()
                
                print(f"✓ State loaded (session #{self.state.session_count})")
                
            except Exception as e:
                print(f"✗ Failed to load state: {e}")
                self.state = AppState()
        else:
            # First run
            print("No state file found, creating new state")
            self.state = AppState()
            self.state.session_count = 1
            self.state.last_session = datetime.now().isoformat()
            
        return self.state
        
    def save_state(self, force: bool = False) -> bool:
        """Save application state to file
        
        Args:
            force: Force save even if auto-save interval hasn't elapsed
            
        Returns:
            True if saved
        """
        if self.state is None:
            return False
            
        # Check auto-save interval
        current_time = datetime.now().timestamp()
        if not force and (current_time - self._last_save_time) < self._auto_save_interval:
            return False
            
        try:
            # Update runtime
            session_duration = (datetime.now() - self.session_start_time).total_seconds()
            self.state.total_runtime += session_duration
            self.session_start_time = datetime.now()  # Reset for next interval
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
                
            self._last_save_time = current_time
            return True
            
        except Exception as e:
            print(f"✗ Failed to save state: {e}")
            return False
            
    def update_window_state(self, x: int, y: int, width: int, height: int,
                          maximized: bool = False, fullscreen: bool = False):
        """Update window position and size"""
        if self.state:
            self.state.window.x = x
            self.state.window.y = y
            self.state.window.width = width
            self.state.window.height = height
            self.state.window.maximized = maximized
            self.state.window.fullscreen = fullscreen
            
    def get_window_state(self) -> WindowState:
        """Get current window state"""
        if self.state:
            return self.state.window
        return WindowState()
        
    def update_panel_visibility(self, panel_name: str, visible: bool):
        """Update panel visibility state"""
        if self.state:
            self.state.session.panel_visibility[panel_name] = visible
            
    def get_panel_visibility(self, panel_name: str) -> bool:
        """Get panel visibility state"""
        if self.state:
            return self.state.session.panel_visibility.get(panel_name, True)
        return True
        
    def set_last_preset(self, preset_name: str):
        """Set last used preset"""
        if self.state:
            self.state.session.last_preset = preset_name
            
    def get_last_preset(self) -> Optional[str]:
        """Get last used preset"""
        if self.state:
            return self.state.session.last_preset
        return None
        
    def set_last_audio_device(self, device_name: str):
        """Set last used audio device"""
        if self.state:
            self.state.session.last_audio_device = device_name
            
    def get_last_audio_device(self) -> Optional[str]:
        """Get last used audio device"""
        if self.state:
            return self.state.session.last_audio_device
        return None
        
    def update_input_gain(self, gain: float):
        """Update input gain state"""
        if self.state:
            self.state.session.input_gain = gain
            
    def get_input_gain(self) -> float:
        """Get input gain state"""
        if self.state:
            return self.state.session.input_gain
        return 4.0
        
    def save_plugin_state(self, plugin_name: str, state: Dict[str, Any]):
        """Save plugin-specific state"""
        if self.state:
            self.state.session.plugin_states[plugin_name] = state
            
    def get_plugin_state(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin-specific state"""
        if self.state:
            return self.state.session.plugin_states.get(plugin_name, {})
        return {}
        
    def add_recent_file(self, file_path: str, max_recent: int = 10):
        """Add file to recent files list"""
        if self.state:
            # Remove if already in list
            if file_path in self.state.session.recent_files:
                self.state.session.recent_files.remove(file_path)
                
            # Add to front
            self.state.session.recent_files.insert(0, file_path)
            
            # Limit size
            self.state.session.recent_files = self.state.session.recent_files[:max_recent]
            
    def get_recent_files(self) -> List[str]:
        """Get recent files list"""
        if self.state:
            return self.state.session.recent_files.copy()
        return []
        
    def clear_recent_files(self):
        """Clear recent files list"""
        if self.state:
            self.state.session.recent_files.clear()
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if self.state:
            # Calculate current session duration
            current_session = (datetime.now() - self.session_start_time).total_seconds()
            total_runtime = self.state.total_runtime + current_session
            
            return {
                "total_runtime_hours": total_runtime / 3600,
                "session_count": self.state.session_count,
                "current_session_minutes": current_session / 60,
                "average_session_minutes": (total_runtime / self.state.session_count) / 60
            }
        return {}
        
    def reset_state(self):
        """Reset state to defaults"""
        self.state = AppState()
        self.state.session_count = 1
        self.state.last_session = datetime.now().isoformat()
        self.save_state(force=True)
        
    def auto_save(self):
        """Perform auto-save if needed"""
        self.save_state(force=False)