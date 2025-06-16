"""
MIDI Output Analyzer Plugin for OMEGA-4
Sends MIDI messages based on audio analysis
"""

import numpy as np
from typing import Dict, Any, Optional, List
import time

from omega4.plugins.base import AnalyzerPlugin, PluginMetadata, PluginType

# MIDI support is optional
try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("Warning: mido not installed - MIDI output disabled")


class MIDIOutputAnalyzer(AnalyzerPlugin):
    """Analyzer that outputs MIDI messages based on audio features"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="MIDI Output",
            version="1.0.0",
            author="OMEGA-4 Team",
            description="Sends MIDI messages based on audio analysis (kick, snare, pitch)",
            plugin_type=PluginType.ANALYZER,
            dependencies=[],
            config_schema={
                "midi_port": {"type": "str", "default": None},
                "kick_note": {"type": "int", "default": 36, "min": 0, "max": 127},
                "snare_note": {"type": "int", "default": 38, "min": 0, "max": 127},
                "kick_channel": {"type": "int", "default": 10, "min": 1, "max": 16},
                "snare_channel": {"type": "int", "default": 10, "min": 1, "max": 16},
                "pitch_channel": {"type": "int", "default": 1, "min": 1, "max": 16},
                "send_pitch": {"type": "bool", "default": True},
                "send_drums": {"type": "bool", "default": True},
                "velocity_scaling": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0}
            }
        )
    
    def initialize(self, config: Dict = None) -> bool:
        """Initialize MIDI output"""
        if not super().initialize(config):
            return False
            
        if not MIDI_AVAILABLE:
            print("MIDI output plugin disabled - mido not available")
            return False
            
        # Configuration
        self.midi_port_name = self._config.get("midi_port", None)
        self.kick_note = self._config.get("kick_note", 36)
        self.snare_note = self._config.get("snare_note", 38)
        self.kick_channel = self._config.get("kick_channel", 10) - 1  # 0-15
        self.snare_channel = self._config.get("snare_channel", 10) - 1
        self.pitch_channel = self._config.get("pitch_channel", 1) - 1
        self.send_pitch = self._config.get("send_pitch", True)
        self.send_drums = self._config.get("send_drums", True)
        self.velocity_scaling = self._config.get("velocity_scaling", 1.0)
        
        # MIDI state
        self.midi_port = None
        self.last_pitch_note = None
        self.last_kick_time = 0
        self.last_snare_time = 0
        self.min_note_interval = 0.05  # 50ms minimum between notes
        
        # Try to open MIDI port
        return self._open_midi_port()
    
    def _open_midi_port(self) -> bool:
        """Open MIDI output port"""
        try:
            available_ports = mido.get_output_names()
            
            if not available_ports:
                print("No MIDI output ports available")
                return False
                
            if self.midi_port_name and self.midi_port_name in available_ports:
                port_name = self.midi_port_name
            else:
                # Use first available port
                port_name = available_ports[0]
                print(f"Using MIDI port: {port_name}")
                
            self.midi_port = mido.open_output(port_name)
            print(f"✓ MIDI output opened: {port_name}")
            
            # Send test note
            self._send_test_note()
            
            return True
            
        except Exception as e:
            print(f"Failed to open MIDI port: {e}")
            return False
    
    def _send_test_note(self):
        """Send a test note to verify MIDI connection"""
        if self.midi_port:
            # Send middle C on channel 1
            msg = mido.Message('note_on', note=60, velocity=64, channel=0)
            self.midi_port.send(msg)
            time.sleep(0.1)
            msg = mido.Message('note_off', note=60, velocity=0, channel=0)
            self.midi_port.send(msg)
    
    def process(self, audio_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Process audio and send MIDI messages"""
        if not self._enabled or not self.midi_port:
            return {}
            
        results = {
            "midi_sent": False,
            "notes_triggered": []
        }
        
        current_time = time.time()
        
        # Process drum triggers
        if self.send_drums:
            drum_info = kwargs.get("drum_info", {})
            
            # Kick drum
            kick_data = drum_info.get("kick", {})
            if (kick_data.get("kick_detected", False) and 
                current_time - self.last_kick_time > self.min_note_interval):
                
                velocity = int(kick_data.get("kick_velocity", 100) * self.velocity_scaling)
                velocity = max(1, min(127, velocity))
                
                self._send_note(self.kick_note, velocity, self.kick_channel, duration=0.1)
                self.last_kick_time = current_time
                results["notes_triggered"].append(("kick", self.kick_note, velocity))
                results["midi_sent"] = True
                
            # Snare drum
            snare_data = drum_info.get("snare", {})
            if (snare_data.get("snare_detected", False) and 
                current_time - self.last_snare_time > self.min_note_interval):
                
                velocity = int(snare_data.get("snare_velocity", 100) * self.velocity_scaling)
                velocity = max(1, min(127, velocity))
                
                self._send_note(self.snare_note, velocity, self.snare_channel, duration=0.1)
                self.last_snare_time = current_time
                results["notes_triggered"].append(("snare", self.snare_note, velocity))
                results["midi_sent"] = True
        
        # Process pitch
        if self.send_pitch:
            pitch_info = kwargs.get("pitch_info", {})
            detected_pitch = pitch_info.get("pitch", 0)
            confidence = pitch_info.get("confidence", 0)
            
            if detected_pitch > 0 and confidence > 0.7:
                # Convert frequency to MIDI note
                midi_note = self._freq_to_midi(detected_pitch)
                
                if midi_note != self.last_pitch_note:
                    # Stop previous note
                    if self.last_pitch_note is not None:
                        self._stop_note(self.last_pitch_note, self.pitch_channel)
                        
                    # Start new note
                    velocity = int(80 * confidence * self.velocity_scaling)
                    velocity = max(1, min(127, velocity))
                    
                    self._send_note_on(midi_note, velocity, self.pitch_channel)
                    self.last_pitch_note = midi_note
                    results["notes_triggered"].append(("pitch", midi_note, velocity))
                    results["midi_sent"] = True
            else:
                # Stop pitch if confidence too low
                if self.last_pitch_note is not None:
                    self._stop_note(self.last_pitch_note, self.pitch_channel)
                    self.last_pitch_note = None
        
        return results
    
    def _send_note(self, note: int, velocity: int, channel: int, duration: float = 0.1):
        """Send a MIDI note with specified duration"""
        if self.midi_port:
            # Note on
            msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
            self.midi_port.send(msg)
            
            # Schedule note off (in a real implementation, this would be handled better)
            # For now, we'll send note off immediately with a small delay
            time.sleep(duration)
            msg = mido.Message('note_off', note=note, velocity=0, channel=channel)
            self.midi_port.send(msg)
    
    def _send_note_on(self, note: int, velocity: int, channel: int):
        """Send MIDI note on"""
        if self.midi_port:
            msg = mido.Message('note_on', note=note, velocity=velocity, channel=channel)
            self.midi_port.send(msg)
    
    def _stop_note(self, note: int, channel: int):
        """Send MIDI note off"""
        if self.midi_port:
            msg = mido.Message('note_off', note=note, velocity=0, channel=channel)
            self.midi_port.send(msg)
    
    def _freq_to_midi(self, frequency: float) -> int:
        """Convert frequency to MIDI note number"""
        if frequency <= 0:
            return 0
            
        # MIDI note = 69 + 12 * log2(f/440)
        midi_note = 69 + 12 * np.log2(frequency / 440.0)
        return int(round(midi_note))
    
    def reset(self):
        """Reset analyzer state"""
        # Stop any playing notes
        if self.last_pitch_note is not None:
            self._stop_note(self.last_pitch_note, self.pitch_channel)
            self.last_pitch_note = None
            
        self.last_kick_time = 0
        self.last_snare_time = 0
    
    def shutdown(self):
        """Clean up MIDI resources"""
        self.reset()
        
        if self.midi_port:
            # Send all notes off
            for channel in range(16):
                msg = mido.Message('control_change', control=123, value=0, channel=channel)
                self.midi_port.send(msg)
                
            self.midi_port.close()
            self.midi_port = None
            
        super().shutdown()
    
    def on_config_change(self):
        """Handle configuration changes"""
        # Update settings
        old_port = self.midi_port_name
        self.midi_port_name = self._config.get("midi_port", None)
        
        # Reopen port if changed
        if old_port != self.midi_port_name:
            if self.midi_port:
                self.midi_port.close()
            self._open_midi_port()
            
        # Update other settings
        self.kick_note = self._config.get("kick_note", 36)
        self.snare_note = self._config.get("snare_note", 38)
        self.kick_channel = self._config.get("kick_channel", 10) - 1
        self.snare_channel = self._config.get("snare_channel", 10) - 1
        self.pitch_channel = self._config.get("pitch_channel", 1) - 1
        self.send_pitch = self._config.get("send_pitch", True)
        self.send_drums = self._config.get("send_drums", True)
        self.velocity_scaling = self._config.get("velocity_scaling", 1.0)
    
    @staticmethod
    def list_midi_ports() -> List[str]:
        """List available MIDI output ports"""
        if not MIDI_AVAILABLE:
            return []
            
        try:
            return mido.get_output_names()
        except:
            return []