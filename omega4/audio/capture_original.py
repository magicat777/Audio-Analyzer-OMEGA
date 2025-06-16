"""
Audio Capture Module for OMEGA-4 Audio Analyzer
Phase 5: Extract audio capture and source management
"""

import numpy as np
import subprocess
import threading
import queue
import time
from typing import Optional, List, Tuple, Dict
from collections import deque


class PipeWireMonitorCapture:
    """Professional audio capture from PulseAudio/PipeWire monitor sources"""

    def __init__(
        self, source_name: str = None, sample_rate: int = 48000, chunk_size: int = 512
    ):
        self.source_name = source_name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=5)
        self.capture_process = None
        self.capture_thread = None
        self.running = False

        # Noise gating
        self.noise_floor = 0.001
        self.silence_samples = 0
        self.silence_threshold = sample_rate // 4
        self.background_level = 0.0
        self.background_alpha = 0.001

    def list_monitor_sources(self) -> List[Tuple[str, str]]:
        """List available monitor sources"""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"], capture_output=True, text=True
            )
            sources = []

            print("\n🔊 AVAILABLE OUTPUT MONITOR SOURCES:")
            print("=" * 70)

            for line in result.stdout.strip().split("\n"):
                if "monitor" in line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        source_id = parts[0]
                        source_name = parts[1]

                        # Categorize sources
                        quality_indicators = []
                        name_lower = source_name.lower()

                        if "focusrite" in name_lower or "scarlett" in name_lower:
                            quality_indicators.append("🎧 FOCUSRITE-OUTPUT")
                        elif "gsx" in name_lower or "sennheiser" in name_lower:
                            quality_indicators.append("🎮 GSX-OUTPUT")
                        elif "hdmi" in name_lower:
                            quality_indicators.append("📺 HDMI-OUTPUT")
                        else:
                            quality_indicators.append("🔊 SYSTEM-OUTPUT")

                        # Check if currently active
                        if len(parts) >= 4:
                            state = parts[3] if len(parts) > 3 else "UNKNOWN"
                            if state == "RUNNING":
                                quality_indicators.append("✅ ACTIVE")
                            elif state == "IDLE":
                                quality_indicators.append("💤 IDLE")
                            else:
                                quality_indicators.append("⚪ SUSPENDED")

                        indicators_str = " ".join(quality_indicators)

                        print(f"ID {source_id:3s}: {source_name}")
                        print(f"       {indicators_str}")
                        print()

                        sources.append((source_id, source_name))

            print("=" * 70)
            return sources

        except Exception as e:
            print(f"❌ Error listing sources: {e}")
            return []

    def select_monitor_source(self) -> Optional[str]:
        """Interactive monitor source selection"""
        sources = self.list_monitor_sources()

        if not sources:
            print("❌ No monitor sources found!")
            return None

        # Auto-select Focusrite if available
        focusrite_sources = [
            s for s in sources if "focusrite" in s[1].lower() or "scarlett" in s[1].lower()
        ]

        if len(focusrite_sources) == 1:
            selected = focusrite_sources[0]
            print(f"🎯 Auto-selected Focusrite: {selected[1]}")
            return selected[1]

        # Interactive selection
        print("📋 SELECT MONITOR SOURCE:")
        print("💡 Choose the output device you want to analyze")

        if focusrite_sources:
            print(f"🌟 RECOMMENDED: ID {focusrite_sources[0][0]} (Focusrite)")

        print(f"\nEnter source ID or press Enter for auto-select:")

        try:
            user_input = input("Source ID: ").strip()

            if user_input == "":
                if focusrite_sources:
                    selected = focusrite_sources[0]
                    print(f"🎯 Auto-selected: {selected[1]}")
                    return selected[1]
                else:
                    selected = sources[0]
                    print(f"🎯 Using first available: {selected[1]}")
                    return selected[1]
            else:
                source_id = user_input
                for sid, sname in sources:
                    if sid == source_id:
                        print(f"✅ Selected: {sname}")
                        return sname

                print(f"❌ Invalid source ID: {source_id}")
                return None

        except KeyboardInterrupt:
            print("❌ Selection cancelled")
            return None

    def start_capture(self) -> bool:
        """Start audio capture from monitor source"""
        if not self.source_name:
            self.source_name = self.select_monitor_source()

        if not self.source_name:
            print("❌ No monitor source selected")
            return False

        print(f"\n🎵 STARTING PROFESSIONAL ANALYSIS V4.1 OMEGA:")
        print(f"   Source: {self.source_name}")
        print(f"   Sample Rate: {self.sample_rate}Hz")
        print(f"   Chunk Size: {self.chunk_size} samples")

        # Start parec process
        try:
            self.capture_process = subprocess.Popen(
                [
                    "parec",
                    "--device=" + self.source_name,
                    "--format=float32le",
                    f"--rate={self.sample_rate}",
                    "--channels=1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            print("✅ Professional analysis v4.1 OMEGA started!")
            return True

        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            return False

    def _capture_loop(self):
        """Audio capture loop"""
        bytes_per_sample = 4  # float32
        chunk_bytes = self.chunk_size * bytes_per_sample

        while self.running and self.capture_process:
            try:
                # Read audio data
                data = self.capture_process.stdout.read(chunk_bytes)
                if not data:
                    break

                # Convert to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)

                if len(audio_data) == self.chunk_size:
                    # Apply noise gating
                    rms_level = np.sqrt(np.mean(audio_data**2))

                    # Update background noise estimate
                    if rms_level < self.noise_floor * 2:
                        self.background_level = (
                            1 - self.background_alpha
                        ) * self.background_level + self.background_alpha * rms_level

                    # Noise gate
                    if rms_level < max(self.noise_floor, self.background_level * 3):
                        self.silence_samples += self.chunk_size
                        if self.silence_samples > self.silence_threshold:
                            # Send silence
                            audio_data = np.zeros_like(audio_data)
                    else:
                        self.silence_samples = 0

                    # Put in queue
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        # Drop oldest sample
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(audio_data)
                        except:
                            pass

            except Exception as e:
                print(f"❌ Capture error: {e}")
                break

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get latest audio data"""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_capture(self):
        """Stop audio capture"""
        self.running = False

        if self.capture_process:
            try:
                self.capture_process.terminate()
                self.capture_process.wait(timeout=2)
            except:
                try:
                    self.capture_process.kill()
                except:
                    pass
            self.capture_process = None

        if self.capture_thread:
            self.capture_thread.join(timeout=1)


class AudioCaptureManager:
    """High-level audio capture management with device selection"""
    
    def __init__(self, sample_rate: int = 48000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.capture = None
        
    def start(self, device: Optional[str] = None) -> bool:
        """Start audio capture with optional device specification"""
        self.capture = PipeWireMonitorCapture(
            source_name=device,
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size
        )
        return self.capture.start_capture()
        
    def stop(self):
        """Stop audio capture"""
        if self.capture:
            self.capture.stop_capture()
            self.capture = None
            
    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get latest audio data"""
        if self.capture:
            return self.capture.get_audio_data()
        return None
        
    def is_running(self) -> bool:
        """Check if capture is running"""
        return self.capture is not None and self.capture.running