"""
Audio Capture Module for OMEGA-4 Audio Analyzer
Phase 5: Extract audio capture and source management
Enhanced with robust process management, validation, and performance monitoring
"""

import numpy as np
import subprocess
import threading
import queue
import time
import signal
import os
import select
import logging
from typing import Optional, List, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioCaptureConfig:
    """Configuration for audio capture"""
    # Audio settings
    sample_rate: int = 48000
    chunk_size: int = 512
    audio_format: str = "float32le"
    channels: int = 1
    
    # Buffer settings
    buffer_size: int = 10
    queue_timeout: float = 0.1
    
    # Noise gate settings
    noise_floor: float = 0.001
    silence_threshold_seconds: float = 0.25
    background_alpha: float = 0.001
    
    # Process management
    process_timeout: float = 5.0
    capture_timeout: float = 30.0
    
    # Performance monitoring
    enable_stats: bool = True
    stats_interval: float = 5.0
    
    # Device selection
    prefer_focusrite: bool = True
    auto_select: bool = False
    
    # Error recovery
    max_consecutive_errors: int = 5
    restart_delay: float = 1.0
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if self.sample_rate <= 0 or self.sample_rate > 192000:
            issues.append("Sample rate must be between 1 and 192000")
            
        if self.chunk_size <= 0 or self.chunk_size > 8192:
            issues.append("Chunk size must be between 1 and 8192")
            
        if self.channels < 1 or self.channels > 8:
            issues.append("Channels must be between 1 and 8")
            
        if self.audio_format not in ["float32le", "s16le", "s32le"]:
            issues.append("Unsupported audio format")
            
        if self.buffer_size < 1 or self.buffer_size > 100:
            issues.append("Buffer size must be between 1 and 100")
            
        if self.noise_floor < 0 or self.noise_floor > 1:
            issues.append("Noise floor must be between 0 and 1")
            
        return issues


class CircularAudioBuffer:
    """High-performance circular buffer for audio data"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer = [None] * max_size
        self.write_index = 0
        self.read_index = 0
        self.count = 0
        self.lock = threading.Lock()
        self.dropped_frames = 0
        
    def put(self, data: np.ndarray) -> bool:
        """Add data to buffer, returns True if successful"""
        with self.lock:
            if self.count >= self.max_size:
                # Overwrite oldest
                self.read_index = (self.read_index + 1) % self.max_size
                self.count -= 1
                self.dropped_frames += 1
                
            self.buffer[self.write_index] = data.copy()
            self.write_index = (self.write_index + 1) % self.max_size
            self.count += 1
            return True
            
    def get(self) -> Optional[np.ndarray]:
        """Get data from buffer"""
        with self.lock:
            if self.count == 0:
                return None
                
            data = self.buffer[self.read_index]
            self.read_index = (self.read_index + 1) % self.max_size
            self.count -= 1
            return data
            
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return self.count == 0
        
    def clear(self):
        """Clear all data"""
        with self.lock:
            self.count = 0
            self.read_index = 0
            self.write_index = 0
            self.dropped_frames = 0
            
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "count": self.count,
                "capacity": self.max_size,
                "dropped_frames": self.dropped_frames
            }


class AudioCaptureStats:
    """Performance monitoring for audio capture"""
    
    def __init__(self):
        self.frames_processed = 0
        self.frames_dropped = 0
        self.total_latency = 0.0
        self.max_latency = 0.0
        self.min_latency = float('inf')
        self.start_time = time.time()
        self.last_stats_time = time.time()
        self.lock = threading.Lock()
        
    def record_frame(self, processing_time: float):
        """Record frame processing statistics"""
        with self.lock:
            self.frames_processed += 1
            self.total_latency += processing_time
            self.max_latency = max(self.max_latency, processing_time)
            self.min_latency = min(self.min_latency, processing_time)
        
    def record_dropped_frame(self):
        """Record dropped frame"""
        with self.lock:
            self.frames_dropped += 1
        
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        with self.lock:
            elapsed = time.time() - self.start_time
            current_fps = self.frames_processed / elapsed if elapsed > 0 else 0
            total_frames = self.frames_processed + self.frames_dropped
            drop_rate = self.frames_dropped / max(1, total_frames)
            avg_latency = self.total_latency / max(1, self.frames_processed)
            
            return {
                "fps": current_fps,
                "drop_rate_percent": drop_rate * 100,
                "avg_latency_ms": avg_latency * 1000,
                "max_latency_ms": self.max_latency * 1000,
                "min_latency_ms": self.min_latency * 1000 if self.min_latency != float('inf') else 0,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "uptime_seconds": elapsed
            }
            
    def reset(self):
        """Reset statistics"""
        with self.lock:
            self.frames_processed = 0
            self.frames_dropped = 0
            self.total_latency = 0.0
            self.max_latency = 0.0
            self.min_latency = float('inf')
            self.start_time = time.time()
            self.last_stats_time = time.time()


class PipeWireMonitorCapture:
    """Professional audio capture from PulseAudio/PipeWire monitor sources"""

    def __init__(self, source_name: str = None, config: Optional[AudioCaptureConfig] = None):
        self.source_name = source_name
        self.config = config or AudioCaptureConfig()
        
        # Validate configuration
        issues = self.config.validate()
        if issues:
            raise ValueError(f"Invalid configuration: {', '.join(issues)}")
        
        # Audio buffer (replaced queue with circular buffer)
        self.audio_buffer = CircularAudioBuffer(max_size=self.config.buffer_size)
        
        # Process management
        self.capture_process = None
        self.capture_thread = None
        self.running = False
        self.cleanup_lock = threading.Lock()
        self._is_cleaning_up = False
        
        # Performance monitoring
        self.stats = AudioCaptureStats()
        
        # Noise gating
        self.silence_samples = 0
        self.silence_threshold = int(self.config.sample_rate * self.config.silence_threshold_seconds)
        self.background_level = 0.0
        
        # Register cleanup handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"PipeWireMonitorCapture initialized: {self.config.sample_rate}Hz, "
                   f"{self.config.chunk_size} samples")

    def _signal_handler(self, signum, frame):
        """Handle system signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_capture()

    def validate_audio_device(self, source_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate audio device capabilities and settings"""
        try:
            # Check if PulseAudio is available
            result = subprocess.run(
                ["pactl", "info"], 
                capture_output=True, 
                text=True, 
                timeout=5.0
            )
            
            if result.returncode != 0:
                return False, {"error": "PulseAudio not available"}

            # Get detailed device info
            device_result = subprocess.run(
                ["pactl", "list", "sources"], 
                capture_output=True, 
                text=True,
                timeout=10.0
            )
            
            device_info = self._parse_device_info(device_result.stdout, source_name)
            
            # Test if device supports required sample rate
            if not self._check_sample_rate_support(source_name, self.config.sample_rate):
                return False, {
                    "error": f"Sample rate {self.config.sample_rate}Hz not supported",
                    "device_info": device_info
                }
                
            device_info["validation"] = "passed"
            return True, device_info
            
        except subprocess.TimeoutExpired:
            return False, {"error": "Device validation timeout"}
        except Exception as e:
            return False, {"error": f"Validation failed: {str(e)}"}

    def _parse_device_info(self, pactl_output: str, source_name: str) -> Dict[str, Any]:
        """Parse device information from pactl output"""
        device_info = {
            "name": source_name,
            "sample_rates": [],
            "formats": [],
            "channels": 0,
            "state": "unknown",
            "description": "",
            "properties": {}
        }
        
        lines = pactl_output.split('\n')
        in_target_source = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if f"Name: {source_name}" in line:
                in_target_source = True
                continue
                
            if in_target_source:
                if line.startswith("Source #") and source_name not in line:
                    # We've moved to another source
                    break
                    
                if "State:" in line:
                    device_info["state"] = line.split("State:")[-1].strip()
                elif "Description:" in line:
                    device_info["description"] = line.split("Description:")[-1].strip()
                elif "Sample Specification:" in line:
                    # Parse sample spec: s16le 2ch 44100Hz
                    spec = line.split(":")[-1].strip()
                    if "Hz" in spec:
                        rate_str = spec.split("Hz")[0].split()[-1]
                        try:
                            device_info["sample_rates"].append(int(rate_str))
                        except ValueError:
                            pass
                    if "ch" in spec:
                        ch_str = spec.split("ch")[0].split()[-1]
                        try:
                            device_info["channels"] = int(ch_str)
                        except ValueError:
                            pass
                elif "Formats:" in line:
                    device_info["formats"] = line.split(":")[-1].strip().split()
                        
        return device_info

    def _check_sample_rate_support(self, source_name: str, sample_rate: int) -> bool:
        """Test if device supports the required sample rate"""
        try:
            # Test with a very short capture
            test_process = subprocess.Popen(
                [
                    "timeout", "1s", "parec",
                    f"--device={source_name}",
                    f"--format={self.config.audio_format}",
                    f"--rate={sample_rate}",
                    f"--channels={self.config.channels}"
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            
            _, stderr = test_process.communicate(timeout=2.0)
            stderr_text = stderr.decode().lower()
            
            # Check for sample rate errors
            if "rate" in stderr_text or "format" in stderr_text or "invalid" in stderr_text:
                return False
                
            return test_process.returncode in [0, 124]  # 124 = timeout (expected)
            
        except Exception as e:
            logger.error(f"Sample rate test failed: {e}")
            return False

    def list_monitor_sources(self) -> List[Tuple[str, str]]:
        """List available monitor sources"""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"], 
                capture_output=True, 
                text=True,
                timeout=5.0
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

        except subprocess.TimeoutExpired:
            print("❌ Timeout while listing sources")
            return []
        except Exception as e:
            print(f"❌ Error listing sources: {e}")
            return []

    def select_monitor_source(self) -> Optional[str]:
        """Interactive monitor source selection"""
        sources = self.list_monitor_sources()

        if not sources:
            print("❌ No monitor sources found!")
            return None

        # Auto-select Focusrite if configured
        if self.config.prefer_focusrite:
            focusrite_sources = [
                s for s in sources 
                if "focusrite" in s[1].lower() or "scarlett" in s[1].lower()
            ]

            if len(focusrite_sources) == 1:
                selected = focusrite_sources[0]
                print(f"🎯 Auto-selected Focusrite: {selected[1]}")
                return selected[1]

        # Auto-select first available if configured
        if self.config.auto_select and sources:
            selected = sources[0]
            print(f"🎯 Auto-selected first available: {selected[1]}")
            return selected[1]

        # Interactive selection
        print("📋 SELECT MONITOR SOURCE:")
        print("💡 Choose the output device you want to analyze")

        focusrite_sources = [
            s for s in sources 
            if "focusrite" in s[1].lower() or "scarlett" in s[1].lower()
        ]
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
            print("\n❌ Selection cancelled")
            return None

    def start_capture(self) -> bool:
        """Start audio capture from monitor source"""
        if not self.source_name:
            self.source_name = self.select_monitor_source()

        if not self.source_name:
            print("❌ No monitor source selected")
            return False

        # Validate device
        is_valid, device_info = self.validate_audio_device(self.source_name)
        if not is_valid:
            print(f"❌ Device validation failed: {device_info.get('error', 'Unknown error')}")
            return False

        print(f"\n🎵 STARTING PROFESSIONAL ANALYSIS V4.1 OMEGA:")
        print(f"   Source: {self.source_name}")
        print(f"   Sample Rate: {self.config.sample_rate}Hz")
        print(f"   Chunk Size: {self.config.chunk_size} samples")
        print(f"   Device State: {device_info.get('state', 'unknown')}")

        # Start capture process
        return self._start_parec_process()

    def _start_parec_process(self) -> bool:
        """Start the parec capture process"""
        try:
            self.capture_process = subprocess.Popen(
                [
                    "parec",
                    f"--device={self.source_name}",
                    f"--format={self.config.audio_format}",
                    f"--rate={self.config.sample_rate}",
                    f"--channels={self.config.channels}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Reset statistics
            self.stats.reset()
            self.audio_buffer.clear()

            print("✅ Professional analysis v4.1 OMEGA started!")
            
            # Start stats reporting if enabled
            if self.config.enable_stats:
                self._start_stats_reporting()
                
            return True

        except Exception as e:
            print(f"❌ Failed to start audio capture: {e}")
            return False

    def _capture_loop(self):
        """Enhanced capture loop with error recovery"""
        bytes_per_sample = 4 if self.config.audio_format == "float32le" else 2
        chunk_bytes = self.config.chunk_size * bytes_per_sample
        consecutive_errors = 0
        
        while self.running and self.capture_process:
            frame_start_time = time.time()
            
            try:
                # Read audio data with timeout
                data = self._read_with_timeout(chunk_bytes, timeout=1.0)
                
                if not data:
                    consecutive_errors += 1
                    if consecutive_errors >= self.config.max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors})")
                        break
                    continue
                    
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Convert to numpy array
                if self.config.audio_format == "float32le":
                    audio_data = np.frombuffer(data, dtype=np.float32)
                else:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if len(audio_data) == self.config.chunk_size:
                    # Process audio frame
                    processed_audio = self._process_audio_frame(audio_data)
                    
                    # Store in buffer
                    if not self.audio_buffer.put(processed_audio):
                        self.stats.record_dropped_frame()
                    
                    # Record performance stats
                    if self.config.enable_stats:
                        processing_time = time.time() - frame_start_time
                        self.stats.record_frame(processing_time)
                        
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Capture error ({consecutive_errors}/{self.config.max_consecutive_errors}): {e}")
                
                if consecutive_errors >= self.config.max_consecutive_errors:
                    logger.error("Maximum errors reached, attempting to restart capture...")
                    if self._attempt_restart():
                        consecutive_errors = 0
                    else:
                        break
                
                time.sleep(0.1)  # Brief pause before retry

    def _read_with_timeout(self, chunk_bytes: int, timeout: float) -> Optional[bytes]:
        """Read data with timeout to prevent hanging"""
        if not self.capture_process or not self.capture_process.stdout:
            return None
            
        # Use select to check if data is available
        ready, _, _ = select.select([self.capture_process.stdout], [], [], timeout)
        
        if ready:
            try:
                return self.capture_process.stdout.read(chunk_bytes)
            except Exception as e:
                logger.error(f"Read error: {e}")
                return None
        else:
            # Timeout occurred
            return None

    def _process_audio_frame(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio frame with noise gating"""
        # Calculate RMS level
        rms_level = np.sqrt(np.mean(audio_data**2))
        
        # Update background noise estimate
        if rms_level < self.config.noise_floor * 2:
            self.background_level = (
                (1 - self.config.background_alpha) * self.background_level + 
                self.config.background_alpha * rms_level
            )
        
        # Apply noise gate
        if rms_level < max(self.config.noise_floor, self.background_level * 3):
            self.silence_samples += self.config.chunk_size
            if self.silence_samples > self.silence_threshold:
                # Send silence
                return np.zeros_like(audio_data)
        else:
            self.silence_samples = 0
            
        return audio_data

    def _attempt_restart(self) -> bool:
        """Attempt to restart the capture process"""
        logger.info("Attempting to restart audio capture...")
        
        try:
            # Clean up current process
            if self.capture_process:
                self.capture_process.terminate()
                try:
                    self.capture_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.capture_process.kill()
                    self.capture_process.wait(timeout=1.0)
                self.capture_process = None
            
            # Brief pause
            time.sleep(self.config.restart_delay)
            
            # Restart
            return self._start_parec_process()
            
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            return False

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get latest audio data from buffer"""
        return self.audio_buffer.get()

    def stop_capture(self):
        """Enhanced stop with proper cleanup"""
        with self.cleanup_lock:
            if self._is_cleaning_up:
                return
            self._is_cleaning_up = True
            
        logger.info("Stopping audio capture...")
        self.running = False

        # Stop capture thread first
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread didn't stop gracefully")

        # Clean up process
        if self.capture_process:
            try:
                # First try gentle termination
                self.capture_process.terminate()
                try:
                    self.capture_process.wait(timeout=self.config.process_timeout)
                    logger.info("Audio process terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate, forcing kill...")
                    self.capture_process.kill()
                    self.capture_process.wait(timeout=2.0)
                    
            except Exception as e:
                logger.error(f"Error during process cleanup: {e}")
                # Force kill if still running
                try:
                    if self.capture_process.poll() is None:
                        os.kill(self.capture_process.pid, signal.SIGKILL)
                except:
                    pass
            finally:
                self.capture_process = None

        # Clear buffer
        self.audio_buffer.clear()
        
        logger.info("Audio capture stopped")

    def _start_stats_reporting(self):
        """Start periodic stats reporting"""
        def report_stats():
            while self.running:
                time.sleep(self.config.stats_interval)
                if self.running:
                    stats = self.get_stats()
                    logger.info(f"Audio capture stats: FPS={stats['fps']:.1f}, "
                              f"Drop rate={stats['drop_rate_percent']:.2f}%, "
                              f"Avg latency={stats['avg_latency_ms']:.2f}ms")
        
        stats_thread = threading.Thread(target=report_stats, daemon=True)
        stats_thread.start()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive capture statistics"""
        capture_stats = self.stats.get_stats()
        buffer_stats = self.audio_buffer.get_stats()
        
        return {
            **capture_stats,
            "buffer_count": buffer_stats["count"],
            "buffer_capacity": buffer_stats["capacity"],
            "buffer_dropped": buffer_stats["dropped_frames"],
            "device": self.source_name,
            "sample_rate": self.config.sample_rate,
            "chunk_size": self.config.chunk_size
        }


class AudioCaptureManager:
    """High-level audio capture management with device selection"""
    
    def __init__(self, config: Optional[AudioCaptureConfig] = None):
        self.config = config or AudioCaptureConfig()
        self.capture = None
        
    def start(self, device: Optional[str] = None) -> bool:
        """Start audio capture with optional device specification"""
        self.capture = PipeWireMonitorCapture(
            source_name=device,
            config=self.config
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
        
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """Get capture statistics"""
        if self.capture:
            return self.capture.get_stats()
        return None