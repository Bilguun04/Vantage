#!/usr/bin/env python3
"""
Mock Client for Gemini Live Voice Assistant

A standalone test client that captures screen and microphone input,
sends them to the Django WebSocket server, and plays back audio responses.

Usage:
    python mock_client.py [--server ws://localhost:8000/ws/live-session/]
    python mock_client.py --ptt         # Enable push-to-talk mode (Right Ctrl)
    python mock_client.py --ptt-key f5  # Use F5 as push-to-talk key

Requirements:
    pip install websockets sounddevice numpy mss pillow pynput
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import threading
from queue import Queue

# Screen capture
import mss
from PIL import Image

# Audio (using sounddevice - better macOS support than pyaudio)
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio capture/playback disabled.")
    print("Install with: pip install sounddevice numpy")

# Global hotkey support for push-to-talk
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Push-to-talk disabled.")
    print("Install with: pip install pynput")

# WebSocket
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio configuration (matching Gemini's expected format)
CHANNELS = 1
SAMPLE_RATE = 16000  # 16kHz for input
OUTPUT_SAMPLE_RATE = 24000  # 24kHz for Gemini output
CHUNK_SIZE = 4096  # Larger chunks = fewer sends
DTYPE = np.int16

# Demo mode settings (token-conscious)
AUDIO_SEND_INTERVAL = 0.1  # Send audio every 100ms (not continuously)
SCREEN_CAPTURE_INTERVAL = 30.0  # Send screen every 30 seconds (reduced for stability)
SCREEN_MAX_SIZE = (640, 360)  # 360p resolution to reduce payload size
ENABLE_SCREEN_CAPTURE = True  # Set to False to disable screen capture for testing
ENABLE_AUDIO_CAPTURE = True  # Set to False to disable audio capture/sending

# Voice activity detection settings (used when PTT is disabled)
SILENCE_THRESHOLD = 100  # Volume level below this = silence
SILENCE_DURATION_TO_END_TURN = 1.5  # Seconds of silence before signaling end of speech
SPEECH_DETECTED_THRESHOLD = 200  # Volume level to consider as speech starting

# Push-to-talk key mapping (string name -> pynput key)
PTT_KEY_MAP = {}
if PYNPUT_AVAILABLE:
    PTT_KEY_MAP = {
        # Character keys (for PTT that doesn't conflict with system shortcuts)
        'grave': keyboard.KeyCode.from_char('`'),  # Backtick - classic PTT key
        'backslash': keyboard.KeyCode.from_char('\\'),
        'insert': keyboard.Key.insert if hasattr(keyboard.Key, 'insert') else None,
        # Special keys
        'ctrl_r': keyboard.Key.ctrl_r,
        'ctrl_l': keyboard.Key.ctrl_l,
        'alt_r': keyboard.Key.alt_r,
        'alt_l': keyboard.Key.alt_l,
        'shift_r': keyboard.Key.shift_r,
        'shift_l': keyboard.Key.shift_l,
        'space': keyboard.Key.space,
        'tab': keyboard.Key.tab,
        'caps_lock': keyboard.Key.caps_lock,
        # Function keys
        'f1': keyboard.Key.f1,
        'f2': keyboard.Key.f2,
        'f3': keyboard.Key.f3,
        'f4': keyboard.Key.f4,
        'f5': keyboard.Key.f5,
        'f6': keyboard.Key.f6,
        'f7': keyboard.Key.f7,
        'f8': keyboard.Key.f8,
        'f9': keyboard.Key.f9,
        'f10': keyboard.Key.f10,
        'f11': keyboard.Key.f11,
        'f12': keyboard.Key.f12,
    }
    # Remove None entries (keys not available on this platform)
    PTT_KEY_MAP = {k: v for k, v in PTT_KEY_MAP.items() if v is not None}
    # Add scroll_lock only if available (not on macOS)
    if hasattr(keyboard.Key, 'scroll_lock'):
        PTT_KEY_MAP['scroll_lock'] = keyboard.Key.scroll_lock

DEFAULT_PTT_KEY = 'grave'  # Backtick key - classic PTT, no system conflicts


class AudioCapture:
    """Handles microphone input capture using sounddevice."""

    def __init__(self, audio_queue: Queue):
        self.audio_queue = audio_queue
        self.is_running = False
        self.stream = None

    def start(self):
        """Start audio capture."""
        if not AUDIO_AVAILABLE:
            logger.warning("sounddevice not available, skipping audio capture")
            return

        self.is_running = True

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_SIZE,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("Audio capture started (sounddevice)")
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.is_running = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio input status: {status}")
        if self.is_running:
            # Convert numpy array to bytes
            self.audio_queue.put(indata.copy().tobytes())

    def stop(self):
        """Stop audio capture."""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("Audio capture stopped")


class AudioPlayback:
    """Handles audio output playback using sounddevice with buffering."""

    def __init__(self):
        self.playback_queue = Queue()
        self.is_running = False
        self.stream = None

    def start(self):
        """Start audio playback stream."""
        if not AUDIO_AVAILABLE:
            logger.warning("sounddevice not available, skipping audio playback")
            return

        self.is_running = True
        try:
            # Use a continuous output stream for smoother playback
            self.stream = sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=2048,  # Larger blocks for smoother playback
            )
            self.stream.start()
            # Start playback thread
            self.thread = threading.Thread(target=self._playback_loop)
            self.thread.start()
            logger.info("Audio playback started (sounddevice streaming)")
        except Exception as e:
            logger.error(f"Failed to start audio playback: {e}")
            self.is_running = False

    def _playback_loop(self):
        """Loop to play audio from queue continuously."""
        while self.is_running:
            try:
                audio_data = self.playback_queue.get(timeout=0.1)
                if audio_data and self.stream:
                    # Convert bytes to numpy array and write to stream
                    audio_array = np.frombuffer(audio_data, dtype=DTYPE)
                    self.stream.write(audio_array)
            except Exception:
                pass

    def play(self, audio_data: bytes):
        """Queue audio data for playback."""
        self.playback_queue.put(audio_data)

    def stop(self):
        """Stop audio playback."""
        self.is_running = False
        if hasattr(self, 'thread') and self.thread:
            self.thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop()
            self.stream.close()
        logger.info("Audio playback stopped")


class ScreenCapture:
    """Handles screen capture with thread-safe mss instance."""

    def __init__(self):
        self.sct = None  # Created lazily per-thread
        self._lock = threading.Lock()

    def _get_sct(self):
        """Get or create mss instance (thread-safe)."""
        # mss stores device contexts in thread-local storage,
        # so we need to create a new instance per thread if needed
        if self.sct is None:
            with self._lock:
                if self.sct is None:
                    self.sct = mss.mss()
        return self.sct

    def capture(self) -> bytes:
        """Capture the screen and return as JPEG bytes."""
        try:
            sct = self._get_sct()
            # Capture the primary monitor
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            # Resize if too large
            if img.width > SCREEN_MAX_SIZE[0] or img.height > SCREEN_MAX_SIZE[1]:
                img.thumbnail(SCREEN_MAX_SIZE, Image.Resampling.LANCZOS)

            # Convert to JPEG bytes
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None


class GeminiLiveClient:
    """WebSocket client for Gemini Live assistant."""

    def __init__(self, server_url: str, ptt_enabled: bool = False, ptt_key: str = DEFAULT_PTT_KEY):
        self.server_url = server_url
        self.websocket = None
        self.is_running = False
        self.gemini_ready = False  # Wait for Gemini to be ready before sending

        self.audio_queue = Queue()
        self.audio_capture = AudioCapture(self.audio_queue)
        self.audio_playback = AudioPlayback()
        self.screen_capture = ScreenCapture()
        
        # Privacy mode - when enabled, screen captures are not sent
        self.privacy_mode = False
        
        # Push-to-talk configuration
        self.ptt_enabled = ptt_enabled and PYNPUT_AVAILABLE
        self.ptt_key = PTT_KEY_MAP.get(ptt_key.lower(), keyboard.Key.ctrl_r) if PYNPUT_AVAILABLE else None
        self.ptt_key_name = ptt_key
        self.ptt_active = False  # True while PTT key is held
        self.ptt_releasing = False  # True when key released, waiting for flush
        self.ptt_screen_data = None  # Screen captured when PTT started
        self.ptt_screen_sent = False  # True after screen has been sent
        self.ptt_audio_chunks_sent = 0  # Count chunks before sending screen
        self.ptt_started = False  # True after first audio chunk sent
        self.keyboard_listener = None
        self._loop = None  # Store event loop for cross-thread scheduling

    def _on_ptt_press(self, key):
        """Handle PTT key press - start recording."""
        # Handle both Key objects and KeyCode objects (for character keys like grave)
        key_matches = (key == self.ptt_key) or (
            hasattr(key, 'char') and hasattr(self.ptt_key, 'char') and 
            key.char == self.ptt_key.char
        )
        if key_matches and not self.ptt_active:
            self.ptt_active = True
            self.ptt_started = False
            self.ptt_screen_sent = False  # Track if screen has been sent
            # Capture screen immediately when PTT starts (but don't send yet)
            # Skip screen capture if privacy mode is enabled
            if self.privacy_mode:
                self.ptt_screen_data = None
            else:
                self.ptt_screen_data = self.screen_capture.capture()
            self.ptt_audio_chunks_sent = 0  # Count audio chunks before sending screen
            # Clear any buffered audio
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    pass
            # Schedule UI update on the event loop
            if self._loop:
                self._loop.call_soon_threadsafe(
                    lambda: print("  🎤 [Recording... release key to send]")
                )
            logger.info(f"PTT activated (key: {self.ptt_key_name})")

    def _on_ptt_release(self, key):
        """Handle PTT key release - signal to flush remaining audio."""
        # Handle both Key objects and KeyCode objects (for character keys like grave)
        key_matches = (key == self.ptt_key) or (
            hasattr(key, 'char') and hasattr(self.ptt_key, 'char') and 
            key.char == self.ptt_key.char
        )
        if key_matches and self.ptt_active:
            self.ptt_active = False
            # Signal the audio loop to flush remaining audio before sending audio_end
            if self.ptt_started:
                self.ptt_releasing = True
            logger.info("PTT released - flushing audio buffer")

    def _start_keyboard_listener(self):
        """Start the global keyboard listener for PTT."""
        if not PYNPUT_AVAILABLE or not self.ptt_enabled:
            return
        
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_ptt_press,
            on_release=self._on_ptt_release
        )
        self.keyboard_listener.start()
        logger.info(f"PTT keyboard listener started (key: {self.ptt_key_name})")

    def _stop_keyboard_listener(self):
        """Stop the global keyboard listener."""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            logger.info("PTT keyboard listener stopped")

    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        self.is_running = True
        logger.info("Connected to server")

    async def run(self):
        """Main run loop."""
        try:
            # Store event loop for cross-thread scheduling (PTT callbacks)
            self._loop = asyncio.get_event_loop()
            
            await self.connect()

            # Start audio capture and playback
            if ENABLE_AUDIO_CAPTURE:
                self.audio_capture.start()
            else:
                logger.info("Audio capture DISABLED")
            self.audio_playback.start()  # Always enable playback to hear responses
            
            # Start PTT keyboard listener if enabled
            if self.ptt_enabled:
                self._start_keyboard_listener()

            # Run tasks concurrently
            await asyncio.gather(
                self._send_audio_loop(),
                self._send_screen_loop(),
                self._receive_loop(),
                self._handle_user_input()
            )

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {e}")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            await self.cleanup()

    async def _send_audio_loop(self):
        """Stream audio to server.
        
        Two modes:
        - PTT mode: Only send audio while push-to-talk key is held
        - VAD mode: Use voice activity detection to detect speech start/end
        """
        if not ENABLE_AUDIO_CAPTURE:
            while self.is_running:
                await asyncio.sleep(1.0)
            return
        
        if self.ptt_enabled:
            await self._send_audio_loop_ptt()
        else:
            await self._send_audio_loop_vad()

    async def _send_audio_loop_ptt(self):
        """PTT mode: Stream audio only while push-to-talk key is held.
        
        Strategy: Send audio first to establish voice context, then send screen
        after a few chunks. This helps the model understand it's receiving a
        voice query with visual context, rather than treating them as separate inputs.
        """
        CHUNKS_BEFORE_SCREEN = 3  # Send screen after this many audio chunks
        
        while self.is_running:
            try:
                # Wait for Gemini to be ready
                if not self.gemini_ready:
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    await asyncio.sleep(0.1)
                    continue
                
                # Process audio when PTT is active OR releasing (flushing)
                if self.ptt_active or self.ptt_releasing:
                    # Collect audio chunks
                    audio_buffer = b''
                    while not self.audio_queue.empty():
                        audio_buffer += self.audio_queue.get_nowait()
                    
                    if audio_buffer:
                        self.ptt_started = True
                        self.ptt_audio_chunks_sent += 1
                        
                        # Always send audio
                        audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                        await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                        # After a few audio chunks, send screen context
                        # This ensures model knows it's a voice turn before seeing the screen
                        if (not self.ptt_screen_sent and 
                            self.ptt_screen_data and 
                            self.ptt_audio_chunks_sent >= CHUNKS_BEFORE_SCREEN):
                            self.ptt_screen_sent = True
                            screen_b64 = base64.b64encode(self.ptt_screen_data).decode("utf-8")
                            await self.websocket.send(json.dumps({"screen_frame": screen_b64}))
                            logger.info(f"PTT: Screen sent after {self.ptt_audio_chunks_sent} audio chunks ({len(self.ptt_screen_data)/1024:.1f} KB)")
                    
                    # Handle PTT release: flush complete, send audio_end
                    if self.ptt_releasing:
                        # Wait a moment for any final audio to arrive
                        await asyncio.sleep(0.2)
                        
                        # Flush any remaining audio that arrived during the delay
                        audio_buffer = b''
                        while not self.audio_queue.empty():
                            audio_buffer += self.audio_queue.get_nowait()
                        if audio_buffer:
                            audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                            await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                        # If screen wasn't sent yet (very short PTT), send it now
                        if not self.ptt_screen_sent and self.ptt_screen_data:
                            screen_b64 = base64.b64encode(self.ptt_screen_data).decode("utf-8")
                            await self.websocket.send(json.dumps({"screen_frame": screen_b64}))
                            logger.info(f"PTT: Screen sent on release ({len(self.ptt_screen_data)/1024:.1f} KB)")
                        
                        # Now send audio_end
                        await self.websocket.send(json.dumps({"audio_end": True}))
                        print("  ⏸️  [Processing...]")
                        logger.info("PTT: Audio end sent after flush")
                        
                        # Reset state
                        self.ptt_releasing = False
                        self.ptt_started = False
                        self.ptt_screen_sent = False
                        self.ptt_audio_chunks_sent = 0
                        self.ptt_screen_data = None
                else:
                    # Discard audio when PTT is not active (and not releasing)
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                
                await asyncio.sleep(AUDIO_SEND_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error sending audio (PTT): {e}")
                await asyncio.sleep(0.1)

    async def _send_audio_loop_vad(self):
        """VAD mode: Use voice activity detection to detect speech start/end."""
        is_streaming = False
        silence_counter = 0
        SILENCE_INTERVALS_TO_STOP = int(SILENCE_DURATION_TO_END_TURN / AUDIO_SEND_INTERVAL)
            
        while self.is_running:
            try:
                # Wait for Gemini to be ready
                if not self.gemini_ready:
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    await asyncio.sleep(0.1)
                    continue
                
                # Collect audio chunks
                audio_buffer = b''
                while not self.audio_queue.empty():
                    audio_buffer += self.audio_queue.get_nowait()
                
                if audio_buffer:
                    audio_array = np.frombuffer(audio_buffer, dtype=DTYPE)
                    volume = np.abs(audio_array).mean()
                    
                    if volume > SPEECH_DETECTED_THRESHOLD:
                        if not is_streaming:
                            is_streaming = True
                            print("  🎤 [Listening...]")
                            logger.info(f"Speech detected (volume: {volume:.0f})")
                            
                            # Capture screen and send WITH first audio chunk
                            screen_data = self.screen_capture.capture()
                            audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                            
                            if screen_data:
                                screen_b64 = base64.b64encode(screen_data).decode("utf-8")
                                await self.websocket.send(json.dumps({
                                    "screen_frame": screen_b64,
                                    "audio_chunk": audio_b64
                                }))
                                logger.info(f"Screen + audio sent ({len(screen_data)/1024:.1f} KB)")
                            else:
                                await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        else:
                            # Continue streaming audio
                            audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                            await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                        silence_counter = 0
                        
                    elif is_streaming:
                        silence_counter += 1
                        # Still send during grace period
                        audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                        await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                        if silence_counter >= SILENCE_INTERVALS_TO_STOP:
                            is_streaming = False
                            silence_counter = 0
                            print("  ⏸️  [Processing...]")
                            logger.info("Speech ended")
                            await self.websocket.send(json.dumps({"audio_end": True}))
                
                await asyncio.sleep(AUDIO_SEND_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error sending audio (VAD): {e}")
                await asyncio.sleep(0.1)

    async def _send_screen_loop(self):
        """Periodically send screen captures - DISABLED.
        
        Periodic screen captures were causing confusion with the model.
        Screen context is now sent only when speech starts or via /describe.
        """
        # Periodic screen capture disabled - context sent with voice queries instead
        while self.is_running:
            await asyncio.sleep(10.0)
    
    async def send_screen_now(self):
        """Manually trigger a screen capture (for on-demand use)."""
        if not self.gemini_ready:
            print("  ⏳ Waiting for Gemini to connect...")
            return
        try:
            screen_data = self.screen_capture.capture()
            if screen_data:
                screen_b64 = base64.b64encode(screen_data).decode("utf-8")
                message = json.dumps({"screen_frame": screen_b64})
                await self.websocket.send(message)
                logger.info("Manual screen capture sent")
        except Exception as e:
            logger.error(f"Error sending manual screen: {e}")

    async def describe_screen(self):
        """Send screen capture + ask the model to describe it (combined in single message)."""
        if not self.gemini_ready:
            print("  ⏳ Waiting for Gemini to connect...")
            return
        if self.privacy_mode:
            print("  🔒 Privacy mode is enabled - screen sharing is paused")
            print("     Say 'disable privacy mode' to resume screen sharing")
            return
        try:
            screen_data = self.screen_capture.capture()
            if screen_data:
                screen_b64 = base64.b64encode(screen_data).decode("utf-8")
                prompt = "Please describe what you see on my screen right now."
                
                # Send BOTH screen and text in ONE message so they're processed together
                message = json.dumps({
                    "screen_frame": screen_b64,
                    "text": prompt
                })
                await self.websocket.send(message)
                logger.info(f"Screen captured and sent with prompt ({len(screen_data)/1024:.1f} KB)")
                print(f"You: [screen + '{prompt}']")
        except Exception as e:
            logger.error(f"Error in describe_screen: {e}")

    async def _receive_loop(self):
        """Receive and process server responses."""
        while self.is_running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)

                msg_type = data.get("type")

                if msg_type == "connection_established":
                    self.gemini_ready = True
                    print(f"\n✓ {data.get('message')}\n")
                
                elif msg_type == "status":
                    print(f"  → {data.get('message')}")

                elif msg_type == "audio_response":
                    # Decode and play audio
                    audio_b64 = data.get("audio_data")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        self.audio_playback.play(audio_bytes)
                        # Show audio indicator (don't spam logs)
                        if not hasattr(self, '_audio_playing'):
                            self._audio_playing = True
                            print("  🔊 [Playing audio response...]")

                elif msg_type == "text_response":
                    text = data.get("text", "")
                    if text:
                        print(f"\nAssistant: {text}")

                elif msg_type == "turn_complete":
                    if hasattr(self, '_audio_playing'):
                        del self._audio_playing
                    print("  ✓ [Response complete]")

                elif msg_type == "tool_call":
                    tool = data.get("tool", "unknown")
                    print(f"\n  🔧 Using tool: {tool}")

                elif msg_type == "privacy_mode":
                    self.privacy_mode = data.get("enabled", False)
                    if self.privacy_mode:
                        print("\n  🔒 Privacy mode ENABLED - screen sharing paused")
                    else:
                        print("\n  🔓 Privacy mode DISABLED - screen sharing resumed")
                    logger.info(f"Privacy mode set to: {self.privacy_mode}")

                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('message')}")

                else:
                    logger.debug(f"Unknown message type: {msg_type}")

            except websockets.exceptions.ConnectionClosed:
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error receiving: {e}")

    async def _handle_user_input(self):
        """Handle text input from user (for testing without mic)."""
        print("\n" + "=" * 50)
        print("Gemini Live Assistant Client")
        print("=" * 50)
        print("Commands:")
        print("  [text]    - Send text message to assistant")
        print("  /describe - Capture screen + ask model to describe it")
        print("  /privacy  - Toggle privacy mode (or say it verbally)")
        print("  /quit     - Exit the client")
        print("")
        print("Features:")
        if self.ptt_enabled:
            print(f"  🎤 Voice: PTT mode - hold [{self.ptt_key_name.upper()}] to talk")
            print(f"  🖥️  Screen: captured when PTT key pressed")
        else:
            print(f"  🎤 Voice: {'VAD mode - auto-detects speech' if ENABLE_AUDIO_CAPTURE else 'DISABLED'}")
            print(f"  🖥️  Screen: captured when speech starts")
        print(f"  🔒 Privacy: Say 'enable/disable privacy mode' to toggle")
        print("")
        if self.ptt_enabled:
            print(f"Tip: Hold [{self.ptt_key_name.upper()}] while speaking, release when done")
        else:
            print("Tip: Use --ptt flag for push-to-talk mode")
        print("=" * 50 + "\n")

        loop = asyncio.get_event_loop()

        while self.is_running:
            try:
                # Use thread executor for blocking input
                user_input = await loop.run_in_executor(
                    None, 
                    lambda: input() if sys.stdin.isatty() else None
                )

                if user_input is None:
                    await asyncio.sleep(1)
                    continue

                user_input = user_input.strip()
                
                if user_input.lower() in ("quit", "/quit", "exit"):
                    self.is_running = False
                    break

                if user_input.lower() == "/describe":
                    await self.describe_screen()
                    continue

                if user_input.lower() == "/privacy":
                    self.privacy_mode = not self.privacy_mode
                    if self.privacy_mode:
                        print("  🔒 Privacy mode ENABLED - screen sharing paused")
                    else:
                        print("  🔓 Privacy mode DISABLED - screen sharing resumed")
                    continue

                if user_input:
                    if not self.gemini_ready:
                        print("  ⏳ Waiting for Gemini to connect...")
                        continue
                    message = json.dumps({"text": user_input})
                    await self.websocket.send(message)
                    print(f"You: {user_input}")

            except EOFError:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Input error: {e}")
                await asyncio.sleep(0.1)

    async def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        self._stop_keyboard_listener()
        self.audio_capture.stop()
        self.audio_playback.stop()
        if self.websocket:
            await self.websocket.close()
        logger.info("Client cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description="Mock client for Gemini Live Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Push-to-talk keys:
  grave             - Backtick/tilde key (default, classic PTT key)
  backslash         - Backslash key
  ctrl_r, ctrl_l    - Right/Left Control
  alt_r, alt_l      - Right/Left Alt/Option
  shift_r, shift_l  - Right/Left Shift
  f1-f12            - Function keys (note: some have macOS bindings)
  space, tab        - Space or Tab key
  caps_lock         - Caps Lock (may need OS config)

Examples:
  python mock_client.py --ptt              # PTT with Right Ctrl
  python mock_client.py --ptt --ptt-key f5 # PTT with F5
  python mock_client.py                    # Voice activity detection mode

Note: On macOS, pynput requires Accessibility permissions.
      Go to System Settings > Privacy & Security > Accessibility
      and add your terminal app.
"""
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws/live-session/",
        help="WebSocket server URL"
    )
    parser.add_argument(
        "--ptt",
        action="store_true",
        help="Enable push-to-talk mode (hold key to record)"
    )
    parser.add_argument(
        "--ptt-key",
        default=DEFAULT_PTT_KEY,
        help=f"Key for push-to-talk (default: {DEFAULT_PTT_KEY})"
    )
    args = parser.parse_args()
    
    # Validate PTT key
    if args.ptt and args.ptt_key.lower() not in PTT_KEY_MAP:
        print(f"Error: Unknown PTT key '{args.ptt_key}'")
        print(f"Available keys: {', '.join(sorted(PTT_KEY_MAP.keys()))}")
        sys.exit(1)
    
    if args.ptt and not PYNPUT_AVAILABLE:
        print("Error: Push-to-talk requires pynput. Install with: pip install pynput")
        sys.exit(1)

    client = GeminiLiveClient(
        server_url=args.server,
        ptt_enabled=args.ptt,
        ptt_key=args.ptt_key
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
