#!/usr/bin/env python3
"""
Mock Client for Gemini Live Voice Assistant

A standalone test client that captures screen and microphone input,
sends them to the Django WebSocket server, and plays back audio responses.

Usage:
    python mock_client.py [--server ws://localhost:8000/ws/live-session/]

Requirements:
    pip install websockets sounddevice numpy mss pillow
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

# Voice activity detection settings
SILENCE_THRESHOLD = 100  # Volume level below this = silence
SILENCE_DURATION_TO_END_TURN = 1.5  # Seconds of silence before signaling end of speech
SPEECH_DETECTED_THRESHOLD = 200  # Volume level to consider as speech starting


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
    """Handles screen capture."""

    def __init__(self):
        self.sct = mss.mss()

    def capture(self) -> bytes:
        """Capture the screen and return as JPEG bytes."""
        try:
            # Capture the primary monitor
            monitor = self.sct.monitors[1]  # Primary monitor
            screenshot = self.sct.grab(monitor)

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

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.websocket = None
        self.is_running = False
        self.gemini_ready = False  # Wait for Gemini to be ready before sending

        self.audio_queue = Queue()
        self.audio_capture = AudioCapture(self.audio_queue)
        self.audio_playback = AudioPlayback()
        self.screen_capture = ScreenCapture()

    async def connect(self):
        """Connect to the WebSocket server."""
        logger.info(f"Connecting to {self.server_url}...")
        self.websocket = await websockets.connect(self.server_url)
        self.is_running = True
        logger.info("Connected to server")

    async def run(self):
        """Main run loop."""
        try:
            await self.connect()

            # Start audio capture and playback
            if ENABLE_AUDIO_CAPTURE:
                self.audio_capture.start()
            else:
                logger.info("Audio capture DISABLED")
            self.audio_playback.start()  # Always enable playback to hear responses

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
        """Send audio chunks to server with speech detection.
        
        Strategy: Start streaming when speech detected, continue for a bit after
        silence is detected to capture trailing audio, then stop and signal end.
        """
        # Skip entirely if audio capture is disabled
        if not ENABLE_AUDIO_CAPTURE:
            while self.is_running:
                await asyncio.sleep(1.0)
            return
        
        # State tracking
        is_streaming = False  # Are we currently sending to Gemini?
        silence_counter = 0   # How many consecutive silent intervals?
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
                        # Speech detected - start or continue streaming
                        if not is_streaming:
                            is_streaming = True
                            print("  🎤 [Listening...]")
                            logger.info(f"Speech detected (volume: {volume:.0f}), starting stream")
                        
                        silence_counter = 0  # Reset silence counter
                        
                        # Send audio
                        audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                        await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                    elif is_streaming:
                        # Currently streaming but volume is low
                        silence_counter += 1
                        
                        # Still send audio during grace period (capture trailing speech)
                        audio_b64 = base64.b64encode(audio_buffer).decode("utf-8")
                        await self.websocket.send(json.dumps({"audio_chunk": audio_b64}))
                        
                        if silence_counter >= SILENCE_INTERVALS_TO_STOP:
                            # Enough silence - stop streaming and signal end
                            is_streaming = False
                            silence_counter = 0
                            print("  ⏸️  [Processing...]")
                            logger.info("Speech ended, signaling audio end")
                            await self.websocket.send(json.dumps({"audio_end": True}))
                    
                    # If not streaming and no speech, just drop the audio (don't flood Gemini)
                
                await asyncio.sleep(AUDIO_SEND_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                await asyncio.sleep(0.1)

    async def _send_screen_loop(self):
        """Periodically send screen captures (demo mode)."""
        if not ENABLE_SCREEN_CAPTURE:
            logger.info("Screen capture disabled")
            return
            
        while self.is_running:
            try:
                # Wait for Gemini to be ready
                if not self.gemini_ready:
                    await asyncio.sleep(0.5)
                    continue
                
                screen_data = self.screen_capture.capture()
                if screen_data:
                    screen_b64 = base64.b64encode(screen_data).decode("utf-8")
                    size_kb = len(screen_data) / 1024
                    
                    message = json.dumps({
                        "screen_frame": screen_b64
                    })
                    await self.websocket.send(message)
                    logger.info(f"Screen captured and sent ({size_kb:.1f} KB)")

                await asyncio.sleep(SCREEN_CAPTURE_INTERVAL)
            except Exception as e:
                logger.error(f"Error sending screen: {e}")
                await asyncio.sleep(SCREEN_CAPTURE_INTERVAL)
    
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
                    print(f"\nUsing tool: {tool}")

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
        print("Gemini Live Assistant Client (Demo Mode)")
        print("=" * 50)
        print("Commands:")
        print("  [text]   - Send text message to assistant")
        print("  /screen  - Capture and send screen now (context only)")
        print("  /describe - Capture screen + ask model to describe it")
        print("  /quit    - Exit the client")
        print("")
        print("Demo Settings:")
        print(f"  Screen capture: {'every ' + str(SCREEN_CAPTURE_INTERVAL) + 's' if ENABLE_SCREEN_CAPTURE else 'DISABLED'}")
        print(f"  Audio input: {'batched every ' + str(int(AUDIO_SEND_INTERVAL*1000)) + 'ms' if ENABLE_AUDIO_CAPTURE else 'DISABLED'}")
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
                
                if user_input.lower() == "/screen":
                    await self.send_screen_now()
                    continue

                if user_input.lower() == "/describe":
                    await self.describe_screen()
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
        self.audio_capture.stop()
        self.audio_playback.stop()
        if self.websocket:
            await self.websocket.close()
        logger.info("Client cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description="Mock client for Gemini Live Voice Assistant"
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000/ws/live-session/",
        help="WebSocket server URL"
    )
    args = parser.parse_args()

    client = GeminiLiveClient(args.server)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
