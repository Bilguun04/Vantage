"""
Gemini Live WebSocket Consumer

Handles bidirectional streaming between client and Google Gemini Multimodal Live API.
Receives audio (PCM) and video (screenshots) from client, forwards to Gemini,
and streams audio responses back.
"""

import asyncio
import base64
import json
import logging
import os
import uuid

from channels.generic.websocket import AsyncWebsocketConsumer
from google import genai
from app.models import GeminiConversation, ConversationMessage, ConversationImage
from datetime import datetime

logger = logging.getLogger(__name__)

# Model configuration - using native audio dialog model
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025"
SYSTEM_INSTRUCTION = """You are a helpful background assistant. You can see the user's screen. 
Use this visual context to answer questions. Use Google Search if you need up-to-date information.

You have a privacy mode feature. When the user asks to enable or disable privacy mode (or similar 
requests like "stop sharing my screen", "hide my screen", "resume screen sharing"), use the 
toggle_privacy_mode tool. After toggling, briefly confirm the change verbally."""

# Privacy mode tool definition
PRIVACY_MODE_TOOL = {
    "function_declarations": [{
        "name": "toggle_privacy_mode",
        "description": "Toggle privacy mode on or off. When privacy mode is ON, screen sharing is paused and the assistant cannot see the user's screen. When OFF, screen sharing resumes. Use this when the user requests privacy, wants to hide their screen, or wants to resume screen sharing.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean", 
                    "description": "True to enable privacy mode (stop screen sharing), False to disable it (resume screen sharing)"
                }
            },
            "required": ["enabled"]
        }
    }]
}


class GeminiLiveConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time Gemini Live sessions.
    
    Handles:
    - Bidirectional audio/video streaming with Gemini
    - Google Search tool integration (auto mode)
    - Base64 encoded data transfer with client
    - Saving conversation history to MongoDB
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.session = None
        self.session_context = None  # Keep reference to context manager
        self.response_listener_task = None
        self.is_connected = False
        self.gemini_ready = False  # True only when Gemini session is active
        
        # MongoDB conversation tracking
        self.conversation = None
        self.user_id = None
        self.session_id = str(uuid.uuid4())  # Unique identifier for this session
        self.current_turn_images = []  # Track images in current turn
        
        # Turn accumulation - collect content until turn completes
        self.current_user_turn = {
            "text": [],       # List of text fragments
            "audio_chunks": [],  # List of audio byte arrays
            "images": []      # Images for this turn
        }
        self.current_assistant_turn = {
            "text": [],       # List of text fragments  
            "thoughts": [],   # List of thought fragments
            "audio_chunks": []  # List of audio byte arrays
        }

    async def connect(self):
        """
        Initialize connection and establish Gemini Live session.
        Includes retry logic for quota errors.
        """
        await self.accept()
        self.is_connected = True
        logger.info("WebSocket connection accepted")

        # Extract user info from scope
        try:
            self.user_id = self.scope.get('user', {}).get('id') or 'anonymous'
        except:
            self.user_id = 'anonymous'

        # Initialize MongoDB conversation document
        try:
            self.conversation = GeminiConversation(
                user_id=self.user_id,
                session_id=self.session_id,
                title=f"Gemini Chat - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                system_instruction=SYSTEM_INSTRUCTION,
                total_messages='0',
                total_images='0',
                total_audio_chunks='0'
            )
            self.conversation.save()
            logger.info(f"Created conversation document: {self.conversation.id}")
        except Exception as e:
            logger.error(f"Error creating conversation document: {e}")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            await self.send_error("GOOGLE_API_KEY environment variable not set")
            await self.close()
            return

        self.client = genai.Client(api_key=api_key)

        # Retry logic for quota errors
        max_retries = 3
        retry_delays = [5, 15, 30]  # seconds between retries
        
        for attempt in range(max_retries):
            try:
                await self.send(text_data=json.dumps({
                    "type": "status",
                    "message": f"Connecting to Gemini... (attempt {attempt + 1}/{max_retries})"
                }))

                # Configure the Live session with Google Search and privacy mode tools
                # Note: native-audio model only supports AUDIO response modality
                config = {
                    "response_modalities": ["AUDIO"],
                    "tools": [
                        {"google_search": {}},
                        PRIVACY_MODE_TOOL
                    ],
                    "system_instruction": SYSTEM_INSTRUCTION,
                }

                # Establish bidirectional session with Gemini
                # Keep reference to context manager to prevent early cleanup
                self.session_context = self.client.aio.live.connect(
                    model=MODEL_ID,
                    config=config
                )
                self.session = await self.session_context.__aenter__()

                logger.info(f"Gemini Live session established with model: {MODEL_ID}")
                logger.info(f"Session object: {type(self.session)}")
                self.gemini_ready = True

                # Start background task to listen for Gemini responses
                self.response_listener_task = asyncio.create_task(
                    self._listen_for_responses()
                )

                # Notify client of successful connection
                await self.send(text_data=json.dumps({
                    "type": "connection_established",
                    "message": "Connected to Gemini Live session. You can now speak or type."
                }))
                return  # Success, exit retry loop

            except Exception as e:
                error_str = str(e).lower()
                is_quota_error = "quota" in error_str or "rate" in error_str or "429" in error_str
                
                if is_quota_error and attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logger.warning(f"Quota error, retrying in {delay}s... (attempt {attempt + 1})")
                    await self.send(text_data=json.dumps({
                        "type": "status",
                        "message": f"Rate limited. Retrying in {delay} seconds..."
                    }))
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Failed to initialize Gemini session: {e}")
                    await self.send_error(f"Failed to connect to Gemini: {str(e)}")
                    await self.close()
                    return

    async def disconnect(self, close_code):
        """
        Clean up resources on disconnect and save final session state.
        """
        self.is_connected = False
        self.gemini_ready = False
        logger.info(f"WebSocket disconnected with code: {close_code}")
    
        # Cancel the response listener task
        if self.response_listener_task:
            self.response_listener_task.cancel()
            try:
                await self.response_listener_task
            except asyncio.CancelledError:
                pass
    
        # Close the Gemini session context
        if self.session_context:
            try:
                await self.session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing Gemini session: {e}")
        
        # Save final session state to MongoDB
        if self.conversation:
            try:
                self.conversation.updated_at = datetime.utcnow()
                self.conversation.last_activity = datetime.utcnow()
                
                # Add final session metadata
                final_message = ConversationMessage(
                    role="system",
                    message_type="text",
                    content=f"Session ended with close code: {close_code}"
                )
                self.conversation.messages.append(final_message)
                
                # Save to database
                self.conversation.save()
                logger.info(f"Conversation saved on disconnect. Total messages: {len(self.conversation.messages)}")
            except Exception as e:
                logger.error(f"Error saving conversation on disconnect: {e}")

    async def receive(self, text_data=None, bytes_data=None):
        """
        Handle incoming messages from the client.
        
        Expected JSON format:
        {
            "audio_chunk": "<base64 encoded PCM audio>",  // optional
            "screen_frame": "<base64 encoded image>"      // optional
        }
        """
        if not self.gemini_ready:
            # Silently drop data until Gemini session is ready
            logger.debug("Dropping data - Gemini not ready yet")
            return

        try:
            if text_data:
                data = json.loads(text_data)
                # Log what keys we received (not the full data - too verbose)
                keys = list(data.keys())
                if "text" in data:
                    logger.info(f"Received text message: {data['text'][:50]}...")
                elif "screen_frame" in data:
                    logger.info("Received screen frame")
                await self._process_client_data(data)
            elif bytes_data:
                # Handle raw binary data as audio
                await self._send_audio_to_gemini(bytes_data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error(f"Error processing message: {str(e)}")

    async def _process_client_data(self, data: dict):
        """
        Process and forward client data to Gemini.
        """
        has_audio_chunk = "audio_chunk" in data and data["audio_chunk"]
        has_screen = "screen_frame" in data and data["screen_frame"]
        has_text = "text" in data and data["text"]
        
        # Handle screen + audio (voice with visual context)
        # Strategy: Send screen then audio via realtime input
        if has_screen and has_audio_chunk:
            image_bytes = base64.b64decode(data["screen_frame"])
            audio_bytes = base64.b64decode(data["audio_chunk"])
            await self._save_image_to_db(image_bytes, mime_type="image/jpeg")
            await self._send_screen_then_audio(image_bytes, audio_bytes)
        
        # Handle audio-only chunk (PTT streaming)
        elif has_audio_chunk:
            audio_bytes = base64.b64decode(data["audio_chunk"])
            await self._send_audio_to_gemini(audio_bytes)
        
        # Handle screen + text (like /describe command)
        elif has_screen and has_text:
            image_bytes = base64.b64decode(data["screen_frame"])
            await self._save_image_to_db(image_bytes, mime_type="image/jpeg")
            await self._send_image_to_gemini(image_bytes, with_prompt=data["text"])
        
        # Handle screen only (mid-stream context, don't trigger response)
        elif has_screen:
            image_bytes = base64.b64decode(data["screen_frame"])
            await self._save_image_to_db(image_bytes, mime_type="image/jpeg")
            await self._send_screen_context(image_bytes)

        # Handle text only
        elif has_text:
            await self._send_text_to_gemini(data["text"])
        
        # Handle audio end signal
        if data.get("audio_end"):
            logger.info("Client signaled audio end")

    async def _send_audio_to_gemini(self, audio_bytes: bytes):
        """
        Send audio data to Gemini Live session using realtime input.
        """
        try:
            await self.session.send_realtime_input(
                audio={"data": audio_bytes, "mime_type": "audio/pcm"}
            )
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")

    async def _send_screen_context(self, image_bytes: bytes):
        """
        Send screen as realtime media input (for mid-stream visual context).
        
        This is used when screen is sent during an audio stream - it doesn't
        trigger a response, just provides visual context for the ongoing audio.
        """
        try:
            mime_type = "image/jpeg"
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            
            logger.info(f"Sending screen context via realtime media ({len(image_bytes)/1024:.1f} KB)")
            await self.session.send_realtime_input(
                media={"data": image_bytes, "mime_type": mime_type}
            )
        except Exception as e:
            logger.error(f"Error sending screen context: {e}")
    
    async def _send_screen_then_audio(self, image_bytes: bytes, audio_bytes: bytes):
        """
        Send screen context AND audio via realtime input (sequentially).
        
        Both must be sent via send_realtime_input so the model correlates them.
        Using send_client_content for image + send_realtime_input for audio
        causes the model to not associate them properly (leading to hallucination).
        
        Note: send_realtime_input only accepts one argument at a time, so we send
        media first, then audio, in quick succession.
        """
        try:
            # Detect image type
            mime_type = "image/jpeg"
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"
            
            logger.info(f"Sending screen ({len(image_bytes)/1024:.1f} KB) + audio via realtime input")
            
            # Send screen first via realtime input (as media/video frame)
            await self.session.send_realtime_input(
                media={"data": image_bytes, "mime_type": mime_type}
            )
            
            # Then send audio via realtime input (same API keeps them correlated)
            await self.session.send_realtime_input(
                audio={"data": audio_bytes, "mime_type": "audio/pcm"}
            )
            logger.info("Screen + audio sent via sequential realtime input calls")
            
        except Exception as e:
            logger.error(f"Error sending screen+audio: {e}")
            # Fallback: try audio only
            try:
                await self.session.send_realtime_input(
                    audio={"data": audio_bytes, "mime_type": "audio/pcm"}
                )
            except Exception:
                pass

    async def _send_image_to_gemini(self, image_bytes: bytes, with_prompt: str = None):
        """
        Send image/screenshot data to Gemini Live session.
        Uses send_client_content with inline_data for discrete images.
        
        Args:
            image_bytes: The image data
            with_prompt: Optional text prompt to send with the image
        Send image/screenshot data to Gemini Live session and save to MongoDB.
        """
        try:
            # Detect image type (default to JPEG for screenshots)
            mime_type = "image/jpeg"
            if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                mime_type = "image/png"

            logger.info(f"Sending screen frame to Gemini ({len(image_bytes)/1024:.1f} KB, mime={mime_type})")
            
            # Base64 encode the image for inline_data
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Build parts list - image first, then optional text
            parts = [
                {"inline_data": {"mime_type": mime_type, "data": image_b64}}
            ]
            
            if with_prompt:
                parts.append({"text": with_prompt})
                logger.info(f"Sending image with prompt: '{with_prompt[:50]}...'")
            
            # Send as client content (not realtime input)
            await self.session.send_client_content(
                turns=[{"role": "user", "parts": parts}],
                turn_complete=True if with_prompt else False  # Only complete turn if there's a prompt
            )
            logger.info("Screen frame sent successfully via send_client_content")
        except Exception as e:
            logger.error(f"Error sending image to Gemini: {e}", exc_info=True)

    async def _send_text_to_gemini(self, text: str):
        """
        Send text message to Gemini Live session.
        Accumulates text for the current user turn - saved when assistant responds.
        """
        try:
            logger.info(f"Sending text to Gemini: '{text[:50]}...'")
            
            # Accumulate user text for this turn (will be saved when turn completes)
            self.current_user_turn["text"].append(text)
            # Move any buffered images to the current turn
            self.current_user_turn["images"].extend(self.current_turn_images)
            self.current_turn_images = []
            
            await self.session.send_client_content(
                turns=[{"role": "user", "parts": [{"text": text}]}],
                turn_complete=True
            )
            
            # Save the user turn now (since turn_complete=True means we're done speaking)
            await self._save_user_turn()
            
            logger.info("Text sent to Gemini successfully")
        except Exception as e:
            logger.error(f"Error sending text to Gemini: {e}")
    
    async def _save_user_turn(self):
        """
        Save accumulated user turn content as a single message.
        Called when user turn is complete (text sent or audio ended).
        """
        try:
            if not self.conversation:
                return
                
            # Combine all text fragments
            full_text = " ".join(self.current_user_turn["text"]) if self.current_user_turn["text"] else None
            
            # Only save if there's content
            if full_text or self.current_user_turn["images"]:
                message = ConversationMessage(
                    role="user",
                    message_type="text",
                    content=full_text,
                    images=self.current_user_turn["images"]
                )
                self.conversation.messages.append(message)
                self.conversation.total_messages = str(int(self.conversation.total_messages or 0) + 1)
                if self.current_user_turn["images"]:
                    self.conversation.total_images = str(int(self.conversation.total_images or 0) + len(self.current_user_turn["images"]))
                self.conversation.last_activity = datetime.utcnow()
                self.conversation.save()
                logger.info(f"✓ Saved user turn: '{(full_text or '')[:50]}...' with {len(self.current_user_turn['images'])} images")
            
            # Reset user turn accumulator
            self.current_user_turn = {"text": [], "audio_chunks": [], "images": []}
            
        except Exception as e:
            logger.error(f"Error saving user turn: {e}")
    
    async def _save_assistant_turn(self):
        """
        Save accumulated assistant turn content as messages.
        Called when assistant turn is complete (turn_complete signal received).
        """
        try:
            if not self.conversation:
                return
            
            saved_count = 0
            
            # Save thoughts if any (as a separate message for transparency)
            if self.current_assistant_turn["thoughts"]:
                thought_text = " ".join(self.current_assistant_turn["thoughts"])
                message = ConversationMessage(
                    role="assistant",
                    message_type="thought",
                    content=thought_text
                )
                self.conversation.messages.append(message)
                saved_count += 1
                logger.info(f"✓ Saved assistant thought: '{thought_text[:50]}...'")
            
            # Save text response if any
            if self.current_assistant_turn["text"]:
                full_text = " ".join(self.current_assistant_turn["text"])
                message = ConversationMessage(
                    role="assistant",
                    message_type="text",
                    content=full_text
                )
                self.conversation.messages.append(message)
                saved_count += 1
                logger.info(f"✓ Saved assistant text: '{full_text[:50]}...'")
            
            # Save audio response if any (combine chunks)
            if self.current_assistant_turn["audio_chunks"]:
                combined_audio = b"".join(self.current_assistant_turn["audio_chunks"])
                audio_size_kb = f"{len(combined_audio) / 1024:.2f}"
                # Estimate duration: 24kHz, 16-bit, mono
                estimated_duration = len(combined_audio) / (24000 * 2 * 1)
                
                message = ConversationMessage(
                    role="assistant",
                    message_type="audio",
                    content=f"[Audio response - {estimated_duration:.1f}s]",
                    audio_data=combined_audio,
                    audio_duration_seconds=f"{estimated_duration:.2f}",
                    audio_size_kb=audio_size_kb
                )
                self.conversation.messages.append(message)
                saved_count += 1
                self.conversation.total_audio_chunks = str(int(self.conversation.total_audio_chunks or 0) + 1)
                logger.info(f"✓ Saved assistant audio: {estimated_duration:.1f}s, {audio_size_kb} KB")
            
            if saved_count > 0:
                self.conversation.total_messages = str(int(self.conversation.total_messages or 0) + saved_count)
                self.conversation.last_activity = datetime.utcnow()
                self.conversation.save()
            
            # Reset assistant turn accumulator
            self.current_assistant_turn = {"text": [], "thoughts": [], "audio_chunks": []}
            
        except Exception as e:
            logger.error(f"Error saving assistant turn: {e}")
    
    async def _accumulate_assistant_content(self, content_type: str, content):
        """
        Accumulate assistant response content during a turn.
        
        Args:
            content_type: 'text', 'thought', or 'audio'
            content: The content to accumulate (str for text/thought, bytes for audio)
        """
        if content_type == "text":
            self.current_assistant_turn["text"].append(content)
        elif content_type == "thought":
            self.current_assistant_turn["thoughts"].append(content)
        elif content_type == "audio":
            self.current_assistant_turn["audio_chunks"].append(content)
    
    async def _save_image_to_db(self, image_bytes: bytes, mime_type: str):
        """
        Buffer an image for the current turn.
        Images are saved when the associated text/audio message is saved.
        """
        try:
            size_kb = f"{len(image_bytes) / 1024:.2f}"
            
            image = ConversationImage(
                data=image_bytes,
                mime_type=mime_type,
                size_kb=size_kb
            )
            
            # Add image to current turn's image list
            self.current_turn_images.append(image)
            
            logger.info(f"✓ Buffered image ({size_kb} KB) for current turn")
            
        except Exception as e:
            logger.error(f"Error processing image for MongoDB: {e}")

    async def _listen_for_responses(self):
        """
        Background task to listen for Gemini responses and stream back to client.
        Loops continuously to handle multiple conversation turns.
        """
        try:
            logger.info("Starting response listener...")
            turn_count = 0
            consecutive_errors = 0
            MAX_CONSECUTIVE_ERRORS = 3  # Stop after 3 consecutive errors
            
            while self.is_connected and self.gemini_ready:
                try:
                    logger.debug(f"Waiting for turn {turn_count + 1}...")
                    # Each receive() call handles responses until turn completes
                    async for response in self.session.receive():
                        if not self.is_connected:
                            logger.info("Disconnected, stopping listener")
                            return
                        await self._handle_gemini_response(response)
                        consecutive_errors = 0  # Reset on successful response
                    # Turn completed, loop to wait for next turn
                    turn_count += 1
                    consecutive_errors = 0  # Reset on successful turn
                    logger.info(f"Turn {turn_count} completed, ready for next input")
                except Exception as turn_error:
                    error_str = str(turn_error)
                    if "1000" in error_str:
                        logger.info("Session closed normally")
                        break
                    elif "1001" in error_str or "going away" in error_str.lower():
                        logger.info("Session ended by server")
                        break
                    elif "precondition" in error_str.lower() or "1007" in error_str:
                        # Session is broken - stop immediately
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.error(f"Session broken (precondition failed {consecutive_errors}x), stopping")
                            self.gemini_ready = False
                            await self.send_error("Gemini session failed. Please reconnect.")
                            break
                        await asyncio.sleep(1.0)  # Longer delay on precondition errors
                    else:
                        consecutive_errors += 1
                        logger.error(f"Turn error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {turn_error}")
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            logger.error("Too many consecutive errors, stopping listener")
                            self.gemini_ready = False
                            await self.send_error("Connection to Gemini lost. Please reconnect.")
                            break
                        await asyncio.sleep(0.5)
                    
        except asyncio.CancelledError:
            logger.info("Response listener cancelled")
        except Exception as e:
            logger.error(f"Error in response listener: {e}")
            if self.is_connected:
                await self.send_error(f"Error receiving from Gemini: {str(e)}")

    async def _handle_gemini_response(self, response):
        """
        Process and forward Gemini response to client.
        Accumulates content during the turn and saves when turn completes.
        """
        try:
            # Handle server content (audio/text responses)
            if response.server_content:
                content = response.server_content

                # Process model parts (audio/text/thought) - accumulate, don't save yet
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        # Handle audio response
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if part.inline_data.mime_type and part.inline_data.mime_type.startswith("audio/"):
                                audio_size = len(part.inline_data.data) if part.inline_data.data else 0
                                if audio_size > 0:
                                    # Accumulate audio (will save when turn completes)
                                    await self._accumulate_assistant_content("audio", part.inline_data.data)
                                    
                                    # Stream to client immediately
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                    await self.send(text_data=json.dumps({
                                        "type": "audio_response",
                                        "audio_data": audio_b64,
                                        "mime_type": part.inline_data.mime_type
                                    }))

                        # Handle text response
                        if hasattr(part, 'text') and part.text:
                            is_thought = getattr(part, 'thought', False) is True
                            if is_thought:
                                logger.info(f"Model thought: {part.text[:100]}...")
                                await self._accumulate_assistant_content("thought", part.text)
                            else:
                                logger.info(f"Model text: {part.text[:100]}...")
                                await self._accumulate_assistant_content("text", part.text)
                            
                            # Stream to client immediately
                            await self.send(text_data=json.dumps({
                                "type": "text_response",
                                "text": part.text
                            }))
                
                # Check if model turn is complete - NOW save the accumulated content
                if content.turn_complete:
                    logger.info("Model turn complete - saving conversation")
                    await self._save_assistant_turn()
                    await self.send(text_data=json.dumps({
                        "type": "turn_complete"
                    }))

            # Handle tool calls (Google Search, Privacy Mode, etc.)
            if hasattr(response, 'tool_call') and response.tool_call:
                logger.info(f"Tool call: {response.tool_call}")
                
                # Process each function call
                function_responses = []
                for fc in response.tool_call.function_calls:
                    if fc.name == "toggle_privacy_mode":
                        # Extract the enabled parameter
                        enabled = fc.args.get("enabled", True)
                        logger.info(f"Privacy mode toggle requested: {enabled}")
                        
                        # Send privacy mode command to client
                        await self.send(text_data=json.dumps({
                            "type": "privacy_mode",
                            "enabled": enabled
                        }))
                        
                        # Build tool response for Gemini
                        status = "enabled" if enabled else "disabled"
                        function_responses.append({
                            "name": fc.name,
                            "id": fc.id,
                            "response": {"status": "success", "privacy_mode": status}
                        })
                        logger.info(f"Privacy mode {status}")
                    else:
                        # Other tool calls (like google_search) - handled automatically by Gemini
                        await self.send(text_data=json.dumps({
                            "type": "tool_call",
                            "tool": fc.name,
                            "status": "executing"
                        }))
                
                # Send tool responses back to Gemini
                if function_responses:
                    try:
                        await self.session.send_tool_response(
                            function_responses=function_responses
                        )
                        logger.info(f"Tool response sent to Gemini: {function_responses}")
                    except Exception as e:
                        logger.error(f"Error sending tool response: {e}")

            # Handle tool call cancellation
            if hasattr(response, 'tool_call_cancellation') and response.tool_call_cancellation:
                await self.send(text_data=json.dumps({
                    "type": "tool_call_cancelled"
                }))

        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}", exc_info=True)

    async def send_error(self, message: str):
        """
        Send error message to client.
        """
        try:
            await self.send(text_data=json.dumps({
                "type": "error",
                "message": message
            }))
        except Exception:
            pass
