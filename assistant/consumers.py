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
Use this visual context to answer questions. Use Google Search if you need up-to-date information."""


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
                system_instruction=SYSTEM_INSTRUCTION
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

                # Configure the Live session with Google Search tool
                # Note: native-audio model only supports AUDIO response modality
                config = {
                    "response_modalities": ["AUDIO"],
                    "tools": [{"google_search": {}}],
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
        Clean up resources on disconnect.
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
        # Strategy: Send screen first with prompt, then stream audio
        if has_screen and has_audio_chunk:
            image_bytes = base64.b64decode(data["screen_frame"])
            # Save image to current turn
            await self._save_image_to_db(image_bytes, mime_type="image/jpeg")
            await self._send_image_to_gemini(image_bytes, with_prompt=data["text"])
        
        # Handle screen only
        elif has_screen:
            image_bytes = base64.b64decode(data["screen_frame"])
            await self._save_image_to_db(image_bytes, mime_type="image/jpeg")
            await self._send_image_to_gemini(
                image_bytes, 
                with_prompt="[Current screen context - acknowledge briefly]"
            )

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
        Send text message to Gemini Live session and save to MongoDB.
        """
        try:
            logger.info(f"Sending text to Gemini: '{text[:50]}...'")
            
            # Save user message to MongoDB
            await self._save_message_to_db("user", "text", text)
            
            await self.session.send_client_content(
                turns=[{"role": "user", "parts": [{"text": text}]}],
                turn_complete=True
            )
            logger.info("Text sent to Gemini successfully")
        except Exception as e:
            logger.error(f"Error sending text to Gemini: {e}")

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
        """
        try:
            # Handle server content (audio/text responses)
            if response.server_content:
                content = response.server_content
                
                # Check if model turn is complete
                if content.turn_complete:
                    logger.info("Model turn complete")
                    await self.send(text_data=json.dumps({
                        "type": "turn_complete"
                    }))

                # Process model parts (audio/text/thought)
                if content.model_turn and content.model_turn.parts:
                    for part in content.model_turn.parts:
                        # Handle audio response
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if part.inline_data.mime_type and part.inline_data.mime_type.startswith("audio/"):
                                audio_size = len(part.inline_data.data) if part.inline_data.data else 0
                                if audio_size > 0:
                                    # Save audio response to MongoDB
                                    await self._save_message_to_db("assistant", "audio", audio_data=part.inline_data.data)
                                    
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                                    await self.send(text_data=json.dumps({
                                        "type": "audio_response",
                                        "audio_data": audio_b64,
                                        "mime_type": part.inline_data.mime_type
                                    }))

                        # Handle text response
                        # Note: part.thought is a boolean flag indicating if text is "thinking"
                        if hasattr(part, 'text') and part.text:
                            is_thought = getattr(part, 'thought', False) is True
                            if is_thought:
                                logger.info(f"Model thought: {part.text[:100]}...")
                                # Save thought to MongoDB
                                await self._save_message_to_db("assistant", "thought", content=part.text)
                            else:
                                logger.info(f"Model text: {part.text[:100]}...")
                                # Save text response to MongoDB
                                await self._save_message_to_db("assistant", "text", content=part.text)
                            await self.send(text_data=json.dumps({
                                "type": "text_response",
                                "text": part.text
                            }))

            # Handle tool calls (Google Search)
            if hasattr(response, 'tool_call') and response.tool_call:
                logger.info(f"Tool call: {response.tool_call}")
                await self.send(text_data=json.dumps({
                    "type": "tool_call",
                    "tool": "google_search",
                    "status": "executing"
                }))

            # Handle tool call cancellation
            if hasattr(response, 'tool_call_cancellation') and response.tool_call_cancellation:
                await self.send(text_data=json.dumps({
                    "type": "tool_call_cancelled"
                }))

        except Exception as e:
            logger.error(f"Error handling Gemini response: {e}", exc_info=True)

    async def _save_message_to_db(self, role: str, message_type: str, content: str = None, audio_data: bytes = None):
        """
        Save a message to MongoDB conversation history.
        
        Args:
            role: 'user' or 'assistant'
            message_type: 'text', 'audio', 'image', or 'thought'
            content: Text content of the message
            audio_data: Binary audio data if message_type is 'audio'
        """
        try:
            if not self.conversation:
                logger.warning("No conversation document initialized")
                return
            
            message = ConversationMessage(
                role=role,
                message_type=message_type,
                content=content,
                audio_data=audio_data,
                images=self.current_turn_images if role == "user" else []
            )
            
            # Add message to conversation
            self.conversation.messages.append(message)
            
            # Update metadata
            if message_type == 'image':
                self.conversation.total_images = str(int(self.conversation.total_images or 0) + 1)
            elif message_type == 'audio':
                self.conversation.total_audio_chunks = str(int(self.conversation.total_audio_chunks or 0) + 1)
            
            self.conversation.total_messages = str(int(self.conversation.total_messages or 0) + 1)
            self.conversation.last_activity = datetime.utcnow()
            
            # Save to database
            self.conversation.save()
            logger.debug(f"Saved {message_type} message from {role} to MongoDB")
            
            # Clear images for next turn
            if role == "assistant":
                self.current_turn_images = []
                
        except Exception as e:
            logger.error(f"Error saving message to MongoDB: {e}")

    async def _save_image_to_db(self, image_bytes: bytes, mime_type: str):
        """
        Save an image to the current conversation turn.
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
            
            logger.debug(f"Buffered image ({size_kb} KB) for current turn")
            
        except Exception as e:
            logger.error(f"Error processing image for MongoDB: {e}")

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
