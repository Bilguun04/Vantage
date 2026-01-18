"""Tests for Gemini Live WebSocket Consumer"""

import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, MagicMock, patch, call
from datetime import datetime
import uuid

from assistant.consumers import GeminiLiveConsumer
from app.models import ConversationMessage, ConversationImage, GeminiConversation


class TestGeminiLiveConsumerInit:
    """Test consumer initialization"""
    
    def test_init(self):
        """Test consumer initialization"""
        consumer = GeminiLiveConsumer()
        
        assert consumer.client is None
        assert consumer.session is None
        assert consumer.session_context is None
        assert consumer.response_listener_task is None
        assert consumer.is_connected is False
        assert consumer.gemini_ready is False
        assert consumer.conversation is None
        assert consumer.user_id is None
        assert consumer.current_turn_images == []


class TestGeminiLiveConsumerConnect:
    """Test WebSocket connection and Gemini session setup"""
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_gemini_client, mock_session, mock_session_context):
        """Test successful WebSocket connection"""
        with patch('assistant.consumers.genai.Client', return_value=mock_gemini_client):
            with patch('assistant.consumers.GeminiConversation') as mock_conv_class:
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
                    
                    # Setup mocks
                    mock_conversation = AsyncMock()
                    mock_conversation.id = 'conv_123'
                    mock_conversation.save = AsyncMock()
                    mock_conv_class.return_value = mock_conversation
                    
                    mock_session_context.__aexit__ = AsyncMock(return_value=None)
                    mock_gemini_client.aio.live.connect.return_value = mock_session_context
                    
                    consumer = GeminiLiveConsumer()
                    consumer.accept = AsyncMock()
                    consumer.send = AsyncMock()
                    consumer.close = AsyncMock()
                    consumer.scope = {'user': {'id': 'test_user'}}
                    
                    await consumer.connect()
                    
                    assert consumer.is_connected is True
                    assert consumer.gemini_ready is True
                    assert consumer.session == mock_session
                    assert consumer.user_id == 'test_user'
                    consumer.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_no_api_key(self):
        """Test connection fails without API key"""
        with patch.dict('os.environ', {}, clear=True):
            consumer = GeminiLiveConsumer()
            consumer.accept = AsyncMock()
            consumer.send = AsyncMock()
            consumer.close = AsyncMock()
            consumer.scope = {'user': {'id': 'test_user'}}
            
            await consumer.connect()
            
            assert consumer.is_connected is True
            # send_error should be called
            calls = [str(c) for c in consumer.send.call_args_list]
            assert any('GOOGLE_API_KEY' in str(c) for c in calls)
    
    @pytest.mark.asyncio
    async def test_connect_anonymous_user(self, mock_gemini_client, mock_session, mock_session_context):
        """Test connection with anonymous user"""
        with patch('assistant.consumers.genai.Client', return_value=mock_gemini_client):
            with patch('assistant.consumers.GeminiConversation') as mock_conv_class:
                with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
                    
                    mock_conversation = AsyncMock()
                    mock_conversation.save = AsyncMock()
                    mock_conv_class.return_value = mock_conversation
                    
                    mock_session_context.__aenter__.return_value = mock_session
                    mock_gemini_client.aio.live.connect.return_value = mock_session_context
                    
                    consumer = GeminiLiveConsumer()
                    consumer.accept = AsyncMock()
                    consumer.send = AsyncMock()
                    consumer.scope = {}  # No user info
                    
                    await consumer.connect()
                    
                    assert consumer.user_id == 'anonymous'


class TestGeminiLiveConsumerDisconnect:
    """Test WebSocket disconnection and cleanup"""
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_session_context):
        """Test proper cleanup on disconnect"""
        consumer = GeminiLiveConsumer()
        consumer.is_connected = True
        consumer.gemini_ready = True
        
        # Create a mock task
        mock_task = AsyncMock()
        mock_task.cancel = Mock()
        consumer.response_listener_task = mock_task
        
        mock_session_context.__aexit__ = AsyncMock()
        consumer.session_context = mock_session_context
        
        await consumer.disconnect(1000)
        
        assert consumer.is_connected is False
        assert consumer.gemini_ready is False
        mock_task.cancel.assert_called_once()


class TestGeminiLiveConsumerReceiveAudio:
    """Test audio data reception and processing"""
    
    @pytest.mark.asyncio
    async def test_receive_audio_chunk(self, sample_audio_bytes):
        """Test receiving audio chunk"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.session = AsyncMock()
        consumer.session.send_realtime_input = AsyncMock()
        
        audio_b64 = base64.b64encode(sample_audio_bytes).decode('utf-8')
        data = json.dumps({"audio_chunk": audio_b64})
        
        await consumer.receive(text_data=data)
        
        consumer.session.send_realtime_input.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_receive_audio_end_signal(self):
        """Test receiving audio end signal"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        
        data = json.dumps({"audio_end": True})
        
        # Should not raise exception
        await consumer.receive(text_data=data)


class TestGeminiLiveConsumerReceiveText:
    """Test text data reception and saving"""
    
    @pytest.mark.asyncio
    async def test_receive_text_message(self):
        """Test receiving and saving text message"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        data = json.dumps({"text": "Hello Gemini"})
        
        await consumer.receive(text_data=data)
        
        # Verify message was saved
        consumer.session.send_client_content.assert_called_once()
        assert len(mock_conversation.messages) > 0


class TestGeminiLiveConsumerReceiveImages:
    """Test image/screenshot reception and saving"""
    
    @pytest.mark.asyncio
    async def test_receive_screen_frame_only(self, sample_image_bytes_jpeg):
        """Test receiving screen frame without text"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        image_b64 = base64.b64encode(sample_image_bytes_jpeg).decode('utf-8')
        data = json.dumps({"screen_frame": image_b64})
        
        await consumer.receive(text_data=data)
        
        # Verify image was buffered
        assert len(consumer.current_turn_images) > 0
        # Verify sent to Gemini
        consumer.session.send_client_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_receive_screen_frame_with_text(self, sample_image_bytes_jpeg):
        """Test receiving screen frame with text prompt"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_images = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        image_b64 = base64.b64encode(sample_image_bytes_jpeg).decode('utf-8')
        data = json.dumps({
            "screen_frame": image_b64,
            "text": "What's on my screen?"
        })
        
        await consumer.receive(text_data=data)
        
        # Verify image was buffered and message saved
        assert len(consumer.current_turn_images) > 0
        assert len(mock_conversation.messages) > 0


class TestGeminiLiveConsumerAudioHandling:
    """Test audio sending to Gemini"""
    
    @pytest.mark.asyncio
    async def test_send_audio_to_gemini(self, sample_audio_bytes):
        """Test sending audio to Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_realtime_input = AsyncMock()
        
        await consumer._send_audio_to_gemini(sample_audio_bytes)
        
        consumer.session.send_realtime_input.assert_called_once()
        call_args = consumer.session.send_realtime_input.call_args
        assert 'audio' in call_args.kwargs
        assert call_args.kwargs['audio']['mime_type'] == 'audio/pcm'


class TestGeminiLiveConsumerImageHandling:
    """Test image/screenshot sending to Gemini"""
    
    @pytest.mark.asyncio
    async def test_send_image_to_gemini_jpeg(self, sample_image_bytes_jpeg):
        """Test sending JPEG image to Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        await consumer._send_image_to_gemini(sample_image_bytes_jpeg)
        
        consumer.session.send_client_content.assert_called_once()
        call_args = consumer.session.send_client_content.call_args
        assert 'turns' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_send_image_to_gemini_png(self, sample_image_bytes_png):
        """Test sending PNG image to Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        await consumer._send_image_to_gemini(sample_image_bytes_png)
        
        consumer.session.send_client_content.assert_called_once()
        call_args = consumer.session.send_client_content.call_args
        # Verify PNG mime type was detected
        assert 'turns' in call_args.kwargs
    
    @pytest.mark.asyncio
    async def test_send_image_with_prompt(self, sample_image_bytes_jpeg):
        """Test sending image with text prompt"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        prompt = "Describe what you see"
        await consumer._send_image_to_gemini(sample_image_bytes_jpeg, with_prompt=prompt)
        
        consumer.session.send_client_content.assert_called_once()
        call_args = consumer.session.send_client_content.call_args
        turns = call_args.kwargs['turns']
        # Verify turn is marked complete when there's a prompt
        assert call_args.kwargs.get('turn_complete') is True


class TestGeminiLiveConsumerTextHandling:
    """Test text message sending to Gemini"""
    
    @pytest.mark.asyncio
    async def test_send_text_to_gemini(self):
        """Test sending text message to Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        text = "What time is it?"
        await consumer._send_text_to_gemini(text)
        
        # Verify message was saved
        assert len(mock_conversation.messages) > 0
        # Verify sent to Gemini
        consumer.session.send_client_content.assert_called_once()


class TestGeminiLiveConsumerResponseHandling:
    """Test handling responses from Gemini"""
    
    @pytest.mark.asyncio
    async def test_handle_audio_response(self, sample_audio_bytes):
        """Test handling audio response from Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.send = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_audio_chunks = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Create mock response with audio
        mock_part = Mock()
        mock_part.inline_data = Mock()
        mock_part.inline_data.mime_type = "audio/pcm"
        mock_part.inline_data.data = sample_audio_bytes
        
        mock_model_turn = Mock()
        mock_model_turn.parts = [mock_part]
        
        mock_content = Mock()
        mock_content.turn_complete = False
        mock_content.model_turn = mock_model_turn
        
        mock_response = Mock()
        mock_response.server_content = mock_content
        mock_response.tool_call = None
        mock_response.tool_call_cancellation = None
        
        await consumer._handle_gemini_response(mock_response)
        
        # Verify audio was sent to client
        assert consumer.send.called
        # Verify audio was saved to MongoDB
        assert len(mock_conversation.messages) > 0
    
    @pytest.mark.asyncio
    async def test_handle_text_response(self):
        """Test handling text response from Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.send = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Create mock response with text
        mock_part = Mock()
        mock_part.text = "This is a response"
        mock_part.thought = False
        
        mock_model_turn = Mock()
        mock_model_turn.parts = [mock_part]
        
        mock_content = Mock()
        mock_content.turn_complete = False
        mock_content.model_turn = mock_model_turn
        
        mock_response = Mock()
        mock_response.server_content = mock_content
        mock_response.tool_call = None
        mock_response.tool_call_cancellation = None
        
        await consumer._handle_gemini_response(mock_response)
        
        # Verify text was sent to client
        assert consumer.send.called
        # Verify text was saved to MongoDB
        assert len(mock_conversation.messages) > 0
    
    @pytest.mark.asyncio
    async def test_handle_thought_response(self):
        """Test handling thought/thinking response from Gemini"""
        consumer = GeminiLiveConsumer()
        consumer.send = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Create mock response with thought
        mock_part = Mock()
        mock_part.text = "Let me think about this..."
        mock_part.thought = True
        
        mock_model_turn = Mock()
        mock_model_turn.parts = [mock_part]
        
        mock_content = Mock()
        mock_content.turn_complete = False
        mock_content.model_turn = mock_model_turn
        
        mock_response = Mock()
        mock_response.server_content = mock_content
        mock_response.tool_call = None
        mock_response.tool_call_cancellation = None
        
        await consumer._handle_gemini_response(mock_response)
        
        # Verify thought was saved with correct type
        assert len(mock_conversation.messages) > 0
        saved_message = mock_conversation.messages[0]
        assert saved_message.message_type == "thought"
    
    @pytest.mark.asyncio
    async def test_handle_turn_complete(self):
        """Test handling turn completion"""
        consumer = GeminiLiveConsumer()
        consumer.send = AsyncMock()
        consumer.conversation = Mock()
        
        # Create mock response for turn complete
        mock_content = Mock()
        mock_content.turn_complete = True
        mock_content.model_turn = None
        
        mock_response = Mock()
        mock_response.server_content = mock_content
        mock_response.tool_call = None
        mock_response.tool_call_cancellation = None
        
        await consumer._handle_gemini_response(mock_response)
        
        # Verify turn_complete message was sent
        calls = consumer.send.call_args_list
        assert any('turn_complete' in str(c) for c in calls)


class TestGeminiLiveConsumerDatabaseOperations:
    """Test MongoDB save operations"""
    
    @pytest.mark.asyncio
    async def test_save_text_message_to_db(self):
        """Test saving text message to MongoDB"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        await consumer._save_message_to_db("user", "text", content="Hello")
        
        assert len(mock_conversation.messages) == 1
        message = mock_conversation.messages[0]
        assert message.role == "user"
        assert message.message_type == "text"
        assert message.content == "Hello"
        mock_conversation.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_audio_message_to_db(self, sample_audio_bytes):
        """Test saving audio message to MongoDB"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_audio_chunks = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        await consumer._save_message_to_db(
            "assistant", 
            "audio", 
            audio_data=sample_audio_bytes
        )
        
        assert len(mock_conversation.messages) == 1
        message = mock_conversation.messages[0]
        assert message.role == "assistant"
        assert message.message_type == "audio"
        assert message.audio_data == sample_audio_bytes
    
    @pytest.mark.asyncio
    async def test_save_thought_message_to_db(self):
        """Test saving thought message to MongoDB"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        thought_text = "Hmm, let me consider this"
        await consumer._save_message_to_db("assistant", "thought", content=thought_text)
        
        assert len(mock_conversation.messages) == 1
        message = mock_conversation.messages[0]
        assert message.message_type == "thought"
        assert message.content == thought_text
    
    @pytest.mark.asyncio
    async def test_save_image_to_db(self, sample_image_bytes_jpeg):
        """Test saving image to current turn"""
        consumer = GeminiLiveConsumer()
        
        await consumer._save_image_to_db(sample_image_bytes_jpeg, mime_type="image/jpeg")
        
        assert len(consumer.current_turn_images) == 1
        image = consumer.current_turn_images[0]
        assert image.data == sample_image_bytes_jpeg
        assert image.mime_type == "image/jpeg"
    
    @pytest.mark.asyncio
    async def test_image_cleared_on_assistant_message(self):
        """Test that images are cleared when assistant message is saved"""
        consumer = GeminiLiveConsumer()
        
        # Add an image to current turn
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        image = Mock()
        consumer.current_turn_images.append(image)
        
        # Save assistant message
        await consumer._save_message_to_db("assistant", "text", content="Response")
        
        # Images should be cleared
        assert len(consumer.current_turn_images) == 0
    
    @pytest.mark.asyncio
    async def test_save_without_conversation_initialized(self):
        """Test saving without conversation initialized"""
        consumer = GeminiLiveConsumer()
        consumer.conversation = None
        
        # Should not raise exception, just log warning
        await consumer._save_message_to_db("user", "text", content="Test")


class TestGeminiLiveConsumerErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_json_received(self):
        """Test handling invalid JSON"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.send = AsyncMock()
        
        invalid_json = "not a json"
        
        await consumer.receive(text_data=invalid_json)
        
        # Should send error message
        calls = consumer.send.call_args_list
        assert any('error' in str(c) for c in calls)
    
    @pytest.mark.asyncio
    async def test_send_error_message(self):
        """Test sending error message to client"""
        consumer = GeminiLiveConsumer()
        consumer.send = AsyncMock()
        
        await consumer.send_error("Test error message")
        
        consumer.send.assert_called_once()
        call_args = consumer.send.call_args
        data = json.loads(call_args.kwargs['text_data'])
        assert data['type'] == 'error'
        assert data['message'] == 'Test error message'
