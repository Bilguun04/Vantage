"""Integration tests for the Gemini chat application"""

import pytest
import asyncio
import json
import base64
from unittest.mock import Mock, AsyncMock, patch
from assistant.consumers import GeminiLiveConsumer


class TestIntegrationConversationFlow:
    """Test complete conversation flows"""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_cycle(self, sample_audio_bytes, sample_image_bytes_jpeg):
        """Test a complete conversation cycle"""
        consumer = GeminiLiveConsumer()
        consumer.gemini_ready = True
        consumer.session = AsyncMock()
        consumer.send = AsyncMock()
        consumer.session.send_client_content = AsyncMock()
        consumer.session.send_realtime_input = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_images = "0"
        mock_conversation.total_audio_chunks = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Step 1: User sends text
        await consumer._send_text_to_gemini("What's on my screen?")
        assert len(mock_conversation.messages) == 1
        
        # Step 2: User sends screen frame
        image_b64 = base64.b64encode(sample_image_bytes_jpeg).decode('utf-8')
        data = json.dumps({"screen_frame": image_b64})
        await consumer.receive(text_data=data)
        assert len(consumer.current_turn_images) > 0
        
        # Step 3: Assistant responds with text
        mock_part = Mock()
        mock_part.text = "I can see a computer screen"
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
        
        # Verify the conversation has both messages
        assert len(mock_conversation.messages) >= 2
        
        # Step 4: User sends audio
        audio_b64 = base64.b64encode(sample_audio_bytes).decode('utf-8')
        data = json.dumps({"audio_chunk": audio_b64})
        await consumer.receive(text_data=data)
        consumer.session.send_realtime_input.assert_called()
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multiple conversation turns"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.send = AsyncMock()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Turn 1
        await consumer._save_message_to_db("user", "text", content="First question")
        await consumer._save_message_to_db("assistant", "text", content="First answer")
        
        # Turn 2
        await consumer._save_message_to_db("user", "text", content="Second question")
        await consumer._save_message_to_db("assistant", "text", content="Second answer")
        
        # Turn 3
        await consumer._save_message_to_db("user", "text", content="Third question")
        await consumer._save_message_to_db("assistant", "text", content="Third answer")
        
        # Verify all messages were saved
        assert len(mock_conversation.messages) == 6
        
        # Verify order
        assert mock_conversation.messages[0].content == "First question"
        assert mock_conversation.messages[1].content == "First answer"
        assert mock_conversation.messages[2].content == "Second question"
        assert mock_conversation.messages[3].content == "Second answer"


class TestIntegrationErrorScenarios:
    """Test error handling in integrated scenarios"""
    
    @pytest.mark.asyncio
    async def test_recovery_from_send_error(self):
        """Test recovery after send error"""
        consumer = GeminiLiveConsumer()
        consumer.session = AsyncMock()
        consumer.session.send_client_content = AsyncMock(
            side_effect=Exception("Send failed")
        )
        consumer.send = AsyncMock()
        
        # Should not raise exception
        await consumer._send_text_to_gemini("Test message")
        
        # Error message should be logged but not crash
    
    @pytest.mark.asyncio
    async def test_conversation_without_api_key(self):
        """Test graceful failure without API key"""
        with patch.dict('os.environ', {}, clear=True):
            consumer = GeminiLiveConsumer()
            consumer.accept = AsyncMock()
            consumer.send = AsyncMock()
            consumer.close = AsyncMock()
            consumer.scope = {'user': {'id': 'test'}}
            
            await consumer.connect()
            
            # Should handle gracefully
            assert consumer.gemini_ready is False


class TestIntegrationDataPersistence:
    """Test data persistence across operations"""
    
    @pytest.mark.asyncio
    async def test_images_attached_to_user_message(self):
        """Test that images are properly attached to user messages"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_images = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Add image
        image = Mock()
        consumer.current_turn_images = [image]
        
        # Save user message
        await consumer._save_message_to_db("user", "text", content="Look at this")
        
        # Images should be attached to the message
        user_message = mock_conversation.messages[0]
        assert len(user_message.images) > 0
    
    @pytest.mark.asyncio
    async def test_images_cleared_on_assistant_response(self):
        """Test that images are cleared after assistant response"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Add image
        image = Mock()
        consumer.current_turn_images = [image]
        assert len(consumer.current_turn_images) == 1
        
        # Assistant response should clear images
        await consumer._save_message_to_db("assistant", "text", content="Response")
        
        assert len(consumer.current_turn_images) == 0
    
    @pytest.mark.asyncio
    async def test_session_metadata_updated(self):
        """Test that session metadata is properly updated"""
        consumer = GeminiLiveConsumer()
        
        mock_conversation = Mock()
        mock_conversation.messages = []
        mock_conversation.total_messages = "0"
        mock_conversation.total_images = "0"
        mock_conversation.total_audio_chunks = "0"
        mock_conversation.last_activity = None
        mock_conversation.save = Mock()
        consumer.conversation = mock_conversation
        
        # Add text message
        await consumer._save_message_to_db("user", "text", content="Test")
        
        # Check metadata was updated
        assert mock_conversation.total_messages == "1"
        assert mock_conversation.last_activity is not None
        
        # Add audio message
        await consumer._save_message_to_db(
            "assistant", 
            "audio", 
            audio_data=b'\x00' * 100
        )
        
        # Check audio count
        assert mock_conversation.total_audio_chunks == "1"
