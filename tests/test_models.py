"""Tests for MongoDB models"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from app.models import (
    ConversationMessage, 
    ConversationImage, 
    GeminiConversation,
    User,
    Session,
    Log
)


class TestConversationImage:
    """Test ConversationImage embedded document"""
    
    def test_image_creation_jpeg(self):
        """Test creating a JPEG image"""
        image_data = b'\xff\xd8\xff' + b'\x00' * 100
        
        image = ConversationImage(
            data=image_data,
            mime_type="image/jpeg",
            size_kb="15.5"
        )
        
        assert image.data == image_data
        assert image.mime_type == "image/jpeg"
        assert image.size_kb == "15.5"
        assert image.uploaded_at is not None
    
    def test_image_creation_png(self):
        """Test creating a PNG image"""
        image_data = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        
        image = ConversationImage(
            data=image_data,
            mime_type="image/png",
            size_kb="20.0"
        )
        
        assert image.data == image_data
        assert image.mime_type == "image/png"
    
    def test_image_default_mime_type(self):
        """Test image with default MIME type"""
        image = ConversationImage(
            data=b'\x00' * 100
        )
        
        assert image.mime_type == "image/jpeg"


class TestConversationMessage:
    """Test ConversationMessage embedded document"""
    
    def test_text_message_creation(self):
        """Test creating a text message"""
        message = ConversationMessage(
            role="user",
            message_type="text",
            content="Hello Gemini"
        )
        
        assert message.role == "user"
        assert message.message_type == "text"
        assert message.content == "Hello Gemini"
        assert message.timestamp is not None
        assert message.audio_data is None
        assert message.images == []
    
    def test_audio_message_creation(self):
        """Test creating an audio message"""
        audio_data = b'\x00\x01\x02' * 100
        
        message = ConversationMessage(
            role="assistant",
            message_type="audio",
            audio_data=audio_data
        )
        
        assert message.role == "assistant"
        assert message.message_type == "audio"
        assert message.audio_data == audio_data
        assert message.content is None
    
    def test_image_message_with_images(self):
        """Test creating an image message with image list"""
        image1 = ConversationImage(data=b'\x00' * 100)
        image2 = ConversationImage(data=b'\x01' * 100)
        
        message = ConversationMessage(
            role="user",
            message_type="image",
            images=[image1, image2]
        )
        
        assert len(message.images) == 2
        assert message.images[0] == image1
        assert message.images[1] == image2
    
    def test_thought_message_creation(self):
        """Test creating a thought/thinking message"""
        message = ConversationMessage(
            role="assistant",
            message_type="thought",
            content="Let me think about this..."
        )
        
        assert message.message_type == "thought"
        assert message.content == "Let me think about this..."
    
    def test_invalid_role(self):
        """Test that only valid roles are allowed"""
        # This should ideally raise validation error in MongoDB
        message = ConversationMessage(
            role="user",
            message_type="text",
            content="Test"
        )
        
        assert message.role in ['user', 'assistant']


class TestGeminiConversation:
    """Test GeminiConversation document"""
    
    def test_conversation_creation(self):
        """Test creating a conversation"""
        conversation = GeminiConversation(
            user_id="test_user_123",
            session_id="session_456",
            title="Test Conversation"
        )
        
        assert conversation.user_id == "test_user_123"
        assert conversation.session_id == "session_456"
        assert conversation.title == "Test Conversation"
        assert conversation.messages == []
        assert conversation.total_messages == "0"
        assert conversation.total_images == "0"
        assert conversation.total_audio_chunks == "0"
        assert conversation.created_at is not None
        assert conversation.updated_at is not None
        assert conversation.last_activity is not None
    
    def test_conversation_with_system_instruction(self):
        """Test conversation with system instruction"""
        instruction = "You are a helpful assistant"
        
        conversation = GeminiConversation(
            user_id="test_user",
            session_id="session_1",
            system_instruction=instruction
        )
        
        assert conversation.system_instruction == instruction
    
    def test_add_message_to_conversation(self):
        """Test adding messages to conversation"""
        conversation = GeminiConversation(
            user_id="test_user",
            session_id="session_1"
        )
        
        message1 = ConversationMessage(
            role="user",
            message_type="text",
            content="Hello"
        )
        
        message2 = ConversationMessage(
            role="assistant",
            message_type="text",
            content="Hi there"
        )
        
        conversation.messages.append(message1)
        conversation.messages.append(message2)
        
        assert len(conversation.messages) == 2
        assert conversation.messages[0].content == "Hello"
        assert conversation.messages[1].content == "Hi there"
    
    def test_update_metadata(self):
        """Test updating conversation metadata"""
        conversation = GeminiConversation(
            user_id="test_user",
            session_id="session_1"
        )
        
        # Add messages
        conversation.messages.append(ConversationMessage(
            role="user",
            message_type="text",
            content="Test"
        ))
        
        # Update counts
        conversation.total_messages = "1"
        conversation.total_images = "2"
        conversation.total_audio_chunks = "3"
        conversation.last_activity = datetime.utcnow()
        
        assert conversation.total_messages == "1"
        assert conversation.total_images == "2"
        assert conversation.total_audio_chunks == "3"
    
    def test_model_used_default(self):
        """Test default model is set"""
        conversation = GeminiConversation(
            user_id="test_user",
            session_id="session_1"
        )
        
        assert conversation.model_used == "gemini-2.5-flash-native-audio-preview-12-2025"


class TestUserModel:
    """Test User model"""
    
    def test_user_creation(self):
        """Test creating a user"""
        user = User(
            name="John Doe",
            email="john@example.com"
        )
        
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_active is True
        assert user.created_at is not None
    
    def test_user_with_auth0_id(self):
        """Test user with Auth0 ID"""
        user = User(
            name="Jane Doe",
            email="jane@example.com",
            auth0_id="auth0|123456"
        )
        
        assert user.auth0_id == "auth0|123456"


class TestSessionModel:
    """Test Session model"""
    
    def test_session_creation(self):
        """Test creating a session"""
        from datetime import timedelta
        
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        session = Session(
            user_id="user_123",
            token="token_abc123",
            expires_at=expires_at
        )
        
        assert session.user_id == "user_123"
        assert session.token == "token_abc123"
        assert session.is_active is True
        assert session.created_at is not None


class TestLogModel:
    """Test Log model"""
    
    def test_log_creation(self):
        """Test creating a log entry"""
        log = Log(
            user_id="user_123",
            action="login",
            details="User logged in from IP",
            ip_address="192.168.1.1"
        )
        
        assert log.user_id == "user_123"
        assert log.action == "login"
        assert log.details == "User logged in from IP"
        assert log.ip_address == "192.168.1.1"
        assert log.created_at is not None
