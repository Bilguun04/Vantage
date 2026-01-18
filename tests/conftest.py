"""Pytest configuration and shared fixtures for tests"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
import uuid

# Configure event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API client"""
    client = AsyncMock()
    client.aio = AsyncMock()
    client.aio.live = AsyncMock()
    return client


@pytest.fixture
def mock_conversation():
    """Mock MongoDB GeminiConversation document"""
    conversation = Mock()
    conversation.id = str(uuid.uuid4())
    conversation.user_id = "test_user"
    conversation.session_id = str(uuid.uuid4())
    conversation.messages = []
    conversation.total_messages = "0"
    conversation.total_images = "0"
    conversation.total_audio_chunks = "0"
    conversation.last_activity = datetime.utcnow()
    conversation.save = Mock()
    return conversation


@pytest.fixture
def mock_websocket_scope():
    """Mock WebSocket scope for consumer"""
    return {
        'type': 'websocket',
        'user': {'id': 'test_user_123'},
        'channel': Mock(),
    }


@pytest.fixture
def mock_session():
    """Mock Gemini Live session"""
    session = AsyncMock()
    session.send_realtime_input = AsyncMock()
    session.send_client_content = AsyncMock()
    session.receive = AsyncMock()
    return session


@pytest.fixture
def mock_session_context():
    """Mock session context manager"""
    context = MagicMock()
    # Make __aenter__ an async method that returns the mock session
    async def async_enter():
        return AsyncMock()
    context.__aenter__ = AsyncMock(return_value=AsyncMock())
    context.__aexit__ = AsyncMock(return_value=None)
    return context


@pytest.fixture
def sample_audio_bytes():
    """Sample audio bytes for testing"""
    return b'\x00\x01\x02\x03\x04\x05' * 100


@pytest.fixture
def sample_image_bytes_jpeg():
    """Sample JPEG image bytes"""
    # JPEG magic number: FF D8 FF
    return b'\xff\xd8\xff\xe0\x00\x10JFIF' + b'\x00' * 1000


@pytest.fixture
def sample_image_bytes_png():
    """Sample PNG image bytes"""
    # PNG magic number
    return b'\x89PNG\r\n\x1a\n' + b'\x00' * 1000
