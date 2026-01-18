from mongoengine import Document, StringField, EmailField, DateTimeField, BooleanField, ListField, EmbeddedDocument, EmbeddedDocumentField, BinaryField
from datetime import datetime

class User(Document):
    """User model for MongoDB"""
    name = StringField(required=True, max_length=100)
    email = EmailField(required=True, unique=True)
    auth0_id = StringField(unique=True)
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    is_active = BooleanField(default=True)
    
    meta = {
        'collection': 'users',
        'indexes': ['email', 'auth0_id']
    }

class Session(Document):
    """Session model for MongoDB"""
    user_id = StringField(required=True)
    token = StringField(required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    expires_at = DateTimeField(required=True)
    is_active = BooleanField(default=True)
    
    meta = {
        'collection': 'sessions',
        'indexes': ['user_id', 'token']
    }

class Log(Document):
    """Audit log model for MongoDB"""
    user_id = StringField()
    action = StringField(required=True)
    details = StringField()
    ip_address = StringField()
    created_at = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'logs',
        'indexes': ['user_id', 'created_at']
    }
class ConversationImage(EmbeddedDocument):
    """Embedded document for images/screenshots in conversation"""
    data = BinaryField(required=True)  # Binary image data
    mime_type = StringField(default="image/jpeg")  # e.g., "image/jpeg", "image/png"
    size_kb = StringField()  # Size in KB for reference
    uploaded_at = DateTimeField(default=datetime.utcnow)

class ConversationMessage(EmbeddedDocument):
    """Embedded document for individual messages in conversation"""
    role = StringField(choices=['user', 'assistant'], required=True)
    message_type = StringField(choices=['text', 'audio', 'image', 'thought'], default='text')
    content = StringField()  # Text content
    audio_data = BinaryField()  # Binary audio data if message_type is 'audio'
    images = ListField(EmbeddedDocumentField(ConversationImage))  # List of images
    timestamp = DateTimeField(default=datetime.utcnow)

class GeminiConversation(Document):
    """Gemini conversation session with history, images, and metadata"""
    user_id = StringField(required=True)  # Link to User
    session_id = StringField(required=True, unique=True)  # Unique session identifier
    title = StringField()  # Optional title for the conversation
    messages = ListField(EmbeddedDocumentField(ConversationMessage))
    
    # Metadata
    model_used = StringField(default="gemini-2.5-flash-native-audio-preview-12-2025")
    system_instruction = StringField()
    
    # Statistics
    total_messages = StringField(default=0)
    total_images = StringField(default=0)
    total_audio_chunks = StringField(default=0)
    
    # Timestamps
    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)
    last_activity = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'gemini_conversations',
        'indexes': [
            'user_id',
            'session_id',
            'created_at',
            ('user_id', '-created_at')
        ]
    }
