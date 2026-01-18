from mongoengine import Document, StringField, EmailField, DateTimeField, BooleanField, ListField, EmbeddedDocument, EmbeddedDocumentField, BinaryField
from datetime import datetime

class User(Document):
    """User model for MongoDB"""
    name = StringField(required=True, max_length=100)
    email = EmailField(required=True, unique=True)
    auth0_id = StringField(unique=True, sparse=True)
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
    role = StringField(choices=['user', 'assistant', 'system'], required=True)
    message_type = StringField(choices=['text', 'audio', 'image', 'thought'], default='text')
    content = StringField()  # Text content
    audio_data = BinaryField()  # Binary audio data if message_type is 'audio'
    audio_duration_seconds = StringField()  # Human-readable audio duration
    audio_size_kb = StringField()  # Audio file size in KB
    images = ListField(EmbeddedDocumentField(ConversationImage))  # List of images
    timestamp = DateTimeField(default=datetime.utcnow)
    
    def to_readable_dict(self):
        """Convert message to human-readable dictionary (excludes binary data)"""
        result = {
            'role': self.role,
            'message_type': self.message_type,
            'timestamp': self.timestamp.strftime('%H:%M') if self.timestamp else None,
            'content': self.content or '',
        }
        
        if self.message_type == 'audio':
            result['audio_duration_seconds'] = self.audio_duration_seconds or 'unknown'
            result['audio_size_kb'] = self.audio_size_kb or 'unknown'
        
        if self.images:
            result['image_count'] = len(self.images)
        else:
            result['image_count'] = 0
        
        return result

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
    total_messages = StringField(default='0')
    total_images = StringField(default='0')
    total_audio_chunks = StringField(default='0')
    
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
    
    def to_readable_dict(self, include_audio_binary=False):
        """
        Convert conversation to human-readable dictionary.
        
        Args:
            include_audio_binary: If False (default), excludes binary audio data
        """
        return {
            'id': str(self.id),
            'session_id': self.session_id,
            'user_id': self.user_id,
            'title': self.title or 'Untitled Conversation',
            'model_used': self.model_used,
            'message_count': self.total_messages or '0',
            'image_count': self.total_images or '0',
            'audio_count': self.total_audio_chunks or '0',
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else None,
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M') if self.updated_at else None,
            'messages': [msg.to_readable_dict() for msg in self.messages]
        }
