from mongoengine import Document, StringField, EmailField, DateTimeField, BooleanField
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
