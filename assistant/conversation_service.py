"""
Service for managing Gemini conversation data in MongoDB.
Provides utilities for retrieving, updating, and exporting conversations.
"""

import logging
from typing import List, Optional
from datetime import datetime
from app.models import GeminiConversation, ConversationMessage

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for conversation management"""

    @staticmethod
    def get_user_conversations(user_id: str, limit: int = 10) -> List[GeminiConversation]:
        """
        Get all conversations for a user, sorted by most recent first.
        
        Args:
            user_id: User ID to fetch conversations for
            limit: Maximum number of conversations to return
            
        Returns:
            List of GeminiConversation documents
        """
        try:
            conversations = GeminiConversation.objects(user_id=user_id).order_by('-created_at').limit(limit)
            return list(conversations)
        except Exception as e:
            logger.error(f"Error fetching user conversations: {e}")
            return []

    @staticmethod
    def get_conversation_by_id(conversation_id: str) -> Optional[GeminiConversation]:
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: MongoDB ObjectId of the conversation
            
        Returns:
            GeminiConversation document or None if not found
        """
        try:
            return GeminiConversation.objects.get(id=conversation_id)
        except Exception as e:
            logger.error(f"Error fetching conversation {conversation_id}: {e}")
            return None

    @staticmethod
    def get_conversation_by_session_id(session_id: str) -> Optional[GeminiConversation]:
        """
        Get a conversation by session ID.
        
        Args:
            session_id: Session ID generated during connection
            
        Returns:
            GeminiConversation document or None if not found
        """
        try:
            return GeminiConversation.objects.get(session_id=session_id)
        except Exception as e:
            logger.error(f"Error fetching conversation by session {session_id}: {e}")
            return None

    @staticmethod
    def search_conversations(user_id: str, query: str) -> List[GeminiConversation]:
        """
        Search conversations by title or content.
        
        Args:
            user_id: User ID to search within
            query: Search query string
            
        Returns:
            List of matching GeminiConversation documents
        """
        try:
            conversations = GeminiConversation.objects(
                user_id=user_id,
                messages__content__icontains=query
            ).order_by('-created_at')
            return list(conversations)
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: MongoDB ObjectId of the conversation
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            conversation = GeminiConversation.objects.get(id=conversation_id)
            conversation.delete()
            logger.info(f"Deleted conversation {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            return False

    @staticmethod
    def update_conversation_title(conversation_id: str, new_title: str) -> bool:
        """
        Update a conversation's title.
        
        Args:
            conversation_id: MongoDB ObjectId of the conversation
            new_title: New title for the conversation
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            conversation = GeminiConversation.objects.get(id=conversation_id)
            conversation.title = new_title
            conversation.updated_at = datetime.utcnow()
            conversation.save()
            logger.info(f"Updated conversation {conversation_id} title to: {new_title}")
            return True
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {e}")
            return False

    @staticmethod
    def get_conversation_stats(conversation_id: str) -> Optional[dict]:
        """
        Get statistics for a conversation.
        
        Args:
            conversation_id: MongoDB ObjectId of the conversation
            
        Returns:
            Dictionary with conversation statistics or None if not found
        """
        try:
            conversation = GeminiConversation.objects.get(id=conversation_id)
            
            user_messages = len([m for m in conversation.messages if m.role == "user"])
            assistant_messages = len([m for m in conversation.messages if m.role == "assistant"])
            image_count = len([m for m in conversation.messages if m.message_type == "image"])
            
            duration = (conversation.updated_at - conversation.created_at).total_seconds()
            
            return {
                "conversation_id": str(conversation.id),
                "session_id": conversation.session_id,
                "title": conversation.title,
                "total_messages": int(conversation.total_messages or 0),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages,
                "total_images": int(conversation.total_images or 0),
                "image_count": image_count,
                "total_audio_chunks": int(conversation.total_audio_chunks or 0),
                "duration_seconds": duration,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting conversation stats {conversation_id}: {e}")
            return None

    @staticmethod
    def export_conversation_as_dict(conversation_id: str) -> Optional[dict]:
        """
        Export a conversation as a dictionary (useful for APIs/JSON).
        
        Args:
            conversation_id: MongoDB ObjectId of the conversation
            
        Returns:
            Dictionary representation of the conversation or None if not found
        """
        try:
            conversation = GeminiConversation.objects.get(id=conversation_id)
            
            # Build message list
            messages = []
            for msg in conversation.messages:
                msg_dict = {
                    "role": msg.role,
                    "type": msg.message_type,
                    "timestamp": msg.timestamp.isoformat(),
                }
                
                if msg.content:
                    msg_dict["content"] = msg.content
                
                if msg.images:
                    msg_dict["images"] = len(msg.images)
                    msg_dict["image_info"] = [
                        {
                            "mime_type": img.mime_type,
                            "size_kb": img.size_kb,
                            "uploaded_at": img.uploaded_at.isoformat()
                        }
                        for img in msg.images
                    ]
                
                if msg.audio_data:
                    msg_dict["audio_size_bytes"] = len(msg.audio_data)
                
                messages.append(msg_dict)
            
            return {
                "conversation_id": str(conversation.id),
                "session_id": conversation.session_id,
                "user_id": conversation.user_id,
                "title": conversation.title,
                "model_used": conversation.model_used,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "messages": messages,
                "stats": {
                    "total_messages": int(conversation.total_messages or 0),
                    "total_images": int(conversation.total_images or 0),
                    "total_audio_chunks": int(conversation.total_audio_chunks or 0),
                }
            }
        except Exception as e:
            logger.error(f"Error exporting conversation {conversation_id}: {e}")
            return None
