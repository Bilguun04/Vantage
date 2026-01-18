"""
Views for displaying and managing Gemini conversations
"""

import logging
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from assistant.conversation_service import ConversationService

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def get_conversations(request):
    """
    Display all conversations for the current user in dashboard.
    
    Query parameters:
        limit: Number of conversations to return (default: 10)
        api: If 'true', returns JSON instead of HTML
    """
    try:
        user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
        limit = int(request.GET.get('limit', 10))
        
        conversations = ConversationService.get_user_conversations(user_id, limit)
        
        data = []
        for conv in conversations:
            data.append({
                "id": str(conv.id),
                "session_id": conv.session_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
                "message_count": int(conv.total_messages or 0),
                "image_count": int(conv.total_images or 0),
            })
        
        # Return JSON if api parameter is set
        if request.GET.get('api') == 'true':
            return JsonResponse({
                "success": True,
                "count": len(data),
                "conversations": data
            })
        
        # Return HTML dashboard
        return render(request, 'dashboard.html', {
            'conversations': data,
            'userinfo': request.session.get('userinfo', {})
        })
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        if request.GET.get('api') == 'true':
            return JsonResponse({
                "success": False,
                "error": str(e)
            }, status=500)
        return render(request, 'dashboard.html', {
            'conversations': [],
            'error': str(e),
            'userinfo': request.session.get('userinfo', {})
        })


@require_http_methods(["GET"])
def get_conversation_detail(request, conversation_id):
    """
    Get detailed information about a specific conversation including all messages.
    
    URL parameters:
        conversation_id: MongoDB ObjectId of the conversation
    """
    try:
        conversation = ConversationService.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return JsonResponse({
                "success": False,
                "error": "Conversation not found"
            }, status=404)
        
        # Verify user owns this conversation
        user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
        if conversation.user_id != user_id:
            return JsonResponse({
                "success": False,
                "error": "Unauthorized"
            }, status=403)
        
        # Build detailed response
        messages = []
        for msg in conversation.messages:
            msg_data = {
                "role": msg.role,
                "type": msg.message_type,
                "timestamp": msg.timestamp.isoformat(),
            }
            
            if msg.content:
                msg_data["content"] = msg.content
            
            # Include image information (without the binary data)
            if msg.images:
                msg_data["images"] = [
                    {
                        "mime_type": img.mime_type,
                        "size_kb": img.size_kb,
                        "uploaded_at": img.uploaded_at.isoformat()
                    }
                    for img in msg.images
                ]
            
            # Include audio info without binary data
            if msg.audio_data:
                msg_data["audio_size_bytes"] = len(msg.audio_data)
            
            messages.append(msg_data)
        
        return JsonResponse({
            "success": True,
            "conversation": {
                "id": str(conversation.id),
                "session_id": conversation.session_id,
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "model_used": conversation.model_used,
                "stats": {
                    "total_messages": int(conversation.total_messages or 0),
                    "total_images": int(conversation.total_images or 0),
                    "total_audio_chunks": int(conversation.total_audio_chunks or 0),
                },
                "messages": messages
            }
        })
    except Exception as e:
        logger.error(f"Error fetching conversation detail: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def delete_conversation(request, conversation_id):
    """
    Delete a conversation.
    
    URL parameters:
        conversation_id: MongoDB ObjectId of the conversation
    """
    try:
        conversation = ConversationService.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return JsonResponse({
                "success": False,
                "error": "Conversation not found"
            }, status=404)
        
        # Verify user owns this conversation
        user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
        if conversation.user_id != user_id:
            return JsonResponse({
                "success": False,
                "error": "Unauthorized"
            }, status=403)
        
        success = ConversationService.delete_conversation(conversation_id)
        
        return JsonResponse({
            "success": success,
            "message": "Conversation deleted successfully" if success else "Failed to delete conversation"
        })
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


@require_http_methods(["POST"])
@csrf_exempt
def update_conversation_title(request, conversation_id):
    """
    Update a conversation's title.
    
    URL parameters:
        conversation_id: MongoDB ObjectId of the conversation
    
    POST data:
        title: New title for the conversation
    """
    try:
        import json
        data = json.loads(request.body)
        new_title = data.get('title', '')
        
        if not new_title:
            return JsonResponse({
                "success": False,
                "error": "Title cannot be empty"
            }, status=400)
        
        conversation = ConversationService.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return JsonResponse({
                "success": False,
                "error": "Conversation not found"
            }, status=404)
        
        # Verify user owns this conversation
        user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
        if conversation.user_id != user_id:
            return JsonResponse({
                "success": False,
                "error": "Unauthorized"
            }, status=403)
        
        success = ConversationService.update_conversation_title(conversation_id, new_title)
        
        return JsonResponse({
            "success": success,
            "message": "Title updated successfully" if success else "Failed to update title"
        })
    except Exception as e:
        logger.error(f"Error updating conversation title: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_image_from_conversation(request, conversation_id, message_index, image_index):
    """
    Retrieve a specific image from a conversation message.
    
    URL parameters:
        conversation_id: MongoDB ObjectId of the conversation
        message_index: Index of the message in the conversation
        image_index: Index of the image in the message's images list
    """
    try:
        conversation = ConversationService.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return JsonResponse({
                "success": False,
                "error": "Conversation not found"
            }, status=404)
        
        # Verify user owns this conversation
        user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
        if conversation.user_id != user_id:
            return JsonResponse({
                "success": False,
                "error": "Unauthorized"
            }, status=403)
        
        message = conversation.messages[int(message_index)]
        if not message.images or len(message.images) <= int(image_index):
            return JsonResponse({
                "success": False,
                "error": "Image not found"
            }, status=404)
        
        image = message.images[int(image_index)]
        
        return JsonResponse({
            "mime_type": image.mime_type,
            "size_kb": image.size_kb,
            "uploaded_at": image.uploaded_at.isoformat()
        })
    except (IndexError, ValueError) as e:
        return JsonResponse({
            "success": False,
            "error": "Invalid index"
        }, status=400)
    except Exception as e:
        logger.error(f"Error retrieving image: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)
