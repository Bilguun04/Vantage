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
    Also displays seed data if user is not authenticated (for demo purposes).
    
    Query parameters:
        limit: Number of conversations to return (default: 10)
        api: If 'true', returns JSON instead of HTML
    """
    try:
        # Get user info from session
        userinfo = request.session.get('userinfo', {})
        user_id = userinfo.get('sub', None)
        user_name = userinfo.get('name', 'Guest')
        limit = int(request.GET.get('limit', 10))
        
        # Fetch conversations
        if user_id:
            # Authenticated user - fetch their conversations
            conversations = ConversationService.get_user_conversations(user_id, limit)
        else:
            # Not authenticated - show all conversations (demo mode)
            from app.models import GeminiConversation
            conversations = list(GeminiConversation.objects.order_by('-created_at').limit(limit))
        
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
            'user': {
                'name': user_name,
                'email': userinfo.get('email', 'demo@example.com')
            },
            'userinfo': userinfo
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
            'user': {
                'name': 'Guest',
                'email': 'demo@example.com'
            },
            'userinfo': {}
        })

@require_http_methods(["GET"])
def get_conversation_detail(request, conversation_id):
    """
    Get detailed information about a specific conversation including all messages.
    Returns human-readable format (no binary data).
    
    URL parameters:
        conversation_id: MongoDB ObjectId of the conversation
    
    Query parameters:
        skip_auth: If 'true', skip ownership check (for demo mode)
    """
    try:
        conversation = ConversationService.get_conversation_by_id(conversation_id)
        
        if not conversation:
            return JsonResponse({
                "success": False,
                "error": "Conversation not found"
            }, status=404)
        
        # Skip auth check in demo mode
        skip_auth = request.GET.get('skip_auth') == 'true'
        if not skip_auth:
            user_id = request.session.get('userinfo', {}).get('sub', 'anonymous')
            if conversation.user_id != user_id and user_id != 'anonymous':
                return JsonResponse({
                    "success": False,
                    "error": "Unauthorized"
                }, status=403)
        
        # Use the new to_readable_dict method for clean output
        return JsonResponse({
            "success": True,
            "conversation": conversation.to_readable_dict()
        })
    except Exception as e:
        logger.error(f"Error fetching conversation detail: {e}")
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)


@require_http_methods(["GET"])
def get_readable_conversations(request):
    """
    Get all conversations in a human-readable JSON format.
    Excludes binary data (audio, images) and shows metadata instead.
    
    Query parameters:
        limit: Number of conversations to return (default: 10)
    """
    try:
        from app.models import GeminiConversation
        
        limit = int(request.GET.get('limit', 10))
        conversations = list(GeminiConversation.objects.order_by('-created_at').limit(limit))
        
        readable_conversations = []
        for conv in conversations:
            readable_conversations.append(conv.to_readable_dict())
        
        return JsonResponse({
            "success": True,
            "count": len(readable_conversations),
            "conversations": readable_conversations
        }, json_dumps_params={'indent': 2})
        
    except Exception as e:
        logger.error(f"Error fetching readable conversations: {e}")
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
