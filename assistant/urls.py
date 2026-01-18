"""
URL routing for assistant app
"""

from django.urls import path
from . import views

urlpatterns = [
    # Conversation management
    path('conversations/', views.get_conversations, name='get_conversations'),
    path('conversations/<str:conversation_id>/', views.get_conversation_detail, name='get_conversation_detail'),
    path('conversations/<str:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('conversations/<str:conversation_id>/update-title/', views.update_conversation_title, name='update_conversation_title'),
    path('conversations/<str:conversation_id>/image/<int:message_index>/<int:image_index>/', views.get_image_from_conversation, name='get_image_from_conversation'),
]
