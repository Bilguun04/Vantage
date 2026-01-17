"""
ASGI config for app project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.settings')

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

# Import routing after Django setup
from assistant import routing

application = ProtocolTypeRouter({
    # HTTP requests are handled by Django
    "http": django_asgi_app,
    # WebSocket connections are handled by Channels
    # Note: Add AllowedHostsOriginValidator wrapper for production
    "websocket": URLRouter(
        routing.websocket_urlpatterns
    ),
})
