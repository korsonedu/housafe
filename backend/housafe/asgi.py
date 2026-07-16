import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "housafe.settings")
from django.core.asgi import get_asgi_application
django_asgi = get_asgi_application()
from channels.routing import ProtocolTypeRouter, URLRouter
import ingest.routing
import realtime.routing
application = ProtocolTypeRouter({
    "http": django_asgi,
    "websocket": URLRouter(ingest.routing.websocket_urlpatterns + realtime.routing.websocket_urlpatterns),
})
