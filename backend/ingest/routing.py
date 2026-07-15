from django.urls import path
from .consumers import IngestConsumer

websocket_urlpatterns = [path("ws/ingest", IngestConsumer.as_asgi())]
