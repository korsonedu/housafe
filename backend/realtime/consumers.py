from urllib.parse import parse_qs

from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async


class AppConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        qs = parse_qs(self.scope["query_string"].decode())
        token = qs.get("token", [None])[0]
        family_id = qs.get("family", [None])[0]
        user = await self._auth(token)
        if user is None or not await self._owns(user, family_id):
            await self.close(code=4401)
            return
        self.group = f"family_{family_id}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    @database_sync_to_async
    def _auth(self, token):
        if not token:
            return None
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            from django.contrib.auth.models import User

            return User.objects.get(id=AccessToken(token)["user_id"])
        except Exception:
            return None

    @database_sync_to_async
    def _owns(self, user, family_id):
        from families.models import Family

        return Family.objects.filter(id=family_id, owner=user).exists()

    async def event_push(self, event):
        await self.send_json({"kind": event["kind"], "payload": event["payload"]})

    async def room_state(self, event):
        await self.send_json(event["payload"])

    async def disconnect(self, code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
