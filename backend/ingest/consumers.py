import time
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from pydantic import ValidationError
from housafe_contracts.events import parse_event
from devices.models import RadarDevice
from events.store import store_event


class IngestConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.device = None
        await self.accept()

    @database_sync_to_async
    def _verify(self, did, secret):
        return RadarDevice.verify(did, secret)

    @database_sync_to_async
    def _family_id(self):
        return self.device.family_id

    @database_sync_to_async
    def _store(self, kind, model):
        store_event(self.device.device_id, kind, model, ts_recv=int(time.time() * 1000))

    async def receive_json(self, content):
        if self.device is None:
            self.device = await self._verify(content.get("device_id"), content.get("secret"))
            if self.device is None:
                await self.close(code=4401)
                return
            self.group = f"family_{await self._family_id()}"
            await self.channel_layer.group_add(self.group, self.channel_name)
            await self.send_json({"ack": "auth"})
            return
        try:
            model = parse_event(content["kind"], content["payload"])
        except (ValidationError, ValueError, KeyError):
            await self.send_json({"error": "invalid"})
            return
        await self._store(content["kind"], model)
        await self.channel_layer.group_send(
            self.group,
            {"type": "event.push", "kind": content["kind"], "payload": content["payload"]},
        )
        await self.send_json({"ack": "stored", "seq": model.seq})

    async def event_push(self, event):
        pass  # ingest consumer itself does not forward; realtime handles delivery (Task 8)

    async def disconnect(self, close_code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
