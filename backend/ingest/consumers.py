import os
import time
import redis
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from pydantic import ValidationError
from housafe_contracts.events import parse_frame
from devices.models import RadarDevice

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
STREAM_POINTCLOUD = "housafe:pointcloud:ingest"
STREAM_VITAL = "housafe:vital:ingest"


class IngestConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        self.device = None
        self._redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)
        await self.accept()

    @database_sync_to_async
    def _verify(self, did, secret):
        return RadarDevice.verify(did, secret)

    @database_sync_to_async
    def _family_id(self):
        return self.device.family_id

    @database_sync_to_async
    def _heartbeat(self, fw_version):
        from django.utils import timezone

        self.device.online = True
        self.device.last_heartbeat = timezone.now()
        self.device.fw_version = fw_version
        self.device.save(update_fields=["online", "last_heartbeat", "fw_version"])

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
            frame = parse_frame(content["kind"], content["payload"])
        except (ValidationError, ValueError, KeyError):
            await self.send_json({"error": "invalid"})
            return

        kind = content["kind"]

        if kind == "heartbeat":
            await self._heartbeat(frame.fw_version)
            await self.send_json({"ack": "heartbeat", "ts": frame.ts})
            return

        # point_cloud / vital → Redis Stream
        stream = STREAM_POINTCLOUD if kind == "point_cloud" else STREAM_VITAL
        family_id = await self._family_id()
        data = {
            b"family_id": str(family_id).encode(),
            b"device_id": self.device.device_id.encode(),
            b"ts": str(frame.ts).encode(),
            b"room": frame.room.encode(),
        }
        if kind == "point_cloud":
            data[b"frame_id"] = frame.frame_id.encode()
            data[b"n_points"] = str(len(frame.points)).encode()
            # 点云数据存为二进制 blob（简化：JSON 序列化 points 数组）
            import json
            pts = [[p.x, p.y, p.z, p.velocity, p.intensity] for p in frame.points]
            data[b"points"] = json.dumps(pts).encode()
        else:
            data[b"resp_rate"] = str(frame.resp_rate).encode() if frame.resp_rate is not None else b""
            data[b"heart_rate"] = str(frame.heart_rate).encode() if frame.heart_rate is not None else b""
            data[b"quality"] = str(frame.quality).encode()

        self._redis.xadd(stream, data, maxlen=1000, approximate=True)
        ack_id = frame.frame_id if kind == "point_cloud" else str(frame.ts)
        await self.send_json({"ack": "buffered", "frame_id": ack_id})

    async def event_push(self, event):
        pass  # ingest 不转发；realtime 负责投递

    async def room_state(self, event):
        pass  # room.state 由 realtime 投递，ingest 不处理

    async def disconnect(self, close_code):
        if hasattr(self, "group"):
            await self.channel_layer.group_discard(self.group, self.channel_name)
        if hasattr(self, "_redis"):
            self._redis.close()
