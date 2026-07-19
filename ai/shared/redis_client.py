"""Redis Stream 消费封装 — 解析 ingest 写入的原始消息"""
import json
import os
import numpy as np
import redis
from ai.shared.types import PointCloudFrame, VitalFrame

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
STREAM_POINTCLOUD = "housafe:pointcloud:ingest"
STREAM_VITAL = "housafe:vital:ingest"


def parse_pointcloud_message(data: dict[bytes, bytes]) -> PointCloudFrame:
    """解析 Redis Stream 中的点云消息"""
    points_raw = data.get(b"points", b"[]")
    pts = json.loads(points_raw) if points_raw else []
    arr = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 5), dtype=np.float32)
    return PointCloudFrame(
        ts=int(data.get(b"ts", b"0")),
        device_id=data.get(b"device_id", b"unknown").decode(),
        family_id=data.get(b"family_id", b"unknown").decode(),
        room=data.get(b"room", b"unknown").decode(),
        frame_id=data.get(b"frame_id", b"unknown").decode(),
        points=arr,
    )


def parse_vital_message(data: dict[bytes, bytes]) -> VitalFrame:
    """解析 Redis Stream 中的生命体征消息"""
    def _float_or_none(key: bytes) -> float | None:
        raw = data.get(key, b"")
        if not raw:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    return VitalFrame(
        ts=int(data.get(b"ts", b"0")),
        device_id=data.get(b"device_id", b"unknown").decode(),
        family_id=data.get(b"family_id", b"unknown").decode(),
        room=data.get(b"room", b"unknown").decode(),
        resp_rate=_float_or_none(b"resp_rate"),
        heart_rate=_float_or_none(b"heart_rate"),
        quality=_float_or_none(b"quality") or 0.0,
    )


class FrameConsumer:
    """从 Redis Stream 消费帧消息的迭代器"""

    def __init__(self, redis_url: str = REDIS_URL):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self._last_id = "$"

    def consume_one(self) -> tuple[str, PointCloudFrame | VitalFrame | None, bytes] | None:
        """
        阻塞式读取一条消息。
        Returns: (stream_name, parsed_frame, raw_message_id) 或 None（超时无消息）
        """
        try:
            resp = self._redis.xread(
                {STREAM_POINTCLOUD: self._last_id, STREAM_VITAL: self._last_id},
                count=1, block=1000,
            )
        except redis.ConnectionError:
            return None

        if resp is None:
            return None

        for stream_name_bytes, messages in resp:
            for msg_id, msg_data in messages:
                self._last_id = msg_id
                stream_name = stream_name_bytes.decode() if isinstance(stream_name_bytes, bytes) else stream_name_bytes
                if stream_name == STREAM_POINTCLOUD:
                    frame = parse_pointcloud_message(msg_data)
                else:
                    frame = parse_vital_message(msg_data)
                return stream_name, frame, msg_id

        return None

    def close(self):
        self._redis.close()
