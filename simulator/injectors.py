"""双通道注入 — Redis Stream 直写 + WebSocket 全链路"""
import json
import asyncio
import redis
import websockets
from simulator.types import FrameGroup

# 与 ai/shared/redis_client.py 对齐
STREAM_POINTCLOUD = "housafe:pointcloud:ingest"
STREAM_VITAL = "housafe:vital:ingest"


class RedisInjector:
    """Redis Stream 直写 — 跳过 ingest，直接到 AI 消费端"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._gt_redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def inject(self, frames: list[FrameGroup]):
        """批量注入帧序列"""
        for fg in frames:
            self._inject_one(fg)

    def _inject_one(self, fg: FrameGroup):
        """注入单帧"""
        ts_str = str(fg.ts)
        device_str = fg.device_id
        room_str = fg.room

        # 点云 → houstafe:pointcloud:ingest
        if fg.points is not None and len(fg.points) > 0:
            pts_json = json.dumps([
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                 "velocity": float(p[3]), "intensity": float(p[4])}
                for p in fg.points
            ])
            self._redis.xadd(STREAM_POINTCLOUD, {
                "ts": ts_str,
                "device_id": device_str,
                "family_id": "sim",
                "room": room_str,
                "frame_id": fg.frame_id,
                "points": pts_json,
            }, maxlen=10000)

        # 体征 → housafe:vital:ingest
        if fg.vitals:
            self._redis.xadd(STREAM_VITAL, {
                "ts": ts_str,
                "device_id": device_str,
                "family_id": "sim",
                "room": room_str,
                "resp_rate": str(fg.vitals.resp_rate),
                "heart_rate": str(fg.vitals.heart_rate),
                "quality": str(fg.vitals.quality),
            }, maxlen=10000)

        # Ground truth → Redis String
        if fg.gt and (fg.gt.anomaly_type or fg.gt.posture):
            self._gt_redis.set(f"gt:{fg.frame_id}", json.dumps(fg.gt.to_dict()))

    def close(self):
        self._redis.close()
        self._gt_redis.close()


class WSInjector:
    """WebSocket 注入器 — 走 /ws/ingest 全链路"""

    async def inject(self, frames: list[FrameGroup], url: str,
                     device_id: str, secret: str):
        """通过 WebSocket 逐帧注入"""
        async with websockets.connect(url) as ws:
            # 鉴权（与 feed.py 协议一致）
            await ws.send(json.dumps({"device_id": device_id, "secret": secret}))
            ack = json.loads(await ws.recv())
            if ack.get("ack") != "auth":
                raise RuntimeError(f"WebSocket 鉴权失败: {ack}")

            for fg in frames:
                # 点云帧
                if fg.points is not None and len(fg.points) > 0:
                    payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "room": fg.room,
                        "frame_id": fg.frame_id,
                        "points": [
                            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                             "velocity": float(p[3]), "intensity": float(p[4])}
                            for p in fg.points
                        ],
                    }
                    await ws.send(json.dumps({"kind": "point_cloud", "payload": payload}))
                    await ws.recv()  # ack

                # 体征帧
                if fg.vitals:
                    vital_payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "room": fg.room,
                        "quiet": True,
                        "resp_rate": fg.vitals.resp_rate,
                        "heart_rate": fg.vitals.heart_rate,
                        "quality": fg.vitals.quality,
                    }
                    await ws.send(json.dumps({"kind": "vital", "payload": vital_payload}))
                    await ws.recv()  # ack

                # 心跳帧
                if fg.heartbeat:
                    hb_payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "status": "online",
                        "fw_version": "sim-1.0",
                    }
                    await ws.send(json.dumps({"kind": "heartbeat", "payload": hb_payload}))
                    await ws.recv()  # ack


async def inject_ws_async(frames: list[FrameGroup], url: str,
                          device_id: str, secret: str):
    """便捷函数：WS 异步注入"""
    injector = WSInjector()
    await injector.inject(frames, url, device_id, secret)
