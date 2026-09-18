"""
AI 占位符：读 Redis Stream 点云帧 → 模拟推理 → POST decoder output 回 backend。

用法：python main.py [redis_url] [backend_url] [token]
"""
import json
import os
import random
import sys
import time

import redis
import requests

REDIS_URL = os.environ.get("REDIS_URL", sys.argv[1] if len(sys.argv) > 1 else "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("BACKEND_URL", sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000")
INTERNAL_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", sys.argv[3] if len(sys.argv) > 3 else "dev-internal-token")

STREAM_POINTCLOUD = "housafe:pointcloud:ingest"
STREAM_VITAL = "housafe:vital:ingest"
DECODER_ENDPOINT = f"{BACKEND_URL}/api/internal/decoder-output"

# 简单规则：根据点云最小包围盒高度判断姿态
POSTURE_BY_HEIGHT = [
    (0.0, "lie"),     # 高度 < 0.3m → 躺卧
    (0.3, "sit"),     # 高度 < 0.8m → 坐着
    (0.8, "stand"),   # 高度 < 1.8m → 站立
    (1.8, "walk"),    # 高度 ≥ 1.8m → 走动
]


def infer_posture(points: list[list[float]]) -> tuple[str, float]:
    """根据点云计算包围盒高度 → 映射姿态 + 模拟置信度"""
    if not points:
        return "stand", 0.5
    zs = [p[2] for p in points]
    height = max(zs) - min(zs) if len(zs) > 1 else 0.5
    posture = "stand"
    for threshold, label in POSTURE_BY_HEIGHT:
        if height >= threshold:
            posture = label
    confidence = round(random.uniform(0.7, 0.99), 2)
    return posture, confidence


def infer_vital() -> dict:
    """模拟生命体征"""
    return {
        "resp_rate": round(random.uniform(12, 20), 1),
        "heart_rate": round(random.uniform(60, 90), 1),
        "quality": round(random.uniform(0.7, 0.99), 2),
    }


def process_frame(stream: str, data: dict) -> list[dict]:
    """处理一帧 → 返回 Decoder output 列表"""
    ts = int(time.time() * 1000)
    device_id = data.get(b"device_id", b"unknown").decode()
    family_id = data.get(b"family_id", b"unknown").decode()
    room = data.get(b"room", b"unknown").decode()
    outputs = []

    if stream == STREAM_POINTCLOUD:
        points_raw = data.get(b"points", b"[]")
        points = json.loads(points_raw) if points_raw else []
        posture, confidence = infer_posture(points)
        outputs.append({
            "kind": "posture",
            "family_id": family_id,
            "payload": {
                "ts": ts, "device_id": device_id, "room": room,
                "posture": posture, "confidence": confidence,
                "moving": len(points) > 5, "presence": True,
            },
        })

    elif stream == STREAM_VITAL:
        vital = infer_vital()
        outputs.append({
            "kind": "vital",
            "family_id": family_id,
            "payload": {
                "ts": ts, "device_id": device_id, "room": room,
                **vital,
            },
        })

    return outputs


def main():
    r = redis.Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=5, socket_connect_timeout=3)
    print(f"[AI] connected to Redis: {REDIS_URL}")
    print(f"[AI] backend: {DECODER_ENDPOINT}")
    print(f"[AI] listening on: {STREAM_POINTCLOUD}, {STREAM_VITAL}")

    # 从当前最新消息开始消费
    last_id = "$"

    while True:
        try:
            resp = r.xread(
                {STREAM_POINTCLOUD: last_id, STREAM_VITAL: last_id},
                count=1,
                block=2000,
            )
        except (redis.ConnectionError, redis.TimeoutError):
            print("[AI] Redis timeout/reconnect, retrying...")
            time.sleep(1)
            try:
                r = redis.Redis.from_url(REDIS_URL, decode_responses=False, socket_timeout=5, socket_connect_timeout=3)
            except Exception:
                pass
            last_id = "$"
            continue

        if resp is None:
            continue

        for stream_name, messages in resp:
            for msg_id, msg_data in messages:
                last_id = msg_id
                msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id

                # 模拟推理延迟
                delay = random.uniform(0.3, 1.0)
                time.sleep(delay)

                outputs = process_frame(stream_name.decode() if isinstance(stream_name, bytes) else stream_name, msg_data)
                if not outputs:
                    continue

                try:
                    rv = requests.post(
                        DECODER_ENDPOINT,
                        json={"token": INTERNAL_TOKEN, "outputs": outputs},
                        timeout=5,
                    )
                    kind = outputs[0]["kind"]
                    if rv.status_code == 200:
                        print(f"[AI] {kind} {msg_id_str}: ok ({rv.json().get('count', 0)} stored)")
                    else:
                        print(f"[AI] {kind} {msg_id_str}: HTTP {rv.status_code} {rv.text[:100]}")
                except requests.RequestException as e:
                    print(f"[AI] POST failed: {e}")


if __name__ == "__main__":
    main()
