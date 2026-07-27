"""点云帧灌数脚本：python feed.py <device_id> <secret> [--mode walk|sit|lie|fall] [ws_url]"""
import asyncio
import json
import random
import sys
import time

import websockets

# 不同姿态对应的点云分布（简化为 3D 包围盒内的随机点）
MODE_POINTS = {
    "stand": lambda: _gen_points(z_min=0.0, z_max=1.7, count=15),
    "sit": lambda: _gen_points(z_min=0.0, z_max=1.0, count=12),
    "lie": lambda: _gen_points(z_min=0.0, z_max=0.3, count=10),
    "walk": lambda: _gen_points(z_min=0.0, z_max=1.8, count=25),
    "fall": lambda: _gen_points(z_min=0.0, z_max=0.2, count=8),
}


def _gen_points(z_min: float, z_max: float, count: int) -> list[dict]:
    return [
        {
            "x": round(random.uniform(-2, 2), 3),
            "y": round(random.uniform(-2, 2), 3),
            "z": round(random.uniform(z_min, z_max), 3),
            "velocity": round(random.uniform(0, 1.5), 3),
            "intensity": round(random.uniform(0.3, 1.0), 3),
        }
        for _ in range(count)
    ]


async def main(device_id: str, secret: str, url: str = "ws://localhost:8000/ws/ingest"):
    mode = "walk"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "walk"
    if mode not in MODE_POINTS:
        print(f"unknown mode: {mode}, using walk")
        mode = "walk"

    async with websockets.connect(url) as ws:
        # 鉴权
        await ws.send(json.dumps({"device_id": device_id, "secret": secret}))
        ack = json.loads(await ws.recv())
        print(f"auth: {ack}")
        if ack.get("ack") != "auth":
            print("auth failed")
            return

        frame_seq = 0
        while True:
            frame_seq += 1
            ts = int(time.time() * 1000)

            # 发送点云帧
            frame_id = f"f-{frame_seq:04d}"
            points = MODE_POINTS[mode]()
            payload = {
                "ts": ts,
                "radar_id": device_id,
                "room": "bedroom",
                "frame_id": frame_id,
                "points": points,
            }
            await ws.send(json.dumps({"kind": "point_cloud", "payload": payload}))
            ack = json.loads(await ws.recv())
            print(f"  [{frame_id}] {mode}: {ack.get('ack')} ({len(points)} points)")

            # 每 5 帧发一次生命体征
            if frame_seq % 5 == 0:
                vital_payload = {
                    "ts": ts,
                    "radar_id": device_id,
                    "room": "bedroom",
                    "quiet": True,
                    "resp_rate": round(random.uniform(12, 20), 1),
                    "heart_rate": round(random.uniform(60, 90), 1),
                    "quality": round(random.uniform(0.7, 0.99), 2),
                }
                await ws.send(json.dumps({"kind": "vital", "payload": vital_payload}))
                ack = json.loads(await ws.recv())
                print(f"  [vital] hr={vital_payload['heart_rate']} rr={vital_payload['resp_rate']}: {ack.get('ack')}")

            # 每 10 帧发一次心跳
            if frame_seq % 10 == 0:
                hb_payload = {
                    "ts": ts,
                    "radar_id": device_id,
                    "status": "online",
                    "fw_version": "sim-0.2",
                }
                await ws.send(json.dumps({"kind": "heartbeat", "payload": hb_payload}))
                ack = json.loads(await ws.recv())
                print(f"  [heartbeat] fw={hb_payload['fw_version']}: {ack.get('ack')}")

            await asyncio.sleep(2)


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--") and a != "walk" and a != "sit" and a != "lie" and a != "fall"]
    device_id = argv[0] if len(argv) > 0 else "rad_demo"
    secret = argv[1] if len(argv) > 1 else "demo_secret"
    ws_url = argv[2] if len(argv) > 2 else "ws://localhost:8000/ws/ingest"
    asyncio.run(main(device_id, secret, ws_url))
