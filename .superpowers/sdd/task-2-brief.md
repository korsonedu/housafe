### Task 2: Redis Stream 消费封装

**Files:**
- Create: `ai/shared/redis_client.py`
- Create: `ai/tests/test_redis_client.py`

**Interfaces:**
- Consumes: `PointCloudFrame`, `VitalFrame` from Task 1
- Produces: `FrameConsumer` class — `consume_one() -> tuple[str, PointCloudFrame | VitalFrame | None, dict]`

- [ ] **Step 1: Write test**

`ai/tests/test_redis_client.py`:

```python
"""测试 Redis Stream 消息解析（不需要真实 Redis）"""
import json
import numpy as np
from ai.shared.redis_client import parse_pointcloud_message, parse_vital_message

def test_parse_pointcloud_message():
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"frame_id": b"f-0042",
        b"n_points": b"3",
        b"points": json.dumps([
            [1.0, 2.0, 0.5, 0.1, 0.8],
            [1.1, 2.1, 1.5, 0.2, 0.9],
            [0.9, 1.9, 1.0, 0.0, 0.7],
        ]).encode(),
    }
    frame = parse_pointcloud_message(data)
    assert frame.device_id == "rad_01"
    assert frame.family_id == "1"
    assert frame.room == "bedroom"
    assert frame.ts == 1700000000000
    assert frame.frame_id == "f-0042"
    assert isinstance(frame.points, np.ndarray)
    assert frame.points.shape == (3, 5)
    assert frame.points.dtype == np.float32
    # 验证点坐标
    assert frame.points[0, 0] == 1.0  # x
    assert frame.points[1, 2] == 1.5  # z

def test_parse_vital_message():
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"resp_rate": b"16.5",
        b"heart_rate": b"72.0",
        b"quality": b"0.85",
    }
    frame = parse_vital_message(data)
    assert frame.device_id == "rad_01"
    assert frame.family_id == "1"
    assert frame.resp_rate == 16.5
    assert frame.heart_rate == 72.0
    assert frame.quality == 0.85

def test_parse_vital_message_none_values():
    """部分生命体征字段可能为空"""
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"resp_rate": b"",
        b"heart_rate": b"",
        b"quality": b"0.5",
    }
    frame = parse_vital_message(data)
    assert frame.resp_rate is None
    assert frame.heart_rate is None
    assert frame.quality == 0.5
```

- [ ] **Step 2: Run test (verify failure)**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_redis_client.py -v
```
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `ai/shared/redis_client.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_redis_client.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ai/shared/redis_client.py ai/tests/test_redis_client.py
git commit -m "feat(ai): add Redis Stream consumer with typed frame parsing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

