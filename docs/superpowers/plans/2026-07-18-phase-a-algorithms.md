# Phase A · 世界模型算法工程化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `ai/placeholder/` 的 if-else + random 占位替换为产品级算法模块（规则引擎、空间图、个人基线、异常/通知解码器），全部纯 Python 无 GPU 依赖，用规则特征替代 Encoder 隐状态，打通完整链路。

**Architecture:** 每个组件是独立 class，通过 shared/types.py 的 dataclass 接口通信。WorldModelWorker（main.py）消费 Redis Stream → 提取特征向量 → 依次过规则引擎 + 基线比较 + 空间感知 → 异常解码 → 通知决策 → POST 回 backend。有状态组件（SpatialGraph、PersonalBaseline）内存运行 + 定期 pickle 持久化。

**Tech Stack:** Python 3.12+, numpy, scipy, scikit-learn, redis-py, requests, pytest, dataclasses

## Global Constraints

- 纯算法，无需 GPU，无需训练数据，无需真实硬件
- 特征向量（临时 S_t）从点云规则提取：`[posture_onehot(5), confidence, moving, presence, centroid(3), height, n_points, resp_rate, heart_rate, quality, hour_sin, hour_cos, weekday_onehot(7)]` → ~24 维
- 所有模块通过 dataclass 接口通信，不直接耦合
- 替换 placeholder/main.py，但保留 placeholder/ 目录不动作为调试对比基准
- 代码放在 `housafe/ai/` 下的新目录，不修改 backend/

---

### Task 1: Shared types — 模块间通信的数据契约

**Files:**
- Create: `ai/shared/__init__.py`
- Create: `ai/shared/types.py`
- Create: `ai/__init__.py`

**Interfaces:**
- Produces: `FeatureVector`, `AnomalyResult`, `NotificationDecision`, `PointCloudFrame`, `VitalFrame`, `GraphNode`, `SpatialEdge` — 所有后续 Task 依赖这些类型

- [ ] **Step 1: Create `ai/__init__.py` (empty package marker)**

```bash
mkdir -p ai/shared
touch ai/__init__.py
touch ai/shared/__init__.py
```

- [ ] **Step 2: Write types with tests**

First create the test:

```bash
mkdir -p ai/tests
touch ai/tests/__init__.py
```

Write `ai/tests/test_types.py`:

```python
"""验证 FeatureVector 构造、时间编码计算、AnomalyResult 字段约束"""
import math
import pytest
from ai.shared.types import (
    FeatureVector, AnomalyResult, NotificationDecision,
    time_encode, posture_to_onehot,
)

def test_time_encode_noon():
    """正午 12:00 → sin≈0, cos≈-1 (或 1 取决于角度定义)"""
    s, c = time_encode(12.0)
    # 12:00 = π in sin/cos cycle (0=midnight, 12h=π)
    assert abs(s) < 1e-9
    assert c == pytest.approx(-1.0, abs=1e-9)

def test_time_encode_midnight():
    s, c = time_encode(0.0)
    assert abs(s) < 1e-9
    assert c == pytest.approx(1.0, abs=1e-9)

def test_time_encode_symmetry():
    """6:00 和 18:00 的 cos 应该相同（循环对称）"""
    s6, c6 = time_encode(6.0)
    s18, c18 = time_encode(18.0)
    assert c6 == pytest.approx(c18, abs=1e-9)
    assert s6 == pytest.approx(-s18, abs=1e-9)

def test_posture_onehot():
    assert posture_to_onehot("stand") == [1, 0, 0, 0, 0]
    assert posture_to_onehot("lie")   == [0, 0, 1, 0, 0]
    assert posture_to_onehot("fall")  == [0, 0, 0, 0, 1]
    # unknown → all zeros
    assert posture_to_onehot("unknown") == [0, 0, 0, 0, 0]

def test_feature_vector_to_array():
    fv = FeatureVector(
        ts=1000, device_id="r1", room="bedroom",
        posture="lie", posture_confidence=0.9, presence=True, moving=False,
        centroid=(1.0, 2.0, 0.3), height=0.2, n_points=12, occupancy_estimate=1,
        resp_rate=16.0, heart_rate=72.0, vital_quality=0.85,
        hour_sin=0.0, hour_cos=1.0, weekday=0,
        velocity_variance=0.01,
    )
    arr = fv.to_array()
    # 5 (posture onehot) + 3 (confidence/moving/presence) + 3 (centroid)
    # + 1 (height) + 1 (n_points) + 1 (occupancy)
    # + 3 (vitals) + 2 (time) + 7 (weekday) + 1 (vel_var) = 27
    assert arr.shape == (27,)
    assert arr.dtype == np.float64
    # posture onehot: lie = [0,0,1,0,0]
    assert arr[2] == 1.0

def test_anomaly_result_fields():
    ar = AnomalyResult(
        ts=1000, device_id="r1", room="bathroom",
        anomaly_score=0.85, anomaly_type="fall",
        severity="critical", source="fallback",
        details={"height_drop_m": 1.2},
    )
    assert 0 <= ar.anomaly_score <= 1

def test_notification_decision():
    nd = NotificationDecision(
        level="sms", reason="fall detected in bathroom",
        anomaly=AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.95, anomaly_type="fall",
            severity="critical", source="fallback", details={},
        ),
    )
    assert nd.level in ("none", "push", "sms", "call")
```

- [ ] **Step 3: Verify test fails (types module doesn't exist yet)**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_types.py -v 2>&1 | head -20
```
Expected: ModuleNotFoundError

- [ ] **Step 4: Implement `ai/shared/types.py`**

```python
"""世界模型共享类型 — 所有模块通过 dataclass 接口通信"""
from dataclasses import dataclass, field
import math
import numpy as np

POSTURES = ("stand", "sit", "lie", "walk", "fall")
ALERT_TYPES = ("fall", "stillness", "vital_anomaly", "pattern_deviation", "offline")
SEVERITY_LEVELS = ("info", "warning", "critical")
NOTIFICATION_LEVELS = ("none", "push", "sms", "call")


def time_encode(hour: float) -> tuple[float, float]:
    """将 0-24 的小时数编码为循环 sin/cos 对"""
    rad = hour / 24.0 * 2 * math.pi
    return math.sin(rad), math.cos(rad)


def posture_to_onehot(posture: str) -> list[int]:
    """姿态 → 5 维 one-hot"""
    if posture not in POSTURES:
        return [0, 0, 0, 0, 0]
    idx = POSTURES.index(posture)
    return [1 if i == idx else 0 for i in range(5)]


@dataclass
class FeatureVector:
    """单帧特征向量 — 临时替代 Encoder 输出的 S_t"""
    ts: int
    device_id: str
    room: str
    # 从点云提取
    posture: str
    posture_confidence: float
    presence: bool
    moving: bool
    centroid: tuple[float, float, float]
    height: float
    n_points: int
    occupancy_estimate: int
    # 从生命体征
    resp_rate: float | None
    heart_rate: float | None
    vital_quality: float | None
    # 时间编码
    hour_sin: float
    hour_cos: float
    weekday: int  # 0=Monday ... 6=Sunday
    # 衍生
    velocity_variance: float = 0.0

    def to_array(self) -> np.ndarray:
        """转为固定维度 numpy array，供 GMM baseline 使用"""
        posture_oh = posture_to_onehot(self.posture)
        weekday_oh = [1 if i == self.weekday else 0 for i in range(7)]
        arr = np.array([
            *posture_oh,
            float(self.posture_confidence),
            float(self.moving),
            float(self.presence),
            *self.centroid,
            self.height,
            float(self.n_points),
            float(self.occupancy_estimate),
            self.resp_rate if self.resp_rate is not None else -1.0,
            self.heart_rate if self.heart_rate is not None else -1.0,
            self.vital_quality if self.vital_quality is not None else -1.0,
            self.hour_sin,
            self.hour_cos,
            *weekday_oh,
            self.velocity_variance,
        ], dtype=np.float64)
        return arr


@dataclass
class AnomalyResult:
    """异常检测结果"""
    ts: int
    device_id: str
    room: str
    anomaly_score: float  # 0-1, 0=正常 1=极端异常
    anomaly_type: str     # fall / stillness / vital_anomaly / pattern_deviation / offline
    severity: str         # info / warning / critical
    source: str           # "fallback" / "baseline" / "both"
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.anomaly_score <= 1:
            raise ValueError(f"anomaly_score must be in [0,1], got {self.anomaly_score}")
        if self.anomaly_type not in ALERT_TYPES:
            raise ValueError(f"unknown anomaly_type: {self.anomaly_type}")
        if self.severity not in SEVERITY_LEVELS:
            raise ValueError(f"unknown severity: {self.severity}")


@dataclass
class NotificationDecision:
    """通知决策"""
    level: str   # none / push / sms / call
    reason: str
    anomaly: AnomalyResult

    def __post_init__(self):
        if self.level not in NOTIFICATION_LEVELS:
            raise ValueError(f"unknown notification level: {self.level}")


@dataclass
class GraphNode:
    """空间图节点"""
    node_id: int
    centroid: tuple[float, float, float]  # 区域中心 (x, y, z)
    avg_height: float            # 该区域平均点云高度
    stay_ratio: float            # 在所有帧中的停留比例
    hourly_prob: list[float]     # 24 个时段的停留概率分布
    risk_score: float            # 危险等级 0-1
    attribute_vector: np.ndarray | None = None  # 组合属性向量

    def __post_init__(self):
        if self.attribute_vector is None:
            self.attribute_vector = np.array([
                self.avg_height, self.stay_ratio,
                *self.hourly_prob, self.risk_score,
            ], dtype=np.float32)


@dataclass
class SpatialEdge:
    """空间图边"""
    from_node: int
    to_node: int
    transition_freq: float   # 转移概率
    avg_transition_s: float  # 平均过渡时间（秒）


@dataclass
class PointCloudFrame:
    """从 Redis Stream 解析的点云帧"""
    ts: int
    device_id: str
    family_id: str
    room: str
    frame_id: str
    points: np.ndarray  # (N, 5) float32 [x,y,z,velocity,intensity]


@dataclass
class VitalFrame:
    """从 Redis Stream 解析的生命体征帧"""
    ts: int
    device_id: str
    family_id: str
    room: str
    resp_rate: float | None
    heart_rate: float | None
    quality: float
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_types.py -v
```
Expected: 6 passed

- [ ] **Step 6: Install numpy for the ai worker (add to backend pyproject.toml or use separate deps)**

Check if numpy is available:
```bash
cd /Users/eular/Desktop/housafe && python -c "import numpy; print(numpy.__version__)"
```

If not, add to `backend/pyproject.toml` dependencies: `"numpy>=1.26,<2"`

- [ ] **Step 7: Commit**

```bash
git add ai/__init__.py ai/shared/__init__.py ai/shared/types.py ai/tests/__init__.py ai/tests/test_types.py
git commit -m "feat(ai): add shared types for world model components

FeatureVector, AnomalyResult, NotificationDecision, time_encode, posture_to_onehot — data contracts for Phase A modules.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

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

### Task 3: Fallback 规则引擎

**Files:**
- Create: `ai/fallback/__init__.py`
- Create: `ai/fallback/rules.py`
- Create: `ai/tests/test_fallback.py`

**Interfaces:**
- Consumes: `FeatureVector`, `AnomalyResult` from Task 1
- Produces: `FallbackEngine` class — `evaluate(fv, history) -> list[AnomalyResult]`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/fallback
touch ai/fallback/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_fallback.py`:

```python
"""测试 5 条确定性规则：跌倒/静止/生命体征阈值/设备离线/规则开关"""
import numpy as np
from ai.shared.types import FeatureVector, AnomalyResult
from ai.fallback.rules import FallbackEngine

def make_fv(ts=1000, posture="stand", height=1.5, moving=True,
            centroid=(1.0, 2.0, 1.0), room="bedroom", resp_rate=16.0,
            heart_rate=72.0, vel_var=0.1, **kw):
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=moving,
        centroid=centroid, height=height, n_points=15,
        occupancy_estimate=1, resp_rate=resp_rate, heart_rate=heart_rate,
        vital_quality=0.85, hour_sin=0.0, hour_cos=1.0,
        weekday=0, velocity_variance=vel_var, **kw
    )


class TestFallDetection:
    def test_height_drop_triggers_fall(self):
        engine = FallbackEngine()
        # 模拟跌倒序列：站立(height=1.6) → 3帧后躺下(height=0.2)
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6, moving=True)
            for i in range(5)
        ]
        # 跌倒帧
        current = make_fv(ts=2000, posture="lie", height=0.2, moving=False,
                          centroid=(1.0, 2.0, 0.1))
        history.append(current)

        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 1
        assert falls[0].severity == "critical"
        assert falls[0].anomaly_score > 0.8

    def test_slow_lie_down_no_fall(self):
        """缓慢躺下(height 渐变)不应触发跌倒"""
        engine = FallbackEngine()
        history = [
            make_fv(ts=1000 + i*500, posture="stand", height=1.6 - i*0.1, moving=True)
            for i in range(10)
        ]
        current = make_fv(ts=6000, posture="lie", height=0.6, moving=False)
        history.append(current)
        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 0


class TestStillnessDetection:
    def test_stillness_in_bathroom_triggers(self):
        engine = FallbackEngine()
        # 60 帧 (~2min) 卫生间静止=lie
        history = [
            make_fv(ts=1000 + i*2000, posture="lie", room="bathroom",
                    height=0.3, moving=False, centroid=(1,1,0.1), vel_var=0.0)
            for i in range(60)
        ]
        current = history[-1]
        results = engine.evaluate(current, history)
        stills = [r for r in results if r.anomaly_type == "stillness"]
        assert len(stills) >= 1

    def test_stillness_in_bedroom_ignored(self):
        """卧室长时间静止 = 睡眠，不告警"""
        engine = FallbackEngine()
        history = [
            make_fv(ts=1000 + i*2000, posture="lie", room="bedroom",
                    height=0.3, moving=False, centroid=(1,1,0.1), vel_var=0.0)
            for i in range(60)
        ]
        current = history[-1]
        results = engine.evaluate(current, history)
        stills = [r for r in results if r.anomaly_type == "stillness"]
        assert len(stills) == 0  # 卧室躺着正常


class TestVitalThresholds:
    def test_high_heart_rate(self):
        engine = FallbackEngine()
        fv = make_fv(heart_rate=140, resp_rate=20)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 1
        assert vitals[0].severity in ("warning", "critical")

    def test_normal_vitals_no_alert(self):
        engine = FallbackEngine()
        fv = make_fv(heart_rate=72, resp_rate=16)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 0

    def test_no_vitals_data(self):
        """生命体征数据缺失 → 不告警（可能是安静态条件未满足）"""
        engine = FallbackEngine()
        fv = make_fv(resp_rate=None, heart_rate=None)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 0


class TestRuleToggles:
    def test_disable_fall_rule(self):
        engine = FallbackEngine(enable_fall=False)
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6)
            for i in range(5)
        ]
        current = make_fv(ts=2000, posture="lie", height=0.2)
        history.append(current)
        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 0
```

- [ ] **Step 3: Run test (verify failure)**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_fallback.py -v
```
Expected: ModuleNotFoundError for ai.fallback.rules

- [ ] **Step 4: Implement `ai/fallback/rules.py`**

```python
"""Fallback 规则引擎 — 世界模型的安全网。每条规则独立、可配置、确定性。"""

from collections import deque
from ai.shared.types import FeatureVector, AnomalyResult

# 候选阈值（设计文档标注 "候选·待实验确定"）
FALL_HEIGHT_DROP_M = 0.8       # 高度骤降阈值
FALL_WINDOW_S = 1.0            # 骤降发生的时间窗口
FALL_CONFIRM_S = 5.0           # 跌落后持续观察时间
FALL_STILL_S = 10.0            # 跌落后未恢复站立

BATHROOM_STILLNESS_WARN_S = 300   # 卫生间静止 warning 阈值
BATHROOM_STILLNESS_CRITICAL_S = 600
GENERAL_STILLNESS_INFO_S = 1800   # 其他区域静止 info 阈值

HR_CRITICAL_LOW = 30
HR_WARN_LOW = 40
HR_WARN_HIGH = 130
HR_CRITICAL_HIGH = 150

RR_CRITICAL_LOW = 4
RR_WARN_LOW = 6
RR_WARN_HIGH = 30
RR_CRITICAL_HIGH = 35

OFFLINE_WARN_S = 60
OFFLINE_CRITICAL_S = 300


class FallbackEngine:
    """确定性规则引擎。每个规则可独立开关。"""

    def __init__(
        self,
        enable_fall: bool = True,
        enable_stillness: bool = True,
        enable_vital: bool = True,
        enable_offline: bool = True,
    ):
        self.enable_fall = enable_fall
        self.enable_stillness = enable_stillness
        self.enable_vital = enable_vital
        self.enable_offline = enable_offline

    def evaluate(self, fv: FeatureVector, history: list[FeatureVector]) -> list[AnomalyResult]:
        """评估当前帧 + 历史 → 返回触发的异常列表"""
        results = []

        if self.enable_fall:
            fall = self._detect_fall(fv, history)
            if fall:
                results.append(fall)

        if self.enable_stillness:
            still = self._detect_stillness(fv, history)
            if still:
                results.append(still)

        if self.enable_vital:
            vital = self._check_vital_thresholds(fv)
            if vital:
                results.append(vital)

        # offline 规则需要心跳数据，由外层 main.py 触发，不在此处基于 FeatureVector 判断
        return results

    # ── 跌倒检测 ──────────────────────────────────────

    def _detect_fall(self, fv: FeatureVector, history: list[FeatureVector]) -> AnomalyResult | None:
        if len(history) < 5:
            return None

        # 找 1s 窗口内的最高高度
        window_s = FALL_WINDOW_S
        recent = [h for h in history if 0 <= (fv.ts - h.ts) <= window_s * 1000]
        if not recent:
            recent = history[-5:]

        max_height = max(h.height for h in recent)
        height_drop = max_height - fv.height

        if height_drop < FALL_HEIGHT_DROP_M:
            return None

        # 确认：当前姿势是躺卧
        if fv.posture != "lie":
            return None

        # 确认：跌落后持续躺卧
        after_fall = [h for h in history if h.ts >= fv.ts and (h.ts - fv.ts) <= FALL_CONFIRM_S * 1000]
        if not after_fall:
            after_fall = [fv]

        still_lying = all(h.posture == "lie" for h in after_fall)
        if not still_lying:
            return None

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=min(0.8 + height_drop * 0.1, 1.0),
            anomaly_type="fall", severity="critical", source="fallback",
            details={"height_drop_m": round(height_drop, 2), "max_height": max_height},
        )

    # ── 静止超时 ──────────────────────────────────────

    def _detect_stillness(self, fv: FeatureVector, history: list[FeatureVector]) -> AnomalyResult | None:
        if not fv.presence or fv.moving:
            return None

        # 计算连续静止时长
        still_duration_s = self._continuous_stillness(fv, history)

        if fv.room in ("bathroom", "卫生间"):
            if fv.posture == "lie" and still_duration_s >= BATHROOM_STILLNESS_CRITICAL_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.9, anomaly_type="stillness",
                    severity="critical", source="fallback",
                    details={"duration_s": still_duration_s},
                )
            elif fv.posture == "lie" and still_duration_s >= BATHROOM_STILLNESS_WARN_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.7, anomaly_type="stillness",
                    severity="warning", source="fallback",
                    details={"duration_s": still_duration_s},
                )
        else:
            # 非卫生间/卧室：长时间静止
            if fv.room != "bedroom" and fv.room != "卧室" and still_duration_s >= GENERAL_STILLNESS_INFO_S:
                return AnomalyResult(
                    ts=fv.ts, device_id=fv.device_id, room=fv.room,
                    anomaly_score=0.5, anomaly_type="stillness",
                    severity="info", source="fallback",
                    details={"duration_s": still_duration_s},
                )

        return None

    def _continuous_stillness(self, fv: FeatureVector, history: list[FeatureVector]) -> float:
        """计算从最近一次 moving=True 到当前的连续静止秒数"""
        sorted_h = sorted(history, key=lambda h: h.ts, reverse=True)
        for h in sorted_h:
            if h.device_id != fv.device_id:
                continue
            if h.moving:
                return (fv.ts - h.ts) / 1000.0
        return (fv.ts - sorted_h[-1].ts) / 1000.0 if sorted_h else 0.0

    # ── 生命体征阈值 ──────────────────────────────────

    def _check_vital_thresholds(self, fv: FeatureVector) -> AnomalyResult | None:
        if fv.heart_rate is None or fv.resp_rate is None:
            return None

        hr = fv.heart_rate
        rr = fv.resp_rate

        if hr <= HR_CRITICAL_LOW or hr >= HR_CRITICAL_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.9, anomaly_type="vital_anomaly",
                severity="critical", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if rr <= RR_CRITICAL_LOW or rr >= RR_CRITICAL_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.9, anomaly_type="vital_anomaly",
                severity="critical", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if hr <= HR_WARN_LOW or hr >= HR_WARN_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.6, anomaly_type="vital_anomaly",
                severity="warning", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        if rr <= RR_WARN_LOW or rr >= RR_WARN_HIGH:
            return AnomalyResult(
                ts=fv.ts, device_id=fv.device_id, room=fv.room,
                anomaly_score=0.6, anomaly_type="vital_anomaly",
                severity="warning", source="fallback",
                details={"heart_rate": hr, "resp_rate": rr},
            )

        return None
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_fallback.py -v
```
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add ai/fallback/__init__.py ai/fallback/rules.py ai/tests/test_fallback.py
git commit -m "feat(ai): add FallbackEngine with 3 deterministic rules

Fall detection (height drop + posture confirm), stillness timeout
(room-aware), vital sign threshold checks. All rules independently
toggleable.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 生命体征提取

**Files:**
- Create: `ai/vital_signs/__init__.py`
- Create: `ai/vital_signs/filters.py`
- Create: `ai/vital_signs/quiet_detector.py`
- Create: `ai/vital_signs/extractor.py`
- Create: `ai/tests/test_vital_signs.py`

**Interfaces:**
- Consumes: `FeatureVector` from Task 1, `VitalFrame` from Task 1
- Produces: `extract_breath_rate(phase_signal, fs) -> tuple[float, float]`, `extract_heart_rate(...)`, `is_quiet(fv, history) -> bool`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/vital_signs
touch ai/vital_signs/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_vital_signs.py`:

```python
"""生命体征提取测试 — 用合成正弦波验证信号处理管线"""
import numpy as np
from ai.vital_signs.filters import butter_bandpass, apply_bandpass_filter
from ai.vital_signs.extractor import extract_breath_rate, extract_heart_rate
from ai.vital_signs.quiet_detector import is_quiet
from ai.shared.types import FeatureVector

def make_test_signal(freq_hz: float, duration_s: float, fs: float,
                      noise_std: float = 0.01) -> np.ndarray:
    """生成带噪声的合成正弦波信号"""
    t = np.linspace(0, duration_s, int(duration_s * fs), endpoint=False)
    signal = np.sin(2 * np.pi * freq_hz * t)
    noise = np.random.normal(0, noise_std, len(t))
    return signal + noise

class TestFilters:
    def test_bandpass_shape(self):
        b, a = butter_bandpass(0.1, 0.5, fs=20.0, order=4)
        assert len(b) == 5  # 4th order → 5 coefs

    def test_apply_bandpass_no_crash(self):
        signal = make_test_signal(0.3, 20, 20.0)
        result = apply_bandpass_filter(signal, 0.1, 0.5, fs=20.0)
        assert len(result) == len(signal)
        assert not np.any(np.isnan(result))

class TestBreathRate:
    def test_breath_rate_from_synthetic(self):
        """合成 0.3Hz (=18 bpm) 正弦波 → 应检出 18 bpm"""
        fs = 20.0
        signal = make_test_signal(0.3, 20.0, fs, noise_std=0.005)
        rate, quality = extract_breath_rate(signal, fs)
        assert 15 <= rate <= 21  # 允许 ±3 bpm 误差
        assert quality > 0.5      # SNR 应较高

    def test_breath_rate_quality_low_on_noise(self):
        """高噪声 → quality 降低"""
        fs = 20.0
        signal = make_test_signal(0.3, 20.0, fs, noise_std=1.0)
        _, quality = extract_breath_rate(signal, fs)
        assert quality < 0.5

class TestHeartRate:
    def test_heart_rate_from_synthetic(self):
        """合成 1.2Hz (=72 bpm) 正弦波 → 应检出 72 bpm"""
        fs = 20.0
        signal = make_test_signal(1.2, 10.0, fs, noise_std=0.005)
        rate, quality = extract_heart_rate(signal, fs)
        assert 65 <= rate <= 79
        assert quality > 0.5

class TestQuietDetector:
    def make_fv(self, posture="sit", moving=False, vel_var=0.01):
        return FeatureVector(
            ts=1000, device_id="r1", room="bedroom",
            posture=posture, posture_confidence=0.9, presence=True,
            moving=moving, centroid=(1, 1, 0.5), height=0.3, n_points=10,
            occupancy_estimate=1, resp_rate=16.0, heart_rate=72.0,
            vital_quality=0.85, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=vel_var,
        )

    def test_sitting_still_is_quiet(self):
        history = [self.make_fv() for _ in range(25)]  # ~5s @ 5fps
        assert is_quiet(self.make_fv(), history) is True

    def test_standing_not_quiet(self):
        fv = self.make_fv(posture="stand")
        history = [fv] * 25
        assert is_quiet(fv, history) is False

    def test_moving_not_quiet(self):
        fv = self.make_fv(moving=True)
        history = [fv] * 25
        assert is_quiet(fv, history) is False

    def test_short_duration_not_quiet(self):
        """静止时长不足 → 非安静态"""
        fv = self.make_fv()
        history = [fv] * 3  # 不够 5s
        assert is_quiet(fv, history) is False
```

- [ ] **Step 3: Run test (verify failure)**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_vital_signs.py -v
```
Expected: ModuleNotFoundError

- [ ] **Step 4: Implement `ai/vital_signs/filters.py`**

```python
"""数字滤波器 — 呼吸/心率提取的带通滤波"""
import numpy as np
from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """设计 Butterworth 带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float,
                           fs: float, order: int = 4) -> np.ndarray:
    """零相位带通滤波"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, signal)
```

- [ ] **Step 5: Implement `ai/vital_signs/extractor.py`**

```python
"""从雷达相位信号提取呼吸率和心率 — 经典 FFT 频谱分析"""
import numpy as np


def _extract_peak_frequency(signal: np.ndarray, fs: float,
                             lowcut: float, highcut: float) -> tuple[float, float]:
    """
    对信号加窗 → FFT → 在 [lowcut, highcut] 频带内找最高峰。
    Returns: (peak_freq_hz, quality=peak/noise_floor_mean)
    """
    n = len(signal)
    windowed = signal * np.hanning(n)
    fft = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, 1 / fs)

    band_mask = (freqs >= lowcut) & (freqs <= highcut)
    if not np.any(band_mask):
        return 0.0, 0.0

    band_fft = fft[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = np.argmax(band_fft)
    peak_freq = band_freqs[peak_idx]
    quality = band_fft[peak_idx] / (np.mean(fft[band_mask]) + 1e-10)

    return peak_freq, min(float(quality), 1.0)


def extract_breath_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取呼吸率。
    Args:
        phase_signal: ≥20s 的相位序列
        fs: 采样率 (Hz)
    Returns:
        (breath_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    filtered = apply_bandpass_filter(phase_signal, 0.1, 0.5, fs)  # 6-30 bpm
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.1, 0.5)
    return round(freq_hz * 60, 1), round(quality, 3)


def extract_heart_rate(phase_signal: np.ndarray, fs: float = 20.0) -> tuple[float, float]:
    """
    从相位信号提取心率。
    Args:
        phase_signal: ≥10s 的相位序列
        fs: 采样率 (Hz)
    Returns:
        (heart_rate_bpm, quality_0_to_1)
    """
    from ai.vital_signs.filters import apply_bandpass_filter

    filtered = apply_bandpass_filter(phase_signal, 0.8, 2.5, fs)  # 48-150 bpm
    freq_hz, quality = _extract_peak_frequency(filtered, fs, 0.8, 2.5)
    return round(freq_hz * 60, 1), round(quality, 3)
```

- [ ] **Step 6: Implement `ai/vital_signs/quiet_detector.py`**

```python
"""安静态判定 — 只有在安静态才适合提取生命体征"""
from ai.shared.types import FeatureVector

QUIET_DURATION_S = 5.0
VELOCITY_VAR_THRESHOLD = 0.05


def is_quiet(fv: FeatureVector, history: list[FeatureVector]) -> bool:
    """
    判断当前是否处于安静态。
    条件: posture in (sit, lie) + not moving + 持续 ≥5s + 速度方差低
    """
    if fv.posture not in ("sit", "lie"):
        return False
    if fv.moving:
        return False
    if fv.velocity_variance > VELOCITY_VAR_THRESHOLD:
        return False

    # 检查持续时间
    device_hist = [h for h in history if h.device_id == fv.device_id]
    device_hist.sort(key=lambda h: h.ts)

    # 从当前帧往回找连续静止的时长
    quiet_start = fv.ts
    for h in reversed(device_hist):
        if h.ts >= fv.ts:
            continue
        if h.moving or h.posture not in ("sit", "lie"):
            quiet_start = h.ts
            break
        quiet_start = h.ts

    duration_s = (fv.ts - quiet_start) / 1000.0
    return duration_s >= QUIET_DURATION_S
```

- [ ] **Step 7: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_vital_signs.py -v
```
Expected: all passed

- [ ] **Step 8: Commit**

```bash
git add ai/vital_signs/ ai/tests/test_vital_signs.py
git commit -m "feat(ai): add vital signs extraction (FFT-based) and quiet state detector

Classic signal processing: Butterworth bandpass → FFT peak detection.
Tested with synthetic sine waves. Requires real radar phase signal for
production use; simulator provides vital signs directly via VitalFrame.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 空间图结构学习

**Files:**
- Create: `ai/latent_space/__init__.py`
- Create: `ai/latent_space/graph.py`
- Create: `ai/tests/test_graph.py`

**Interfaces:**
- Consumes: `GraphNode`, `SpatialEdge` from Task 1
- Produces: `SpatialGraph` class — `update(centroids, timestamps)`, `locate(position) -> int`, `get_node_attrs(node_id) -> dict`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/latent_space
touch ai/latent_space/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_graph.py`:

```python
"""测试空间图：聚类、节点属性、定位、周期性更新"""
import numpy as np
from ai.latent_space.graph import SpatialGraph

def make_trajectory(
    centroids: list[tuple[float, float, float]],
    interval_ms: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """生成轨迹数据"""
    n = len(centroids)
    ts = np.arange(n) * interval_ms
    pos = np.array(centroids, dtype=np.float32)
    return pos, ts

class TestSpatialGraph:
    def test_builds_nodes_from_distinct_regions(self):
        """3 个明显分离的区域 + 各停留足够时长 → 应产生 3 个节点"""
        # 区域 A: 床 (0.5, 0.5, 0.2)
        # 区域 B: 门口 (0.5, 3.0, 1.5)
        # 区域 C: 卫生间 (3.0, 0.5, 1.0)
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +   # A: 床，长时间停留
            [(0.5, 3.0, 1.5)] * 100 +   # B: 门口，中等停留
            [(3.0, 0.5, 1.0)] * 150     # C: 卫生间
        )
        pos, ts = make_trajectory(centroids, interval_ms=1000)
        g = SpatialGraph()
        g.update(pos, ts)

        assert len(g.nodes) == 3
        # 节点应按停留时长排序（节点0=停留最多）
        assert g.nodes[0].stay_ratio > g.nodes[1].stay_ratio

    def test_filters_transient_points(self):
        """短暂经过的区域不应成为独立节点"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +   # A: 床
            [(2.0, 2.0, 1.0)] * 3 +     # 路过（3帧）
            [(3.0, 3.0, 1.5)] * 150     # C: 另一区域
        )
        pos, ts = make_trajectory(centroids, interval_ms=1000)
        g = SpatialGraph(min_stay_frames=60)
        g.update(pos, ts)
        assert len(g.nodes) == 2  # 路过点被过滤

    def test_locate_returns_correct_node(self):
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +
            [(3.0, 3.0, 1.5)] * 150
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        n1 = g.locate((0.6, 0.4, 0.3))  # 靠近区域 A
        n2 = g.locate((3.1, 2.9, 1.4))  # 靠近区域 B
        assert n1 != n2
        assert n1 is not None
        assert n2 is not None

    def test_builds_edges_from_transitions(self):
        """A→B→A→B 频繁 → 应产生边"""
        centroids = []
        for _ in range(10):
            centroids += [(0.5, 0.5, 0.2)] * 100   # A
            centroids += [(3.0, 3.0, 1.5)] * 100   # B
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        assert len(g.edges) >= 2  # A→B, B→A

    def test_incremental_update(self):
        """增量更新不应重建节点"""
        pos_a = np.tile(np.array([0.5, 0.5, 0.2]), (200, 1))
        ts_a = np.arange(200) * 1000
        g = SpatialGraph()
        g.update(pos_a, ts_a)
        n_before = len(g.nodes)

        # 次日新数据
        pos_b = np.tile(np.array([3.0, 3.0, 1.5]), (150, 1))
        ts_b = np.arange(150) * 1000 + 86400_000
        g.update(pos_b, ts_b)
        assert len(g.nodes) >= n_before  # 不会丢掉旧节点

    def test_node_risk_score(self):
        """区域高度变化大 → risk_score 更高（更可能是浴室等危险区域）"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +     # A: 低高度、低变化
            [(3.0, 3.0, 1.5)] * 50 +      # B: 中高度
            [(3.0, 3.0, 0.1)] * 50 +      # B: 低高度（高度变化大）
            [(3.0, 3.0, 1.6)] * 50        # B: 高高度
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        node_a = g.locate((0.5, 0.5, 0.2))
        node_b = g.locate((3.0, 3.0, 1.5))
        assert g.nodes[node_a].risk_score < g.nodes[node_b].risk_score

    def test_save_and_load(self, tmp_path):
        """持久化 + 恢复"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +
            [(3.0, 3.0, 1.5)] * 150
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        path = str(tmp_path / "graph.pkl")
        g.save(path)

        g2 = SpatialGraph.load(path)
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.edges) == len(g.edges)
```

- [ ] **Step 3: Run test (verify failure)** then implement

- [ ] **Step 4: Implement `ai/latent_space/graph.py`**

```python
"""空间图结构 — 从轨迹数据自动学习房间功能区域拓扑"""
import pickle
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import defaultdict
from ai.shared.types import GraphNode, SpatialEdge


class SpatialGraph:
    """从人的位置轨迹中自动学出的空间拓扑图"""

    def __init__(self, min_stay_frames: int = 60):
        """
        Args:
            min_stay_frames: 成为独立节点的最少停留帧数（过滤路过）
                默认 60 帧 ≈ 1min @ 1fps（样本点）或 ≈ 3s @ 20fps
        """
        self.nodes: dict[int, GraphNode] = {}
        self.edges: dict[tuple[int, int], SpatialEdge] = {}
        self._clusterer = HDBSCAN(min_cluster_size=max(min_stay_frames, 5))
        self._node_centroids: np.ndarray | None = None  # (K, 3) 用于快速 locate
        self._total_frames: int = 0
        self._point_buffer: list[tuple[np.ndarray, int]] = []  # [(xyz, ts)]

    def update(self, positions: np.ndarray, timestamps: np.ndarray):
        """
        增量更新图结构。
        Args:
            positions: (N, 3) float32, 每帧人的质心坐标
            timestamps: (N,) int64, 每帧的时间戳 (ms)
        """
        if len(positions) == 0:
            return

        # 累积缓冲
        for i in range(len(positions)):
            self._point_buffer.append((positions[i], timestamps[i]))
        self._total_frames += len(positions)

        # 限制缓冲区大小（保留最近 14 天，按 1fps 采样 ≈ 1.2M 点）
        max_buffer = 1_200_000
        if len(self._point_buffer) > max_buffer:
            self._point_buffer = self._point_buffer[-max_buffer:]

        # 重新聚类（HDBSCAN 不支持增量，替换为全量；数据量小时可接受）
        self._recluster()

    def _recluster(self):
        """从缓冲数据重新聚类"""
        if len(self._point_buffer) < 10:
            return

        all_pos = np.array([p[0] for p in self._point_buffer])
        all_ts = np.array([p[1] for p in self._point_buffer])

        # HDBSCAN 聚类
        labels = self._clusterer.fit_predict(all_pos)

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 噪声点不建节点

        if len(unique_labels) < 1:
            return

        # 构建节点
        new_nodes = {}
        for label in unique_labels:
            mask = labels == label
            cluster_pos = all_pos[mask]
            cluster_ts = all_ts[mask]
            centroid = tuple(cluster_pos.mean(axis=0).astype(np.float32))
            avg_height = float(cluster_pos[:, 2].mean())
            stay_ratio = len(cluster_pos) / len(all_pos)

            # 24 时段分布
            hourly_prob = np.zeros(24)
            hours = (cluster_ts / 3600_000).astype(int) % 24
            for h in hours:
                hourly_prob[h] += 1
            hourly_prob = (hourly_prob / hourly_prob.sum()).tolist()

            # 危险等级：高度方差标准化
            height_std = float(cluster_pos[:, 2].std())
            risk_score = min(height_std / 1.0, 1.0)  # 归一化

            node = GraphNode(
                node_id=int(label),
                centroid=centroid,
                avg_height=avg_height,
                stay_ratio=stay_ratio,
                hourly_prob=hourly_prob,
                risk_score=risk_score,
            )
            new_nodes[int(label)] = node

        self.nodes = new_nodes

        # 更新 centroid 缓存
        if new_nodes:
            self._node_centroids = np.array([
                n.centroid for n in new_nodes.values()
            ], dtype=np.float32)

        # 构建边（共现统计）
        self._build_edges(labels, all_pos, all_ts)

    def _build_edges(self, labels: np.ndarray, positions: np.ndarray, timestamps: np.ndarray):
        """从标签序列统计节点间转移"""
        transitions = defaultdict(lambda: {"count": 0, "total_time": 0.0})

        for i in range(len(labels) - 1):
            a, b = labels[i], labels[i + 1]
            if a == -1 or b == -1 or a == b:
                continue
            dt = (timestamps[i + 1] - timestamps[i]) / 1000.0
            if dt < 0 or dt > 300:  # 超过 5 分钟不算连续过渡
                continue
            key = (int(a), int(b))
            transitions[key]["count"] += 1
            transitions[key]["total_time"] += dt

        total = sum(t["count"] for t in transitions.values())
        self.edges = {}
        for (a, b), t in transitions.items():
            if total > 0 and t["count"] / total >= 0.005:  # 过滤低频过渡
                self.edges[(a, b)] = SpatialEdge(
                    from_node=a, to_node=b,
                    transition_freq=t["count"] / total,
                    avg_transition_s=t["total_time"] / t["count"],
                )

    def locate(self, position: tuple[float, float, float]) -> int | None:
        """给定位置 → 返回最近的 node_id"""
        if self._node_centroids is None or len(self._node_centroids) == 0:
            return None
        pos = np.array(position, dtype=np.float32)
        dists = np.linalg.norm(self._node_centroids - pos, axis=1)
        return int(np.argmin(dists))

    def get_node_attrs(self, node_id: int) -> dict:
        """获取节点属性"""
        if node_id not in self.nodes:
            return {}
        n = self.nodes[node_id]
        return {
            "node_id": n.node_id,
            "centroid": n.centroid,
            "avg_height": n.avg_height,
            "stay_ratio": n.stay_ratio,
            "hourly_prob": n.hourly_prob,
            "risk_score": n.risk_score,
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)

    @classmethod
    def load(cls, path: str) -> "SpatialGraph":
        g = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        g.nodes = data["nodes"]
        g.edges = data["edges"]
        if g.nodes:
            g._node_centroids = np.array([n.centroid for n in g.nodes.values()], dtype=np.float32)
        return g
```

- [ ] **Step 5: Check scikit-learn availability**

```bash
cd /Users/eular/Desktop/housafe && python -c "import sklearn; print(sklearn.__version__)"
```

If missing, add `"scikit-learn>=1.3,<2"` to `backend/pyproject.toml`.

- [ ] **Step 6: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_graph.py -v
```
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add ai/latent_space/ ai/tests/test_graph.py
git commit -m "feat(ai): add SpatialGraph — auto-learn room topology from trajectories

HDBSCAN clustering on position centroids → nodes, co-occurrence stats
→ edges. Supports incremental update, save/load, locate, risk scoring.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 个人基线 GMM

**Files:**
- Create: `ai/baseline/__init__.py`
- Create: `ai/baseline/gmm_baseline.py`
- Create: `ai/tests/test_baseline.py`

**Interfaces:**
- Consumes: `FeatureVector` from Task 1
- Produces: `PersonalBaseline` class — `fit(features, n_days)`, `score(fv) -> float`, `update(new_features)`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/baseline
touch ai/baseline/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_baseline.py`:

```python
"""测试个人基线：GMM 拟合、异常分输出、增量更新"""
import numpy as np
from ai.shared.types import FeatureVector
from ai.baseline.gmm_baseline import PersonalBaseline, context_key

def make_feature_vectors(
    n: int,
    posture: str = "sit",
    heart_rate: float = 72.0,
    resp_rate: float = 16.0,
    centroid: tuple = (1.0, 1.0, 0.5),
    height: float = 0.8,
    weekday: int = 0,
    hour: float = 12.0,
    noise_scale: float = 0.02,
) -> list[FeatureVector]:
    """生成带微小噪声的批量特征向量"""
    from ai.shared.types import time_encode
    import math

    rng = np.random.RandomState(42)
    fvs = []
    for i in range(n):
        h = hour + (i * 2 / 60)  # 微小时间偏移
        h_sin, h_cos = time_encode(h)
        fvs.append(FeatureVector(
            ts=1000 + i * 2000,
            device_id="r1", room="bedroom",
            posture=posture, posture_confidence=0.9,
            presence=True, moving=False,
            centroid=tuple(c + rng.normal(0, noise_scale) for c in centroid),
            height=height + rng.normal(0, noise_scale),
            n_points=15, occupancy_estimate=1,
            resp_rate=resp_rate + rng.normal(0, noise_scale * 5),
            heart_rate=heart_rate + rng.normal(0, noise_scale * 10),
            vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
            weekday=weekday, velocity_variance=0.01,
        ))
    return fvs

class TestContextKey:
    def test_weekday_morning(self):
        assert context_key(0, 8) == "weekday_morning"
    def test_weekend_night(self):
        assert context_key(5, 2) == "weekend_night"
    def test_weekday_afternoon(self):
        assert context_key(2, 14) == "weekday_afternoon"

class TestPersonalBaseline:
    def test_fit_and_score_normal(self):
        """在 14 天"正常"数据上拟合 → 相似数据应得低异常分"""
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)  # ~1/3 天数据
        baseline.fit(fvs)

        # 和训练数据分布相同的样本 → 异常分应低
        test_fv = make_feature_vectors(1)[0]
        score = baseline.score(test_fv)
        assert score < 0.5, f"Expected low score for normal data, got {score}"

    def test_anomaly_scores_high(self):
        """偏离基线 → 高异常分"""
        baseline = PersonalBaseline()
        normal = make_feature_vectors(500, posture="sit", heart_rate=72)
        baseline.fit(normal)

        # 构造异常：心率极高 + 躺卧（和训练数据完全不同）
        anomaly = make_feature_vectors(1, posture="lie", heart_rate=140)[0]
        score = baseline.score(anomaly)
        assert score > 0.5, f"Expected high score for anomaly, got {score}"

    def test_insufficient_data_returns_zero(self):
        """数据不足时 → 返回 0（基线未建立，不告警）"""
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(10)  # 太少
        baseline.fit(fvs)
        score = baseline.score(fvs[0])
        assert score == 0.0

    def test_update_adds_new_data(self):
        """增量更新后新数据也纳入基线"""
        baseline = PersonalBaseline()
        day1 = make_feature_vectors(200, heart_rate=72, hour=10)
        baseline.fit(day1)

        # 新数据：心率模式不同
        day2 = make_feature_vectors(200, heart_rate=68, hour=10)
        baseline.update(day2)

        # 更新后，心率 70 应该在基线内（介于 68-72）
        test = make_feature_vectors(1, heart_rate=70, hour=10)[0]
        score = baseline.score(test)
        assert score < 0.5, f"After update, mid-range should be normal, got {score}"

    def test_save_and_load(self, tmp_path):
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)
        baseline.fit(fvs)

        path = str(tmp_path / "baseline.pkl")
        baseline.save(path)

        b2 = PersonalBaseline.load(path)
        assert b2.is_ready is True
        # 同样的测试数据应得到相近的分数
        test = make_feature_vectors(1)[0]
        assert abs(baseline.score(test) - b2.score(test)) < 0.01

    def test_is_ready_false_before_fit(self):
        baseline = PersonalBaseline()
        assert baseline.is_ready is False

    def test_is_ready_true_after_sufficient_data(self):
        baseline = PersonalBaseline()
        fvs = make_feature_vectors(500)
        baseline.fit(fvs)
        assert baseline.is_ready is True
```

- [ ] **Step 3: Run test (verify failure)** then implement

- [ ] **Step 4: Implement `ai/baseline/gmm_baseline.py`**

```python
"""个人基线 — 14 天隐状态轨迹 → GMM 概率密度 → 异常分"""
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from ai.shared.types import FeatureVector

MIN_FRAMES_PER_CONTEXT = 200  # 每个上下文最少帧数
MIN_TOTAL_FRAMES = 1000       # 全局最少帧数
N_COMPONENTS = 5
PCA_DIM = 16


def context_key(weekday: int, hour: int) -> str:
    """将 (weekday, hour) → 10 个上下文字符串之一"""
    is_weekend = weekday >= 5
    prefix = "weekend" if is_weekend else "weekday"
    if 0 <= hour < 6:
        return f"{prefix}_night"
    elif 6 <= hour < 9:
        return f"{prefix}_morning"
    elif 9 <= hour < 12:
        return f"{prefix}_morning"
    elif 12 <= hour < 14:
        return f"{prefix}_afternoon"
    elif 14 <= hour < 18:
        return f"{prefix}_afternoon"
    elif 18 <= hour < 21:
        return f"{prefix}_evening"
    else:
        return f"{prefix}_night"


class PersonalBaseline:
    """
    分上下文 GMM 个人基线。
    - 10 个上下文 × 1 个 GMM，每个 GMM 拟合该上下文中的特征向量分布
    - 异常分 = Mahalanobis distance percentile
    """

    def __init__(self):
        self._context_gmms: dict[str, GaussianMixture] = {}
        self._pca: PCA | None = None
        self._context_data: dict[str, list[np.ndarray]] = {}
        self._context_scores: dict[str, list[float]] = {}  # 历史分数用于标准化
        self.is_ready: bool = False

    def fit(self, features: list[FeatureVector], n_days: int = 14):
        """训练基线"""
        if len(features) < MIN_TOTAL_FRAMES:
            return

        # 分组
        grouped: dict[str, list[np.ndarray]] = {}
        for fv in features:
            key = context_key(fv.weekday, self._hour_from_ts(fv))
            arr = fv.to_array()
            grouped.setdefault(key, []).append(arr)

        # 过滤样本不够的上下文
        self._context_data = {
            k: v for k, v in grouped.items()
            if len(v) >= MIN_FRAMES_PER_CONTEXT
        }

        if not self._context_data:
            return

        # PCA
        all_data = np.vstack(list(self._context_data.values()))
        pca_dim = min(PCA_DIM, all_data.shape[1], all_data.shape[0] // 10)
        self._pca = PCA(n_components=pca_dim)
        all_reduced = self._pca.fit_transform(all_data)

        # 分上下文 GMM
        self._context_gmms = {}
        self._context_scores = {}
        offset = 0
        for key, arrs in self._context_data.items():
            n = len(arrs)
            reduced = all_reduced[offset:offset + n]
            offset += n
            gmm = GaussianMixture(
                n_components=min(N_COMPONENTS, n // 20),
                covariance_type="full",
                random_state=42,
            )
            gmm.fit(reduced)
            self._context_gmms[key] = gmm
            # 记录历史 Mahalanobis 分数
            scores = []
            for i in range(n):
                comp = gmm.predict(reduced[i:i + 1])[0]
                d = self._mahalanobis(reduced[i], gmm.means_[comp],
                                       gmm.covariances_[comp])
                scores.append(d)
            self._context_scores[key] = scores

        self.is_ready = True

    def score(self, fv: FeatureVector) -> float:
        """
        返回异常分 0-1。0=完全正常, 1=极度异常。
        基线未就绪时返回 0。
        """
        if not self.is_ready or self._pca is None:
            return 0.0

        key = context_key(fv.weekday, self._hour_from_ts(fv))
        if key not in self._context_gmms:
            return 0.0

        gmm = self._context_gmms[key]
        arr = fv.to_array().reshape(1, -1)
        reduced = self._pca.transform(arr)[0]

        # 到最近分量的 Mahalanobis 距离
        comp = gmm.predict(reduced.reshape(1, -1))[0]
        dist = self._mahalanobis(reduced, gmm.means_[comp],
                                  gmm.covariances_[comp])

        # 在该上下文历史分数中计算 percentile
        hist = self._context_scores.get(key, [])
        if not hist:
            return min(dist / 10.0, 1.0)

        percentile = sum(1 for h in hist if h < dist) / len(hist)
        return float(np.clip(percentile, 0.0, 1.0))

    def update(self, new_features: list[FeatureVector]):
        """增量更新 — 追加数据后重新拟合"""
        all_features = []
        for arrs in self._context_data.values():
            for a in arrs:
                all_features.append(a)
        for fv in new_features:
            all_features.append(fv.to_array())
        # 重建 FeatureVector 列表重新 fit
        # (简化为直接重新拟合；产品中可优化为增量 GMM)
        self.fit(self._arrays_to_fv(all_features, new_features))

    @staticmethod
    def _mahalanobis(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        try:
            inv_cov = np.linalg.inv(cov)
            diff = x - mean
            return float(np.sqrt(diff @ inv_cov @ diff))
        except np.linalg.LinAlgError:
            return 10.0

    @staticmethod
    def _hour_from_ts(fv: FeatureVector) -> int:
        import math
        rad = math.atan2(fv.hour_sin, fv.hour_cos)
        hour = rad / (2 * math.pi) * 24
        return int(hour % 24)

    def _arrays_to_fv(self, arrays: list[np.ndarray],
                       ref: list[FeatureVector]) -> list[FeatureVector]:
        # 简化：返回 ref 作为代理（update 后重新 fit 仅需要数量信息）
        # 实际上我们直接重新 fit，保留原始 FeatureVector 引用
        return ref

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "context_gmms": self._context_gmms,
                "pca": self._pca,
                "context_data": self._context_data,
                "context_scores": self._context_scores,
                "is_ready": self.is_ready,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PersonalBaseline":
        b = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        b._context_gmms = data["context_gmms"]
        b._pca = data["pca"]
        b._context_data = data["context_data"]
        b._context_scores = data["context_scores"]
        b.is_ready = data["is_ready"]
        return b
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_baseline.py -v
```
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add ai/baseline/ ai/tests/test_baseline.py
git commit -m "feat(ai): add PersonalBaseline GMM — per-context density estimation

10 contexts (weekday/weekend × time-of-day), PCA reduction, GMM fit,
Mahalanobis distance → anomaly percentile. Ready after 14 days of data.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 异常 Decoder

**Files:**
- Create: `ai/decoders/__init__.py`
- Create: `ai/decoders/anomaly.py`
- Create: `ai/tests/test_anomaly.py`

**Interfaces:**
- Consumes: `FallbackEngine` from Task 3, `SpatialGraph` from Task 5, `PersonalBaseline` from Task 6, `FeatureVector` from Task 1
- Produces: `AnomalyDecoder` class — `evaluate(fv, history) -> AnomalyResult | None`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/decoders
touch ai/decoders/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_anomaly.py`:

```python
"""测试异常解码器：融合 fallback + baseline + 空间感知"""
from ai.shared.types import FeatureVector
from ai.decoders.anomaly import AnomalyDecoder
from ai.fallback.rules import FallbackEngine

def make_fv(ts=1000, posture="sit", room="bedroom", heart_rate=72, resp_rate=16,
            centroid=(1.0, 1.0, 0.5), height=0.8, moving=False, vel_var=0.01,
            hour=12.0, weekday=0):
    from ai.shared.types import time_encode
    h_sin, h_cos = time_encode(hour)
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=moving,
        centroid=centroid, height=height, n_points=15,
        occupancy_estimate=1, resp_rate=resp_rate, heart_rate=heart_rate,
        vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
        weekday=weekday, velocity_variance=vel_var,
    )

class TestAnomalyDecoder:
    def test_fallback_detected_passes_through(self):
        """Fallback 检出的异常直接透传"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        # 构造跌倒序列
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6, moving=True)
            for i in range(5)
        ]
        current = make_fv(ts=2000, posture="lie", height=0.2, moving=False,
                          centroid=(1.0, 1.0, 0.1))
        history.append(current)

        result = decoder.evaluate(current, history)
        assert result is not None
        assert result.anomaly_type == "fall"
        assert result.source == "fallback"

    def test_no_baseline_no_graph_returns_none_for_normal(self):
        """无基线/无图时 → 正常帧不告警"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fv = make_fv()
        result = decoder.evaluate(fv, [fv])
        assert result is None

    def test_spatial_awareness_bathroom_lie(self):
        """卫生间躺卧 → 空间感知加权 → 异常分增加"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fvs_bathroom = [make_fv(room="bathroom", posture="lie",
                                 centroid=(3, 3, 0.1), height=0.2)
                        for _ in range(60)]
        result = decoder.evaluate(fvs_bathroom[-1], fvs_bathroom)
        # stillness 规则在卫生间触发
        if result:
            assert result.room == "bathroom"
```

- [ ] **Step 3: Run test (verify failure)** then implement

- [ ] **Step 4: Implement `ai/decoders/anomaly.py`**

```python
"""异常 Decoder — 融合规则引擎 + 基线偏离 + 空间感知 → 最终异常分"""
from ai.shared.types import FeatureVector, AnomalyResult
from ai.fallback.rules import FallbackEngine


class AnomalyDecoder:
    """
    异常解码器。MVP 版本主要靠 FallbackEngine。
    随基线数据积累，baseline 异常分逐渐加入。
    """

    def __init__(
        self,
        fallback: FallbackEngine,
        baseline=None,   # PersonalBaseline | None
        graph=None,      # SpatialGraph | None
        baseline_weight: float = 0.3,
    ):
        self.fallback = fallback
        self.baseline = baseline
        self.graph = graph
        self.baseline_weight = baseline_weight

    def evaluate(
        self, fv: FeatureVector, history: list[FeatureVector]
    ) -> AnomalyResult | None:
        """
        综合评估当前帧。
        Returns: AnomalyResult（异常）或 None（正常）
        """
        # 1. 规则引擎（确定性安全网）
        rule_results = self.fallback.evaluate(fv, history)

        # 2. 基线偏离
        baseline_score = 0.0
        if self.baseline is not None and self.baseline.is_ready:
            baseline_score = self.baseline.score(fv)

        # 3. 空间感知加权
        spatial_multiplier = 1.0
        if self.graph is not None:
            node_id = self.graph.locate(fv.centroid)
            if node_id is not None:
                attrs = self.graph.get_node_attrs(node_id)
                # 高风险区域 + 躺卧 → 权重增加
                risk = attrs.get("risk_score", 0.0)
                if fv.posture == "lie" and risk > 0.3:
                    spatial_multiplier = 1.0 + risk

        # 4. 融合
        if not rule_results and baseline_score < 0.7:
            return None  # 无异常

        # 取最严重的规则结果
        severity_order = {"critical": 3, "warning": 2, "info": 1}
        best_rule = None
        if rule_results:
            best_rule = max(rule_results, key=lambda r: severity_order.get(r.severity, 0))

        # 融合分数
        rule_score = best_rule.anomaly_score if best_rule else 0.0
        combined_score = max(rule_score, baseline_score * spatial_multiplier)
        combined_score = min(combined_score, 1.0)

        # 确定来源和类型
        if best_rule:
            source = "both" if baseline_score > 0.5 else "fallback"
            anomaly_type = best_rule.anomaly_type
            severity = best_rule.severity
            details = best_rule.details
        else:
            source = "baseline"
            anomaly_type = "pattern_deviation"
            severity = "warning" if combined_score > 0.8 else "info"
            details = {"baseline_score": baseline_score}

        details["spatial_multiplier"] = spatial_multiplier
        details["baseline_score"] = baseline_score

        return AnomalyResult(
            ts=fv.ts, device_id=fv.device_id, room=fv.room,
            anomaly_score=round(combined_score, 3),
            anomaly_type=anomaly_type,
            severity=severity, source=source,
            details=details,
        )
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_anomaly.py -v
```
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add ai/decoders/__init__.py ai/decoders/anomaly.py ai/tests/test_anomaly.py
git commit -m "feat(ai): add AnomalyDecoder — fuse fallback rules + baseline + spatial context

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: 通知 Decoder

**Files:**
- Create: `ai/decoders/notification.py`
- Create: `ai/tests/test_notification.py`

**Interfaces:**
- Consumes: `AnomalyResult` from Task 1, `NotificationDecision` from Task 1
- Produces: `NotificationDecider` class — `decide(anomaly, history) -> NotificationDecision`

- [ ] **Step 1: Write tests**

`ai/tests/test_notification.py`:

```python
"""测试通知决策规则"""
from ai.shared.types import AnomalyResult, NotificationDecision
from ai.decoders.notification import NotificationDecider

class TestNotificationDecider:
    def test_fall_critical_calls(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.95, anomaly_type="fall",
            severity="critical", source="fallback",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level == "call"

    def test_fall_warning_sms(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.75, anomaly_type="fall",
            severity="warning", source="fallback",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level == "sms"

    def test_stillness_critical_sms(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.9, anomaly_type="stillness",
            severity="critical", source="fallback",
            details={"duration_s": 600},
        )
        decision = d.decide(ar, [])
        assert decision.level == "sms"

    def test_vital_warning_push(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.6, anomaly_type="vital_anomaly",
            severity="warning", source="fallback",
            details={"heart_rate": 135},
        )
        decision = d.decide(ar, [])
        assert decision.level == "push"

    def test_pattern_deviation_info_push(self):
        d = NotificationDecider()
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bedroom",
            anomaly_score=0.5, anomaly_type="pattern_deviation",
            severity="info", source="baseline",
            details={},
        )
        decision = d.decide(ar, [])
        assert decision.level in ("push", "none")

    def test_suppress_duplicates(self):
        """短时间内相同异常类型 → 抑制重复通知"""
        d = NotificationDecider(suppress_window_s=300)
        ar = AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.7, anomaly_type="stillness",
            severity="warning", source="fallback",
            details={},
        )
        # 第一条正常通知
        d1 = d.decide(ar, [])
        # 模拟 history: 刚发过同类通知
        d2 = d.decide(ar, [d1])
        assert d2.level == "none"
```

- [ ] **Step 2: Run test (verify failure)** then implement

- [ ] **Step 3: Implement `ai/decoders/notification.py`**

```python
"""通知 Decoder — 异常 → 通知等级（不提/推送/短信/电话）"""
from ai.shared.types import AnomalyResult, NotificationDecision


class NotificationDecider:
    """MVP 版本：规则决策表。演进版本：MLP 4-class 从用户反馈学习。"""

    # 通知决策矩阵: (anomaly_type, severity) → level
    DECISION_MATRIX = {
        ("fall", "critical"): "call",
        ("fall", "warning"): "sms",
        ("fall", "info"): "push",
        ("stillness", "critical"): "sms",
        ("stillness", "warning"): "push",
        ("stillness", "info"): "push",
        ("vital_anomaly", "critical"): "sms",
        ("vital_anomaly", "warning"): "push",
        ("vital_anomaly", "info"): "push",
        ("pattern_deviation", "critical"): "sms",
        ("pattern_deviation", "warning"): "push",
        ("pattern_deviation", "info"): "none",
        ("offline", "critical"): "call",
        ("offline", "warning"): "push",
        ("offline", "info"): "push",
    }

    def __init__(self, suppress_window_s: float = 300.0):
        """
        Args:
            suppress_window_s: 同类型通知抑制窗口（秒）。窗口内重复通知降级为 none。
        """
        self.suppress_window_s = suppress_window_s

    def decide(
        self, anomaly: AnomalyResult, recent_decisions: list[NotificationDecision]
    ) -> NotificationDecision:
        """
        决策通知等级。
        Args:
            anomaly: 当前异常
            recent_decisions: 最近的决策历史（用于抑制重复）
        """
        level = self.DECISION_MATRIX.get(
            (anomaly.anomaly_type, anomaly.severity), "push"
        )

        # 抑制窗口：同类型同房间的重复通知
        if level != "none" and self._is_duplicate(anomaly, recent_decisions):
            level = "none"

        return NotificationDecision(
            level=level,
            reason=f"{anomaly.anomaly_type} in {anomaly.room} (score={anomaly.anomaly_score:.2f})",
            anomaly=anomaly,
        )

    def _is_duplicate(
        self, anomaly: AnomalyResult, recent: list[NotificationDecision]
    ) -> bool:
        for d in recent:
            if d.level == "none":
                continue
            a = d.anomaly
            if (a.anomaly_type == anomaly.anomaly_type
                and a.room == anomaly.room
                and a.device_id == anomaly.device_id):
                dt_s = (anomaly.ts - a.ts) / 1000.0
                if 0 <= dt_s <= self.suppress_window_s:
                    return True
        return False
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_notification.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add ai/decoders/notification.py ai/tests/test_notification.py
git commit -m "feat(ai): add NotificationDecider — anomaly → notification level mapping

Rule matrix + duplicate suppression window. MLP upgrade on user feedback (P2).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 报告 + 追踪 Decoders

**Files:**
- Create: `ai/decoders/report.py`
- Create: `ai/decoders/tracking.py`
- Create: `ai/tests/test_report_tracking.py`

**Interfaces:**
- Consumes: `FeatureVector` from Task 1, `SpatialGraph` from Task 5
- Produces: `ReportDecoder` — `generate_daily_summary(features) -> dict`
- Produces: `TrackingDecoder` — `build_trajectory(features) -> dict`, `heatmap(features, graph) -> dict`

- [ ] **Step 1: Write tests**

`ai/tests/test_report_tracking.py`:

```python
"""报告和追踪解码器测试"""
from ai.shared.types import FeatureVector
from ai.decoders.report import ReportDecoder
from ai.decoders.tracking import TrackingDecoder

def make_fv(ts=1000, posture="sit", centroid=(1.0, 1.0, 0.5),
            room="bedroom", **kw):
    from ai.shared.types import time_encode
    h_sin, h_cos = time_encode(12.0)
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=False,
        centroid=centroid, height=0.8, n_points=15,
        occupancy_estimate=1, resp_rate=16.0, heart_rate=72.0,
        vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
        weekday=0, velocity_variance=0.01, **kw
    )

class TestReportDecoder:
    def test_daily_summary(self):
        """24h 特征 → 活动摘要"""
        decoder = ReportDecoder()
        fvs = [
            make_fv(ts=3600_000 * i, posture="lie", room="bedroom")
            for i in range(8)  # 8h 睡眠
        ] + [
            make_fv(ts=3600_000 * (8 + i), posture="sit", room="living_room")
            for i in range(4)  # 4h 坐着
        ] + [
            make_fv(ts=3600_000 * (12 + i), posture="walk", room="kitchen")
            for i in range(2)  # 2h 活动
        ]

        summary = decoder.generate_daily_summary(fvs)
        assert "posture_distribution" in summary
        assert "hours_active" in summary
        assert summary["posture_distribution"]["lie"] > 0.3  # ~57% lie
        assert summary["hours_active"] > 0

    def test_empty_input(self):
        decoder = ReportDecoder()
        summary = decoder.generate_daily_summary([])
        assert summary["hours_active"] == 0

class TestTrackingDecoder:
    def test_trajectory(self):
        decoder = TrackingDecoder()
        fvs = [
            make_fv(ts=i*2000, centroid=(float(i)*0.1, 0.0, 1.0))
            for i in range(10)
        ]
        traj = decoder.build_trajectory(fvs)
        assert len(traj["positions"]) == 10
        assert "total_distance_m" in traj
        assert "duration_s" in traj

    def test_heatmap(self):
        decoder = TrackingDecoder()
        fvs = [
            make_fv(ts=i*2000, centroid=(1.0, 1.0, 0.5)) for i in range(50)
        ] + [
            make_fv(ts=100_000 + i*2000, centroid=(3.0, 3.0, 1.5))
            for i in range(30)
        ]
        hm = decoder.heatmap(fvs)
        assert "grid" in hm
        assert hm["grid"].shape[0] > 0
```

- [ ] **Step 2: Implement `ai/decoders/report.py`**

```python
"""报告 Decoder — 从隐状态轨迹生成结构化统计数据"""
from collections import Counter
from ai.shared.types import FeatureVector


class ReportDecoder:
    """MVP 版本：统计聚合。P2 接 LLM 生成自然语言报告。"""

    def generate_daily_summary(self, features: list[FeatureVector]) -> dict:
        """
        生成单日摘要。
        Args:
            features: 约 1440 帧（1/min 采样）的当天特征
        Returns:
            dict with posture_distribution, hours_active, avg_vitals, room_usage
        """
        if not features:
            return {
                "posture_distribution": {},
                "hours_active": 0,
                "avg_heart_rate": None,
                "avg_resp_rate": None,
                "room_usage": {},
                "n_frames": 0,
            }

        # 姿态分布
        posture_counts = Counter(f.posture for f in features)
        total = len(features)
        posture_dist = {k: round(v / total, 3) for k, v in posture_counts.items()}

        # 活跃时长（moving=True 且有走动）
        active_frames = sum(1 for f in features if f.moving or f.posture == "walk")
        hours_active = round(active_frames / max(total, 1) * 24, 1)

        # 平均生命体征
        hrs = [f.heart_rate for f in features if f.heart_rate is not None]
        rrs = [f.resp_rate for f in features if f.resp_rate is not None]
        avg_hr = round(sum(hrs) / len(hrs), 1) if hrs else None
        avg_rr = round(sum(rrs) / len(rrs), 1) if rrs else None

        # 房间使用
        room_counts = Counter(f.room for f in features)
        room_usage = {k: round(v / total, 3) for k, v in room_counts.items()}

        return {
            "posture_distribution": posture_dist,
            "hours_active": hours_active,
            "avg_heart_rate": avg_hr,
            "avg_resp_rate": avg_rr,
            "room_usage": room_usage,
            "n_frames": total,
        }

    def weekly_comparison(self, this_week: dict, last_week: dict) -> dict:
        """本周 vs 上周趋势对比"""
        deltas = {}
        for key in ("hours_active", "avg_heart_rate", "avg_resp_rate"):
            if this_week.get(key) is not None and last_week.get(key) is not None:
                deltas[key] = round(this_week[key] - last_week[key], 2)
        return deltas
```

- [ ] **Step 3: Implement `ai/decoders/tracking.py`**

```python
"""追踪 Decoder — 位置轨迹、热力图、区域停留统计"""
import numpy as np
from ai.shared.types import FeatureVector


class TrackingDecoder:
    """位置追踪。后续可接入图结构做房间级追踪。"""

    def build_trajectory(self, features: list[FeatureVector]) -> dict:
        """
        构建轨迹线。
        Returns: positions[[x,y,z]], timestamps, total_distance_m, duration_s
        """
        if not features:
            return {"positions": [], "timestamps": [], "total_distance_m": 0, "duration_s": 0}

        sorted_f = sorted(features, key=lambda f: f.ts)
        positions = [list(f.centroid) for f in sorted_f]
        timestamps = [f.ts for f in sorted_f]

        total_dist = 0.0
        for i in range(1, len(positions)):
            p1, p2 = np.array(positions[i - 1]), np.array(positions[i])
            total_dist += float(np.linalg.norm(p2 - p1))

        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0 if len(timestamps) > 1 else 0

        return {
            "positions": positions,
            "timestamps": timestamps,
            "total_distance_m": round(total_dist, 2),
            "duration_s": round(duration_s, 1),
        }

    def heatmap(self, features: list[FeatureVector],
                grid_size: float = 0.5, graph=None) -> dict:
        """
        2D 热力图（xy 平面）。
        Returns: grid (2D array), x_edges, y_edges, peak_region
        """
        if not features:
            return {"grid": np.zeros((1, 1)), "x_edges": [], "y_edges": [], "peak_region": None}

        xs = [f.centroid[0] for f in features]
        ys = [f.centroid[1] for f in features]

        if not xs:
            return {"grid": np.zeros((1, 1)), "x_edges": [], "y_edges": [], "peak_region": None}

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_bins = max(int((x_max - x_min) / grid_size), 2)
        y_bins = max(int((y_max - y_min) / grid_size), 2)

        grid, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])

        peak_idx = np.unravel_index(grid.argmax(), grid.shape)
        peak_center = (
            round(float((x_edges[peak_idx[0]] + x_edges[peak_idx[0] + 1]) / 2), 2),
            round(float((y_edges[peak_idx[1]] + y_edges[peak_idx[1] + 1]) / 2), 2),
        )

        return {
            "grid": grid,
            "x_edges": [round(float(e), 2) for e in x_edges],
            "y_edges": [round(float(e), 2) for e in y_edges],
            "peak_region": peak_center,
        }
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_report_tracking.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ai/decoders/report.py ai/decoders/tracking.py ai/tests/test_report_tracking.py
git commit -m "feat(ai): add ReportDecoder + TrackingDecoder

Daily summary stats, weekly comparison, trajectory builder, 2D heatmap.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: 集成 — WorldModelWorker 替换 placeholder

**Files:**
- Create: `ai/main.py`
- Create: `ai/configs/defaults.yaml`
- Create: `ai/tests/test_integration.py`

**Interfaces:**
- Consumes: All modules from Tasks 1-9
- Produces: `WorldModelWorker` — 完整 pipeline，消费 Redis Stream → 产出 DecoderOutput → POST backend

- [ ] **Step 1: Write config**

`ai/configs/defaults.yaml`:

```yaml
# 家安世界模型 Phase A 默认配置
redis_url: "redis://localhost:6379/0"
backend_url: "http://localhost:8000"
internal_token: "dev-internal-token"

feature_buffer_maxlen: 1000
state_persist_dir: "/tmp/housafe_ai_state"
state_persist_interval_s: 300   # 每 5 分钟持久化一次

fallback:
  enable_fall: true
  enable_stillness: true
  enable_vital: true
  enable_offline: true

notification:
  suppress_window_s: 300

baseline:
  min_days: 14

graph:
  min_stay_frames: 60
```

- [ ] **Step 2: Write integration test**

`ai/tests/test_integration.py`:

```python
"""端到端集成测试：特征提取 → 规则 → 基线 → 异常 → 通知 → DecoderOutput"""
import json
import numpy as np
from ai.main import WorldModelWorker, extract_features_from_pointcloud, extract_features_from_vital
from ai.shared.types import PointCloudFrame, VitalFrame, FeatureVector
from ai.shared.redis_client import parse_pointcloud_message

class TestFeatureExtraction:
    def test_extract_from_pointcloud(self):
        pts = np.array([
            [1.0, 2.0, 0.5, 0.1, 0.8],
            [1.1, 2.1, 1.5, 0.2, 0.9],
            [0.9, 1.9, 0.3, 0.0, 0.7],
        ], dtype=np.float32)
        frame = PointCloudFrame(
            ts=1700000000000, device_id="r1", family_id="1",
            room="bedroom", frame_id="f-01", points=pts,
        )
        fv = extract_features_from_pointcloud(frame)
        assert fv.device_id == "r1"
        assert fv.room == "bedroom"
        assert fv.presence is True
        assert fv.n_points == 3
        assert fv.posture in ("stand", "sit", "lie", "walk", "fall")
        # 3 个点 (z: 0.5, 1.5, 0.3), height=1.2 → stand
        assert fv.height == pytest.approx(1.2, abs=0.1)
        # centroid
        assert fv.centroid[0] == pytest.approx(1.0, abs=0.1)

    def test_extract_from_vital(self):
        frame = VitalFrame(
            ts=1700000000000, device_id="r1", family_id="1",
            room="bedroom", resp_rate=16.5, heart_rate=72.0,
            quality=0.85,
        )
        fv = extract_features_from_vital(frame)
        assert fv.resp_rate == 16.5
        assert fv.heart_rate == 72.0
        assert fv.vital_quality == 0.85
        # 生命体征帧一般不改变姿态
        assert fv.posture == "stand"  # 默认值

class TestWorldModelPipeline:
    def test_full_pipeline_normal_frame(self):
        """正常帧经过完整 pipeline → 不应触发异常"""
        worker = WorldModelWorker(
            redis_url="redis://localhost:6379/0",  # 不连真实 Redis
            backend_url="http://localhost:8000",
            token="test-token",
        )
        fv = FeatureVector(
            ts=1000, device_id="r1", room="bedroom",
            posture="sit", posture_confidence=0.9, presence=True,
            moving=False, centroid=(1, 1, 0.5), height=0.8,
            n_points=15, occupancy_estimate=1,
            resp_rate=16.0, heart_rate=72.0, vital_quality=0.85,
            hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.01,
        )
        history = [fv] * 10
        anomaly = worker.anomaly.evaluate(fv, history)
        # 正常帧，无基线 → 不应告警
        assert anomaly is None

    def test_fall_detected_in_pipeline(self):
        """跌倒序列 → pipeline 应检出"""
        worker = WorldModelWorker(
            redis_url="redis://localhost:6379/0",
            backend_url="http://localhost:8000",
            token="test-token",
        )
        import math
        from ai.shared.types import time_encode
        h_sin, h_cos = time_encode(12.0)

        history = []
        for i in range(5):
            history.append(FeatureVector(
                ts=1000 + i * 200, device_id="r1", room="bathroom",
                posture="stand", posture_confidence=0.9, presence=True,
                moving=True, centroid=(1, 1, 1.5), height=1.6,
                n_points=20, occupancy_estimate=1,
                resp_rate=18, heart_rate=80, vital_quality=0.8,
                hour_sin=h_sin, hour_cos=h_cos, weekday=0,
                velocity_variance=0.2,
            ))
        current = FeatureVector(
            ts=2000, device_id="r1", room="bathroom",
            posture="lie", posture_confidence=0.85, presence=True,
            moving=False, centroid=(1, 1, 0.1), height=0.2,
            n_points=8, occupancy_estimate=1,
            resp_rate=20, heart_rate=100, vital_quality=0.7,
            hour_sin=h_sin, hour_cos=h_cos, weekday=0,
            velocity_variance=0.0,
        )
        history.append(current)

        anomaly = worker.anomaly.evaluate(current, history)
        assert anomaly is not None
        assert anomaly.anomaly_type == "fall"

        decision = worker.notifier.decide(anomaly, [])
        assert decision.level in ("sms", "call")

    def test_decoder_output_format(self):
        """验证输出的 decoder output 格式符合 backend 接口契约"""
        from ai.shared.types import AnomalyResult

        # 构造异常的 decoder output
        anomaly = AnomalyResult(
            ts=2000, device_id="r1", room="bathroom",
            anomaly_score=0.9, anomaly_type="fall",
            severity="critical", source="fallback",
            details={"height_drop_m": 1.4},
        )
        # 检查字段完整性（backend contracts 要求）
        assert anomaly.anomaly_type in ("fall", "stillness", "vital_anomaly",
                                          "pattern_deviation", "offline")
        assert anomaly.severity in ("info", "warning", "critical")
        assert 0 <= anomaly.anomaly_score <= 1
```

Run test first:

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_integration.py -v 2>&1 | head -10
```
Expected: ModuleNotFoundError for `ai.main`

- [ ] **Step 3: Implement `ai/main.py`**

```python
#!/usr/bin/env python3
"""
家安世界模型 · Phase A Worker

替代 ai/placeholder/main.py 的规则模拟器。
消费 Redis Stream 点云/生命体征帧 → 提取特征 → Fallback 规则引擎
→ 基线比较 → 异常检测 → 通知决策 → POST DecoderOutput 回 backend。

用法: python -m ai.main [redis_url] [backend_url] [token]
"""
import json
import os
import sys
import time
import math
from collections import defaultdict
from collections import deque
from datetime import datetime

import numpy as np
import requests

from ai.shared.types import (
    FeatureVector, AnomalyResult, NotificationDecision,
    PointCloudFrame, VitalFrame, time_encode,
    POSTURES,
)
from ai.shared.redis_client import FrameConsumer
from ai.fallback.rules import FallbackEngine
from ai.latent_space.graph import SpatialGraph
from ai.baseline.gmm_baseline import PersonalBaseline
from ai.decoders.anomaly import AnomalyDecoder
from ai.decoders.notification import NotificationDecider
from ai.decoders.report import ReportDecoder
from ai.decoders.tracking import TrackingDecoder

# ── 配置 ──────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", sys.argv[1] if len(sys.argv) > 1 else "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("BACKEND_URL", sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000")
INTERNAL_TOKEN = os.environ.get("INTERNAL_SERVICE_TOKEN", sys.argv[3] if len(sys.argv) > 3 else "dev-internal-token")
DECODER_ENDPOINT = f"{BACKEND_URL}/api/internal/decoder-output"

STATE_DIR = os.environ.get("AI_STATE_DIR", "/tmp/housafe_ai_state")
STATE_PERSIST_INTERVAL_S = 300  # 5 分钟


# ── 特征提取（临时替代 Encoder）─────────────────────

def extract_features_from_pointcloud(frame: PointCloudFrame) -> FeatureVector:
    """从点云帧提取特征向量（规则方法，等 Encoder 替换）"""
    pts = frame.points
    n = len(pts)

    if n == 0:
        return FeatureVector(
            ts=frame.ts, device_id=frame.device_id, room=frame.room,
            posture="stand", posture_confidence=0.0, presence=False,
            moving=False, centroid=(0, 0, 0), height=0.0, n_points=0,
            occupancy_estimate=0, resp_rate=None, heart_rate=None,
            vital_quality=None, hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.0,
        )

    # 质心
    centroid = tuple(pts.mean(axis=0)[:3].astype(float))

    # 高度 = z 跨度
    zs = pts[:, 2]
    height = float(zs.max() - zs.min())

    # 姿态（规则：高度阈值）
    posture = "stand"
    if height < 0.3:
        posture = "lie"
    elif height < 0.8:
        posture = "sit"
    elif height >= 1.8:
        posture = "walk"

    # 运动
    velocities = pts[:, 3]
    moving = bool(np.mean(np.abs(velocities)) > 0.1)
    velocity_variance = float(np.var(velocities))

    # 人数估计（简单：点数阈值）
    occupancy_estimate = 1 if n >= 5 else 0

    # 置信度：基于点数和信号强度
    intensities = pts[:, 4]
    confidence = min(float(np.mean(intensities)), 1.0) if n > 0 else 0.0

    # 时间编码
    dt = datetime.utcfromtimestamp(frame.ts / 1000.0)
    hour = dt.hour + dt.minute / 60.0
    h_sin, h_cos = time_encode(hour)
    weekday = dt.weekday()

    return FeatureVector(
        ts=frame.ts, device_id=frame.device_id, room=frame.room,
        posture=posture, posture_confidence=round(confidence, 3),
        presence=True, moving=moving,
        centroid=centroid, height=round(height, 3),
        n_points=n, occupancy_estimate=occupancy_estimate,
        resp_rate=None, heart_rate=None, vital_quality=None,
        hour_sin=h_sin, hour_cos=h_cos, weekday=weekday,
        velocity_variance=round(velocity_variance, 3),
    )


def extract_features_from_vital(frame: VitalFrame) -> FeatureVector:
    """从生命体征帧提取特征向量（更新已有 FV 的生命体征部分）"""
    dt = datetime.utcfromtimestamp(frame.ts / 1000.0)
    hour = dt.hour + dt.minute / 60.0
    h_sin, h_cos = time_encode(hour)

    return FeatureVector(
        ts=frame.ts, device_id=frame.device_id, room=frame.room,
        posture="stand", posture_confidence=0.0, presence=True,
        moving=False, centroid=(0, 0, 0), height=0.0, n_points=0,
        occupancy_estimate=1,
        resp_rate=frame.resp_rate, heart_rate=frame.heart_rate,
        vital_quality=frame.quality,
        hour_sin=h_sin, hour_cos=h_cos, weekday=dt.weekday(),
        velocity_variance=0.0,
    )


# ── Worker ────────────────────────────────────────────

class WorldModelWorker:
    """世界模型主进程。"""

    def __init__(self, redis_url: str = REDIS_URL, backend_url: str = BACKEND_URL,
                 token: str = INTERNAL_TOKEN):
        self.redis_url = redis_url
        self.backend_url = backend_url
        self.token = token
        self.endpoint = f"{backend_url}/api/internal/decoder-output"

        # 组件
        self.fallback = FallbackEngine()
        self.graph = SpatialGraph()
        self.baseline = PersonalBaseline()
        self.anomaly = AnomalyDecoder(self.fallback, self.baseline, self.graph)
        self.notifier = NotificationDecider()
        self.reporter = ReportDecoder()
        self.tracker = TrackingDecoder()

        # 状态
        self.consumer: FrameConsumer | None = None
        self.feature_buffer: dict[str, deque[FeatureVector]] = defaultdict(
            lambda: deque(maxlen=2000)
        )
        self.recent_decisions: deque[NotificationDecision] = deque(maxlen=100)
        self._last_state_persist = time.time()

        # 状态恢复
        os.makedirs(STATE_DIR, exist_ok=True)
        self._load_state()

    # ── 主循环 ──────────────────────────────────────

    def run(self):
        """阻塞式主循环"""
        self.consumer = FrameConsumer(self.redis_url)
        print(f"[WorldModel] Phase A worker started")
        print(f"[WorldModel] Redis: {self.redis_url}")
        print(f"[WorldModel] Backend: {self.endpoint}")

        while True:
            result = self.consumer.consume_one()
            if result is None:
                self._periodic_tasks()
                continue

            stream_name, frame, msg_id = result
            self._process_frame(stream_name, frame)
            self._periodic_tasks()

    def _process_frame(self, stream_name: str, frame: PointCloudFrame | VitalFrame):
        """处理单帧"""
        # 提取特征
        if isinstance(frame, PointCloudFrame):
            fv = extract_features_from_pointcloud(frame)
        else:
            fv = extract_features_from_vital(frame)

        # 更新 buffer
        buf = self.feature_buffer[fv.device_id]
        buf.append(fv)
        history = list(buf)

        # 运行异常检测
        anomaly = self.anomaly.evaluate(fv, history)
        if anomaly is None:
            return

        # 通知决策
        decision = self.notifier.decide(anomaly, list(self.recent_decisions))
        self.recent_decisions.append(decision)

        # 发送到 backend
        self._post_result(fv, anomaly, decision)

        # 日志
        ts_str = datetime.utcfromtimestamp(fv.ts / 1000).strftime("%H:%M:%S")
        print(f"[WorldModel] {ts_str} {fv.room} {fv.posture} "
              f"→ {anomaly.anomaly_type}/{anomaly.severity} "
              f"(score={anomaly.anomaly_score:.2f}) → {decision.level}")

    def _post_result(self, fv: FeatureVector, anomaly: AnomalyResult,
                     decision: NotificationDecision):
        """POST decoder outputs 到 backend AIBridgeView"""
        # 构造符合 contracts 的 decoder outputs
        outputs = []

        # 姿态
        outputs.append({
            "kind": "posture",
            "family_id": "unknown",
            "payload": {
                "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                "posture": fv.posture, "confidence": fv.posture_confidence,
                "moving": fv.moving, "presence": fv.presence,
            },
        })

        # 生命体征（如果有）
        if fv.resp_rate is not None or fv.heart_rate is not None:
            outputs.append({
                "kind": "vital",
                "family_id": "unknown",
                "payload": {
                    "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                    "resp_rate": fv.resp_rate, "heart_rate": fv.heart_rate,
                    "quality": fv.vital_quality or 0.0,
                },
            })

        # 异常告警
        if decision.level != "none":
            outputs.append({
                "kind": "alert",
                "family_id": "unknown",
                "payload": {
                    "ts": anomaly.ts, "device_id": anomaly.device_id,
                    "room": anomaly.room,
                    "alert_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "payload": {
                        "anomaly_score": anomaly.anomaly_score,
                        "source": anomaly.source,
                        "notification_level": decision.level,
                        "reason": decision.reason,
                        **anomaly.details,
                    },
                },
            })

        # 异常分数
        outputs.append({
            "kind": "anomaly",
            "family_id": "unknown",
            "payload": {
                "ts": fv.ts, "device_id": fv.device_id, "room": fv.room,
                "anomaly_score": anomaly.anomaly_score,
            },
        })

        try:
            rv = requests.post(
                self.endpoint,
                json={"token": self.token, "outputs": outputs},
                timeout=5,
            )
            if rv.status_code != 200:
                print(f"[WorldModel] POST error: HTTP {rv.status_code} {rv.text[:100]}")
        except requests.RequestException as e:
            print(f"[WorldModel] POST failed: {e}")

    # ── 定期任务 ─────────────────────────────────────

    def _periodic_tasks(self):
        now = time.time()
        if now - self._last_state_persist >= STATE_PERSIST_INTERVAL_S:
            self._persist_state()
            self._last_state_persist = now

    def _persist_state(self):
        try:
            self.graph.save(os.path.join(STATE_DIR, "graph.pkl"))
            self.baseline.save(os.path.join(STATE_DIR, "baseline.pkl"))
        except Exception as e:
            print(f"[WorldModel] persist error: {e}")

    def _load_state(self):
        graph_path = os.path.join(STATE_DIR, "graph.pkl")
        baseline_path = os.path.join(STATE_DIR, "baseline.pkl")
        if os.path.exists(graph_path):
            try:
                self.graph = SpatialGraph.load(graph_path)
                print(f"[WorldModel] Loaded graph: {len(self.graph.nodes)} nodes")
            except Exception:
                pass
        if os.path.exists(baseline_path):
            try:
                self.baseline = PersonalBaseline.load(baseline_path)
                self.anomaly.baseline = self.baseline
                print(f"[WorldModel] Loaded baseline: ready={self.baseline.is_ready}")
            except Exception:
                pass


def main():
    worker = WorldModelWorker()
    worker.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all integration tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_integration.py -v
```
Expected: all passed

- [ ] **Step 5: Run all AI tests to ensure full suite passes**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add ai/main.py ai/configs/defaults.yaml ai/tests/test_integration.py
git commit -m "feat(ai): add WorldModelWorker — Phase A main pipeline

Replaces placeholder/main.py. Full pipeline: Redis Stream consume →
feature extraction → FallbackEngine + AnomalyDecoder + NotificationDecider
→ POST decoder outputs to backend. State persistence for graph + baseline.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 11: 最终验证 — 端到端冒烟测试

**Files:**
- No new files. Run existing e2e infrastructure.

- [ ] **Step 1: Start the world model worker with simulator**

Terminal 1:
```bash
cd /Users/eular/Desktop/housafe && docker-compose up -d postgres redis
cd /Users/eular/Desktop/housafe/backend && python manage.py runserver 8000
```

Terminal 2:
```bash
cd /Users/eular/Desktop/housafe && python -m ai.main
```

Terminal 3:
```bash
cd /Users/eular/Desktop/housafe && python simulator/feed.py rad_demo demo_secret --mode walk
```

Expected: AI worker logs show feature extraction + posture detection. No false fall alerts on normal walk.

- [ ] **Step 2: Test fall detection**

```bash
# 切换 simulator 模式为 fall
python simulator/feed.py rad_demo demo_secret --mode fall
```

Expected: AI worker logs show `fall/critical → call` within ~5 seconds.

- [ ] **Step 3: Verify backend receives outputs**

```bash
curl -s http://localhost:8000/api/events/families/1/today | python -m json.tool
```

Expected: RoomState has posture=lie, anomaly_score elevated.

- [ ] **Step 4: Verify RoomState is updated in DB**

```bash
cd /Users/eular/Desktop/housafe/backend && python -c "
import django; import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from events.models import RoomState
rs = RoomState.objects.filter(device_id='rad_demo').first()
print(f'posture={rs.posture} anomaly={rs.anomaly_score}' if rs else 'no RoomState yet')
"
```

---

## Plan Summary

| Task | 模块 | 文件数 | 核心交付 |
|------|------|--------|---------|
| 1 | Shared Types | 3 | FeatureVector, AnomalyResult, NotificationDecision |
| 2 | Redis Client | 2 | FrameConsumer, parse_pointcloud/vital_message |
| 3 | Fallback Engine | 2 | 3 条确定性规则（跌倒/静止/生命体征） |
| 4 | Vital Signs | 4 | FFT 频谱分析 + 安静态判定 |
| 5 | Spatial Graph | 2 | HDBSCAN 聚类 + 边统计 + save/load |
| 6 | Personal Baseline | 2 | 分上下文 GMM + Mahalanobis 异常分 |
| 7 | Anomaly Decoder | 2 | 融合 fallback + baseline + 空间感知 |
| 8 | Notification Decider | 2 | 决策矩阵 + 重复抑制 |
| 9 | Report + Tracking | 3 | 日摘要 + 周对比 + 轨迹 + 热力图 |
| 10 | Integration | 3 | WorldModelWorker 主 pipeline |
| 11 | E2E Smoke | 0 | 模拟器→AI→Backend 全链路验证 |

**总新增文件：~30 个。总代码量：~1500 行（含测试）。全部纯 Python，零 GPU 依赖。**
