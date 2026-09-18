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

