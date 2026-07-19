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
