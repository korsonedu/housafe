from typing import Literal
from pydantic import BaseModel, Field

POSTURES = ("stand", "sit", "lie", "walk", "fall")
EVENT_KINDS = ("presence", "posture", "vital", "occupancy", "heartbeat")
FRAME_KINDS = ("point_cloud", "vital", "heartbeat")
ALERT_TYPES = ("fall", "stillness", "vital_anomaly", "pattern_deviation", "offline")
SEVERITY_LEVELS = ("info", "warning", "critical")

# ── Deprecated: 旧事件模型（v2.x 分模块管线，保留向后兼容）──────────────

class _Base(BaseModel):
    ts: int = Field(..., description="UTC ms, source-authoritative")
    radar_id: str
    room: str
    seq: int


class PresenceEvent(_Base):
    presence: bool
    moving: bool


class PostureEvent(_Base):
    posture: Literal["stand", "sit", "lie", "walk", "fall"]
    confidence: float = Field(..., ge=0, le=1)


class VitalEvent(_Base):
    quiet: bool
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float = Field(..., ge=0, le=1)


class OccupancyEvent(_Base):
    count: int = Field(..., ge=0)


class Heartbeat(BaseModel):
    ts: int
    radar_id: str
    status: Literal["online", "offline"]
    fw_version: str


_MODELS = {
    "presence": PresenceEvent,
    "posture": PostureEvent,
    "vital": VitalEvent,
    "occupancy": OccupancyEvent,
    "heartbeat": Heartbeat,
}


def parse_event(kind: str, payload: dict) -> BaseModel:
    """Deprecated: 旧事件解析，v3.0 起用 parse_frame()"""
    if kind not in _MODELS:
        raise ValueError(f"unknown event kind: {kind}")
    return _MODELS[kind](**payload)


# ── v3.0: 点云帧 / 生命体征帧 / Decoder 输出 ────────────────────────────

class PointXYZVI(BaseModel):
    x: float
    y: float
    z: float
    velocity: float = 0.0
    intensity: float = 0.0


class PointCloudFrame(BaseModel):
    ts: int = Field(..., description="UTC ms, source-authoritative")
    radar_id: str
    room: str
    frame_id: str
    points: list[PointXYZVI] = Field(..., min_length=1)


class VitalSignsFrame(BaseModel):
    ts: int = Field(..., description="UTC ms, source-authoritative")
    radar_id: str
    room: str
    quiet: bool
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float = Field(..., ge=0, le=1)


_FRAME_MODELS = {
    "point_cloud": PointCloudFrame,
    "vital": VitalSignsFrame,
    "heartbeat": Heartbeat,
}


def parse_frame(kind: str, payload: dict) -> PointCloudFrame | VitalSignsFrame | Heartbeat:
    if kind not in _FRAME_MODELS:
        raise ValueError(f"unknown frame kind: {kind}")
    return _FRAME_MODELS[kind](**payload)


# ── Decoder 输出（AI → Backend）──────────────────────────────────────────

class DecoderPosture(BaseModel):
    ts: int = Field(..., description="UTC ms")
    device_id: str
    room: str
    posture: Literal["stand", "sit", "lie", "walk", "fall"]
    confidence: float = Field(..., ge=0, le=1)
    moving: bool = False
    presence: bool = True


class DecoderVital(BaseModel):
    ts: int = Field(..., description="UTC ms")
    device_id: str
    room: str
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float = Field(..., ge=0, le=1)


class DecoderAlert(BaseModel):
    ts: int = Field(..., description="UTC ms")
    device_id: str
    room: str
    alert_type: Literal["fall", "stillness", "vital_anomaly", "pattern_deviation", "offline"]
    severity: Literal["info", "warning", "critical"]
    payload: dict = Field(default_factory=dict)


class DecoderOccupancy(BaseModel):
    ts: int = Field(..., description="UTC ms")
    device_id: str
    room: str
    count: int = Field(..., ge=0)


class DecoderAnomaly(BaseModel):
    ts: int = Field(..., description="UTC ms")
    device_id: str
    room: str
    anomaly_score: float = Field(..., ge=0, le=1)


DecoderOutput = DecoderPosture | DecoderVital | DecoderAlert | DecoderOccupancy | DecoderAnomaly


# ── 房间状态（Backend 聚合，推 App）─────────────────────────────────────

class RoomState(BaseModel):
    device_id: str
    room: str
    ts: int = Field(..., description="UTC ms of latest decoder output")
    posture: str | None = None
    confidence: float | None = None
    presence: bool = True
    moving: bool = False
    resp_rate: float | None = None
    heart_rate: float | None = None
    quality: float | None = None
    anomaly_score: float = 0.0
    occupancy_count: int = 0
