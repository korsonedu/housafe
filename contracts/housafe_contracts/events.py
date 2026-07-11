from typing import Literal
from pydantic import BaseModel, Field

POSTURES = ("stand", "sit", "lie", "walk", "fall")
EVENT_KINDS = ("presence", "posture", "vital", "occupancy", "heartbeat")


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
    if kind not in _MODELS:
        raise ValueError(f"unknown event kind: {kind}")
    return _MODELS[kind](**payload)
