import pytest
from pydantic import ValidationError
from housafe_contracts.events import (
    parse_event,
    parse_frame,
    PostureEvent,
    PointCloudFrame,
    PointXYZVI,
    VitalSignsFrame,
    DecoderPosture,
    DecoderVital,
    DecoderAlert,
    DecoderOccupancy,
    DecoderAnomaly,
    RoomState,
    Heartbeat,
    POSTURES,
    FRAME_KINDS,
    ALERT_TYPES,
    SEVERITY_LEVELS,
)


# ── 旧 parse_event（deprecated，保留向后兼容）───────────────────────────

def test_posture_valid():
    e = parse_event("posture", {"ts": 1700000000000, "radar_id": "r1", "room": "bedroom", "seq": 5, "posture": "walk", "confidence": 0.9})
    assert isinstance(e, PostureEvent) and e.posture == "walk"


def test_posture_rejects_bad_enum():
    with pytest.raises(ValidationError):
        parse_event("posture", {"ts": 1, "radar_id": "r1", "room": "b", "seq": 1, "posture": "jump", "confidence": 0.5})


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_event("nope", {})


def test_postures_frozen():
    assert POSTURES == ("stand", "sit", "lie", "walk", "fall")


# ── parse_frame：新点云/体征帧 ───────────────────────────────────────────

class TestPointCloudFrame:
    def test_valid(self):
        f = parse_frame("point_cloud", {
            "ts": 1700000000000,
            "radar_id": "r1",
            "room": "bedroom",
            "frame_id": "f-001",
            "points": [{"x": 1.0, "y": 2.0, "z": 3.0, "velocity": 0.5, "intensity": 0.9}],
        })
        assert isinstance(f, PointCloudFrame)
        assert f.frame_id == "f-001"
        assert len(f.points) == 1
        assert f.points[0].x == 1.0
        assert f.points[0].velocity == 0.5

    def test_empty_points_rejected(self):
        with pytest.raises(ValidationError):
            parse_frame("point_cloud", {
                "ts": 1700000000000,
                "radar_id": "r1",
                "room": "bedroom",
                "frame_id": "f-001",
                "points": [],
            })

    def test_default_velocity_intensity(self):
        p = PointXYZVI(x=0, y=0, z=0)
        assert p.velocity == 0.0
        assert p.intensity == 0.0


class TestVitalSignsFrame:
    def test_valid(self):
        f = parse_frame("vital", {
            "ts": 1700000000000,
            "radar_id": "r1",
            "room": "bedroom",
            "quiet": True,
            "resp_rate": 16.0,
            "heart_rate": 72.0,
            "quality": 0.85,
        })
        assert isinstance(f, VitalSignsFrame)
        assert f.heart_rate == 72.0

    def test_nullable_vitals(self):
        f = parse_frame("vital", {
            "ts": 1700000000000,
            "radar_id": "r1",
            "room": "bedroom",
            "quiet": False,
            "quality": 0.5,
        })
        assert f.resp_rate is None
        assert f.heart_rate is None


class TestHeartbeatFrame:
    def test_heartbeat_via_parse_frame(self):
        f = parse_frame("heartbeat", {
            "ts": 1700000000000,
            "radar_id": "r1",
            "status": "online",
            "fw_version": "1.2.3",
        })
        assert isinstance(f, Heartbeat)
        assert f.status == "online"


def test_parse_frame_unknown_kind():
    with pytest.raises(ValueError):
        parse_frame("posture", {})


def test_frame_kinds():
    assert FRAME_KINDS == ("point_cloud", "vital", "heartbeat")


# ── Decoder 输出模型 ─────────────────────────────────────────────────────

class TestDecoderPosture:
    def test_valid(self):
        d = DecoderPosture(ts=1700000000000, device_id="d1", room="bedroom", posture="walk", confidence=0.9)
        assert d.posture == "walk"
        assert d.moving is False  # default
        assert d.presence is True  # default

    def test_bad_posture_rejected(self):
        with pytest.raises(ValidationError):
            DecoderPosture(ts=1, device_id="d1", room="b", posture="jump", confidence=0.5)

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            DecoderPosture(ts=1, device_id="d1", room="b", posture="stand", confidence=1.5)


class TestDecoderVital:
    def test_valid(self):
        d = DecoderVital(ts=1700000000000, device_id="d1", room="bedroom", resp_rate=16.0, heart_rate=72.0, quality=0.8)
        assert d.heart_rate == 72.0

    def test_nullable(self):
        d = DecoderVital(ts=1, device_id="d1", room="b", quality=0.5)
        assert d.resp_rate is None


class TestDecoderAlert:
    def test_valid(self):
        d = DecoderAlert(ts=1700000000000, device_id="d1", room="bedroom", alert_type="fall", severity="critical", payload={"detail": "test"})
        assert d.alert_type == "fall"
        assert d.severity == "critical"
        assert d.payload == {"detail": "test"}

    def test_default_payload(self):
        d = DecoderAlert(ts=1, device_id="d1", room="b", alert_type="offline", severity="info")
        assert d.payload == {}


class TestDecoderOccupancy:
    def test_valid(self):
        d = DecoderOccupancy(ts=1, device_id="d1", room="b", count=2)
        assert d.count == 2

    def test_negative_count_rejected(self):
        with pytest.raises(ValidationError):
            DecoderOccupancy(ts=1, device_id="d1", room="b", count=-1)


class TestDecoderAnomaly:
    def test_valid(self):
        d = DecoderAnomaly(ts=1, device_id="d1", room="b", anomaly_score=0.75)
        assert d.anomaly_score == 0.75

    def test_score_range(self):
        with pytest.raises(ValidationError):
            DecoderAnomaly(ts=1, device_id="d1", room="b", anomaly_score=1.5)


# ── RoomState ─────────────────────────────────────────────────────────────

class TestRoomState:
    def test_minimal(self):
        rs = RoomState(device_id="d1", room="bedroom", ts=1700000000000)
        assert rs.posture is None
        assert rs.anomaly_score == 0.0
        assert rs.occupancy_count == 0

    def test_full(self):
        rs = RoomState(
            device_id="d1", room="bedroom", ts=1700000000000,
            posture="lie", confidence=0.95, presence=True, moving=False,
            resp_rate=15.0, heart_rate=68.0, quality=0.9,
            anomaly_score=0.05, occupancy_count=1,
        )
        assert rs.posture == "lie"


# ── 常量 ──────────────────────────────────────────────────────────────────

def test_alert_types():
    assert ALERT_TYPES == ("fall", "stillness", "vital_anomaly", "pattern_deviation", "offline")


def test_severity_levels():
    assert SEVERITY_LEVELS == ("info", "warning", "critical")
