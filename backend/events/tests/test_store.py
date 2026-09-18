import pytest
pytestmark = pytest.mark.django_db
from events.store import store_decoder_output
from events.models import RoomState, Alert
from housafe_contracts.events import (
    DecoderPosture,
    DecoderVital,
    DecoderAlert,
    DecoderOccupancy,
    DecoderAnomaly,
)


class TestStoreDecoderOutput:
    def test_posture_upserts_room_state(self):
        d = DecoderPosture(ts=1700000000000, device_id="d1", room="bed", posture="walk", confidence=0.9)
        store_decoder_output("d1", "bed", d, ts_recv=1700000000050)
        rs = RoomState.objects.get(device_id="d1")
        assert rs.posture == "walk"
        assert rs.confidence == 0.9
        assert rs.presence is True

        # 第二次写入同一 device_id → upsert，只有一行
        d2 = DecoderPosture(ts=1700000001000, device_id="d1", room="bed", posture="lie", confidence=0.95, moving=True, presence=True)
        store_decoder_output("d1", "bed", d2, ts_recv=1700000001050)
        assert RoomState.objects.filter(device_id="d1").count() == 1
        rs.refresh_from_db()
        assert rs.posture == "lie"
        assert rs.moving is True

    def test_vital_upserts_room_state(self):
        d = DecoderVital(ts=1700000000000, device_id="d1", room="bed", resp_rate=16.0, heart_rate=72.0, quality=0.85)
        store_decoder_output("d1", "bed", d)
        rs = RoomState.objects.get(device_id="d1")
        assert rs.heart_rate == 72.0
        assert rs.resp_rate == 16.0
        # posture fields should be untouched
        assert rs.posture is None

    def test_alert_creates_record(self):
        d = DecoderAlert(ts=1700000000000, device_id="d1", room="bed", alert_type="fall", severity="critical", payload={"detail": "detected"})
        store_decoder_output("d1", "bed", d)
        alerts = Alert.objects.filter(device_id="d1")
        assert alerts.count() == 1
        a = alerts.first()
        assert a.alert_type == "fall"
        assert a.severity == "critical"
        assert a.payload == {"detail": "detected"}

    def test_occupancy_upserts_room_state(self):
        d = DecoderOccupancy(ts=1, device_id="d1", room="bed", count=2)
        store_decoder_output("d1", "bed", d)
        rs = RoomState.objects.get(device_id="d1")
        assert rs.occupancy_count == 2

    def test_anomaly_upserts_room_state(self):
        d = DecoderAnomaly(ts=1, device_id="d1", room="bed", anomaly_score=0.75)
        store_decoder_output("d1", "bed", d)
        rs = RoomState.objects.get(device_id="d1")
        assert rs.anomaly_score == 0.75

    def test_multiple_output_types_merge(self):
        """posture + vital 先后写入，RoomState 合并两方字段"""
        store_decoder_output("d2", "bed", DecoderPosture(ts=1, device_id="d2", room="bed", posture="stand", confidence=0.8))
        store_decoder_output("d2", "bed", DecoderVital(ts=2, device_id="d2", room="bed", resp_rate=15.0, heart_rate=65.0, quality=0.9))
        rs = RoomState.objects.get(device_id="d2")
        assert rs.posture == "stand"
        assert rs.heart_rate == 65.0
        assert rs.ts == 2  # latest decoder output ts
