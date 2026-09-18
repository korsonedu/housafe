import pytest
pytestmark = pytest.mark.django_db
from events.store import store_decoder_output
from events.models import RoomState
from housafe_contracts.events import DecoderPosture, DecoderVital


def test_today_returns_latest_per_room(api):
    """TodayView 从 RoomState 读取每个房间的最新状态"""
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice

    u = User.objects.create_user("k", "", "pw12345678")
    api.post("/api/auth/login", {"username": "k", "password": "pw12345678"}, format="json")
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {api.handler._force_token}")
    # 直接 API 登录获取 token
    r = api.post("/api/auth/login", {"username": "k", "password": "pw12345678"}, format="json")
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {r.data['access']}")

    fid = api.post("/api/families", {"name": "f"}, format="json").data["id"]
    did = api.post(f"/api/families/{fid}/devices/bind", {"room": "bed"}, format="json").data["device_id"]

    # 写入 RoomState
    store_decoder_output(did, "bed", DecoderPosture(ts=1700000000001, device_id=did, room="bed", posture="walk", confidence=0.9))
    store_decoder_output(did, "bed", DecoderVital(ts=1700000000002, device_id=did, room="bed", resp_rate=16.0, heart_rate=72.0, quality=0.85))

    r = api.get(f"/api/families/{fid}/today")
    assert r.status_code == 200
    rooms = r.data["rooms"]
    assert len(rooms) == 1
    assert rooms[0]["room_name"] == "bed"
    assert rooms[0]["posture"] == "walk"
    assert rooms[0]["heart_rate"] == 72.0
    assert rooms[0]["breath_rate"] == 16.0


def test_ai_bridge_unauthorized(api):
    """AIBridgeView 需要 INTERNAL_SERVICE_TOKEN"""
    r = api.post("/api/internal/decoder-output", {"token": "wrong", "outputs": []}, format="json")
    assert r.status_code == 401


def test_ai_bridge_stores_and_pushes(api):
    """AIBridgeView 接收 Decoder outputs，落库 + 推送"""
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice

    u = User.objects.create_user("k", "", "pw12345678")
    fam = Family.objects.create(name="f", owner=u)
    dev = RadarDevice.objects.create(device_id="d1", secret="s1", family=fam, room="bed")

    r = api.post("/api/internal/decoder-output", {
        "token": "dev-internal-token",
        "outputs": [{
            "kind": "posture",
            "family_id": fam.id,
            "payload": {"ts": 1700000000000, "device_id": "d1", "room": "bed", "posture": "walk", "confidence": 0.9},
        }],
    }, format="json")
    assert r.status_code == 200
    assert r.data["count"] == 1

    rs = RoomState.objects.get(device_id="d1")
    assert rs.posture == "walk"


def test_ai_bridge_empty_outputs(api):
    r = api.post("/api/internal/decoder-output", {"token": "dev-internal-token", "outputs": []}, format="json")
    assert r.status_code == 400
