import pytest
from unittest.mock import patch, MagicMock
from channels.testing import WebsocketCommunicator
from housafe.asgi import application

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
async def test_full_chain_ingest_to_db(db):
    """端到端：ingest 收点云帧 → AIBridgeView 落库 → RoomState 可查"""
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice

    u = await User.objects.acreate_user("k", "", "pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")

    # Step 1: ingest 收点云帧
    with patch("ingest.consumers.redis.Redis") as m:
        mock_client = MagicMock()
        m.from_url.return_value = mock_client

        ing = WebsocketCommunicator(application, "/ws/ingest")
        await ing.connect()
        await ing.send_json_to({"device_id": "d1", "secret": "s1"})
        await ing.receive_json_from()  # auth ack

        for i in range(3):
            await ing.send_json_to({
                "kind": "point_cloud",
                "payload": {
                    "ts": 1700000000000 + i,
                    "radar_id": "d1",
                    "room": "bed",
                    "frame_id": f"f-{i:03d}",
                    "points": [{"x": 1.0, "y": 0.5, "z": 1.5, "velocity": 0.1, "intensity": 0.9}],
                },
            })
            ack = await ing.receive_json_from()
            assert ack["ack"] == "buffered"

        await ing.disconnect()

    # Step 2: AIBridgeView 落库
    from channels.db import database_sync_to_async

    @database_sync_to_async
    def call_ai_bridge():
        from rest_framework.test import APIClient
        api = APIClient()
        return api.post("/api/internal/decoder-output", {
            "token": "dev-internal-token",
            "outputs": [
                {
                    "kind": "posture",
                    "family_id": fam.id,
                    "payload": {"ts": 1700000000100, "device_id": "d1", "room": "bed", "posture": "walk", "confidence": 0.9},
                },
                {
                    "kind": "vital",
                    "family_id": fam.id,
                    "payload": {"ts": 1700000000200, "device_id": "d1", "room": "bed", "resp_rate": 16.0, "heart_rate": 72.0, "quality": 0.85},
                },
            ],
        }, format="json")

    r = await call_ai_bridge()
    assert r.status_code == 200
    assert r.data["count"] == 2

    # Step 3: 查询 RoomState
    @database_sync_to_async
    def check_room_state():
        from events.models import RoomState, Alert
        rs = RoomState.objects.get(device_id="d1")
        assert rs.posture == "walk"
        assert rs.heart_rate == 72.0
        assert rs.resp_rate == 16.0
        return True

    assert await check_room_state()
