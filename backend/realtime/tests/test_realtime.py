import pytest
from unittest.mock import patch, MagicMock
from channels.testing import WebsocketCommunicator
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from housafe.asgi import application

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
async def test_app_room_state_handler(db):
    """AppConsumer 收到 room.state → 推送给客户端"""
    from django.contrib.auth.models import User
    from families.models import Family
    from rest_framework_simplejwt.tokens import AccessToken

    u = await User.objects.acreate_user("k", "", "pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    token = str(AccessToken.for_user(u))

    app = WebsocketCommunicator(application, f"/ws/app?token={token}&family={fam.id}")
    assert (await app.connect())[0]

    # 模拟 AIBridgeView → group_send room.state
    payload = {
        "type": "room.state",
        "room_name": "bed",
        "device_online": True,
        "posture": "walk",
        "posture_ts": 1700000000000,
        "heart_rate": 72.0,
        "breath_rate": 16.0,
        "anomaly_score": 0.02,
    }
    channel = get_channel_layer()
    await channel.group_send(
        f"family_{fam.id}",
        {"type": "room.state", "payload": payload},
    )

    msg = await app.receive_json_from()
    assert msg["type"] == "room.state"
    assert msg["room_name"] == "bed"
    assert msg["posture"] == "walk"
    assert msg["heart_rate"] == 72.0

    await app.disconnect()


@pytest.mark.asyncio
async def test_ingest_to_ai_bridge_to_app_flow(db):
    """端到端：ingest 收点云 → AIBridgeView store + push → App 收到 room.state"""
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    from rest_framework_simplejwt.tokens import AccessToken

    u = await User.objects.acreate_user("k", "", "pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    token = str(AccessToken.for_user(u))

    app = WebsocketCommunicator(application, f"/ws/app?token={token}&family={fam.id}")
    assert (await app.connect())[0]

    # Step 1: ingest 收点云帧（mock Redis XADD）
    with patch("ingest.consumers.redis.Redis") as m:
        mock_client = MagicMock()
        m.from_url.return_value = mock_client

        ing = WebsocketCommunicator(application, "/ws/ingest")
        await ing.connect()
        await ing.send_json_to({"device_id": "d1", "secret": "s1"})
        await ing.receive_json_from()  # auth ack

        await ing.send_json_to({
            "kind": "point_cloud",
            "payload": {
                "ts": 1700000000000,
                "radar_id": "d1",
                "room": "bed",
                "frame_id": "f-001",
                "points": [{"x": 1.0, "y": 0.5, "z": 2.0, "velocity": 0.1, "intensity": 0.9}],
            },
        })
        ack = await ing.receive_json_from()
        assert ack["ack"] == "buffered"
        mock_client.xadd.assert_called_once()

        await ing.disconnect()

    # Step 2: AIBridgeView 接收 decoder output（用 sync_to_async 避免 Django async unsafe）
    from channels.db import database_sync_to_async

    @database_sync_to_async
    def call_ai_bridge():
        from rest_framework.test import APIClient
        api = APIClient()
        return api.post("/api/internal/decoder-output", {
            "token": "dev-internal-token",
            "outputs": [{
                "kind": "posture",
                "family_id": fam.id,
                "payload": {"ts": 1700000001000, "device_id": "d1", "room": "bed", "posture": "walk", "confidence": 0.9},
            }],
        }, format="json")

    r = await call_ai_bridge()
    assert r.status_code == 200

    # Step 3: App 应收到 room.state 推送
    msg = await app.receive_json_from()
    assert msg["type"] == "room.state"
    assert msg["room_name"] == "bed"
    assert msg["posture"] == "walk"

    await app.disconnect()
