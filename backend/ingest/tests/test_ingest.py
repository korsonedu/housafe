import pytest
from unittest.mock import patch, MagicMock
from channels.testing import WebsocketCommunicator
from housafe.asgi import application

pytestmark = pytest.mark.django_db(transaction=True)


async def _device(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice

    u = await User.objects.acreate_user("k", "", "pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    return await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed"), fam


@pytest.fixture
def mock_redis():
    with patch("ingest.consumers.redis.Redis") as m:
        mock_client = MagicMock()
        m.from_url.return_value = mock_client
        yield mock_client


@pytest.mark.asyncio
async def test_bad_secret_closes(db):
    await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "WRONG"})
    msg = await c.receive_output()
    assert msg["type"] == "websocket.close"


@pytest.mark.asyncio
async def test_point_cloud_xadd(mock_redis, db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    ack = await c.receive_json_from()
    assert ack == {"ack": "auth"}

    await c.send_json_to({
        "kind": "point_cloud",
        "payload": {
            "ts": 1700000000000,
            "radar_id": "d1",
            "room": "bed",
            "frame_id": "f-001",
            "points": [{"x": 1.0, "y": 0.5, "z": 2.0, "velocity": 0.1, "intensity": 0.9}],
        },
    })
    ack = await c.receive_json_from()
    assert ack["ack"] == "buffered"
    assert ack["frame_id"] == "f-001"

    mock_redis.xadd.assert_called_once()
    args, kwargs = mock_redis.xadd.call_args
    assert args[0] == "housafe:pointcloud:ingest"
    assert kwargs["maxlen"] == 1000

    await c.disconnect()


@pytest.mark.asyncio
async def test_vital_xadd(mock_redis, db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    await c.receive_json_from()

    await c.send_json_to({
        "kind": "vital",
        "payload": {
            "ts": 1700000000000,
            "radar_id": "d1",
            "room": "bed",
            "quiet": True,
            "resp_rate": 16.0,
            "heart_rate": 72.0,
            "quality": 0.85,
        },
    })
    ack = await c.receive_json_from()
    assert ack["ack"] == "buffered"

    mock_redis.xadd.assert_called_once()
    args, _ = mock_redis.xadd.call_args
    assert args[0] == "housafe:vital:ingest"

    await c.disconnect()


@pytest.mark.asyncio
async def test_heartbeat_updates_device(mock_redis, db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    await c.receive_json_from()

    await c.send_json_to({
        "kind": "heartbeat",
        "payload": {
            "ts": 1700000000000,
            "radar_id": "d1",
            "status": "online",
            "fw_version": "2.0.0",
        },
    })
    ack = await c.receive_json_from()
    assert ack["ack"] == "heartbeat"

    # verify device updated
    from devices.models import RadarDevice
    d = await RadarDevice.objects.aget(device_id="d1")
    assert d.online is True
    assert d.fw_version == "2.0.0"
    assert d.last_heartbeat is not None

    mock_redis.xadd.assert_not_called()
    await c.disconnect()


@pytest.mark.asyncio
async def test_invalid_frame_returns_error(mock_redis, db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    await c.receive_json_from()

    await c.send_json_to({"kind": "point_cloud", "payload": {"ts": 1, "radar_id": "d1", "room": "b", "frame_id": "x", "points": []}})
    msg = await c.receive_json_from()
    assert msg == {"error": "invalid"}

    await c.disconnect()


@pytest.mark.asyncio
async def test_unknown_kind_returns_error(mock_redis, db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    await c.receive_json_from()

    await c.send_json_to({"kind": "posture", "payload": {"ts": 1, "radar_id": "d1", "room": "b", "seq": 1, "posture": "walk", "confidence": 0.9}})
    msg = await c.receive_json_from()
    assert msg == {"error": "invalid"}

    await c.disconnect()
