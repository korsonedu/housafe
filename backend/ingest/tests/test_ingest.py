import pytest
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


@pytest.mark.asyncio
async def test_bad_secret_closes(db):
    await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "WRONG"})
    msg = await c.receive_output()
    assert msg["type"] == "websocket.close"


@pytest.mark.asyncio
async def test_valid_event_persists(db):
    dev, _ = await _device(db)
    c = WebsocketCommunicator(application, "/ws/ingest")
    await c.connect()
    await c.send_json_to({"device_id": "d1", "secret": "s1"})
    await c.receive_json_from()  # ack
    await c.send_json_to({"kind": "posture", "payload": {"ts": 1700000000000, "radar_id": "d1", "room": "bed", "seq": 1, "posture": "walk", "confidence": 0.9}})
    await c.receive_json_from()  # stored ack
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 1
    await c.disconnect()
