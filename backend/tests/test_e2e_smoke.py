import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.mark.asyncio
async def test_full_chain_device_to_query(db):
    from django.contrib.auth.models import User
    from families.models import Family
    from devices.models import RadarDevice
    u = await User.objects.acreate_user("k","","pw12345678")
    fam = await Family.objects.acreate(name="f", owner=u)
    await RadarDevice.objects.acreate(device_id="d1", secret="s1", family=fam, room="bed")
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id":"d1","secret":"s1"}); await ing.receive_json_from()
    for seq,p in [(1,"stand"),(2,"walk"),(3,"lie")]:
        await ing.send_json_to({"kind":"posture","payload":{"ts":1700000000000+seq,"radar_id":"d1","room":"bed","seq":seq,"posture":p,"confidence":0.9}})
        await ing.receive_json_from()
    from events.models import PostureRow
    assert await PostureRow.objects.filter(device_id="d1").acount() == 3
    await ing.disconnect()
