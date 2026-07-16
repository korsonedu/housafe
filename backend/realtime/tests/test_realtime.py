import pytest
from channels.testing import WebsocketCommunicator
from housafe.asgi import application

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.mark.asyncio
async def test_ingest_to_app_flow(db):
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
    ing = WebsocketCommunicator(application, "/ws/ingest")
    await ing.connect()
    await ing.send_json_to({"device_id": "d1", "secret": "s1"})
    await ing.receive_json_from()
    await ing.send_json_to(
        {
            "kind": "posture",
            "payload": {
                "ts": 1,
                "radar_id": "d1",
                "room": "bed",
                "seq": 1,
                "posture": "walk",
                "confidence": 0.9,
            },
        }
    )
    pushed = await app.receive_json_from()
    assert pushed["kind"] == "posture" and pushed["payload"]["room"] == "bed"
    await app.disconnect()
    await ing.disconnect()
