import pytest
pytestmark = pytest.mark.django_db
from housafe_contracts.events import parse_event
from events.store import store_event

def _setup(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    fid = api.post("/api/families", {"name":"f"}, format="json").data["id"]
    did = api.post(f"/api/families/{fid}/devices/bind", {"room":"bed"}, format="json").data["device_id"]
    return fid, did

def test_today_returns_latest_per_room(api):
    fid, did = _setup(api)
    for seq,p in [(1,"stand"),(2,"walk")]:
        store_event(did,"posture",parse_event("posture",
            {"ts":1700000000000+seq,"radar_id":did,"room":"bed","seq":seq,"posture":p,"confidence":0.9}),ts_recv=0)
    r = api.get(f"/api/families/{fid}/today")
    assert r.status_code == 200
    rooms = r.data["rooms"]
    assert len(rooms) == 1
    assert rooms[0]["room_name"] == "bed"
    assert rooms[0]["posture"] == "walk"
