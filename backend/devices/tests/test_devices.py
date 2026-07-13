import pytest
pytestmark = pytest.mark.django_db

def auth_and_family(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")
    return api.post("/api/families", {"name":"我家"}, format="json").data["id"]

def test_bind_returns_credentials_and_lists(api):
    fid = auth_and_family(api)
    r = api.post(f"/api/families/{fid}/devices/bind", {"room":"bedroom"}, format="json")
    assert r.status_code == 201 and r.data["device_id"] and r.data["secret"]
    r = api.get(f"/api/families/{fid}/devices")
    assert len(r.data) == 1 and r.data[0]["room"] == "bedroom"

def test_verify_credentials():
    from devices.models import RadarDevice
    from families.models import Family
    from django.contrib.auth.models import User
    u = User.objects.create_user("k","","pw12345678")
    fam = Family.objects.create(name="f", owner=u)
    d = RadarDevice.objects.create(device_id="d1", secret="s1", family=fam, room="bath")
    assert RadarDevice.verify("d1","s1") == d
    assert RadarDevice.verify("d1","bad") is None
