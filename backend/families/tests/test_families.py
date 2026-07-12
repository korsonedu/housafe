import pytest
pytestmark = pytest.mark.django_db

def auth(api):
    api.post("/api/auth/register", {"username":"k","password":"pw12345678"}, format="json")
    t = api.post("/api/auth/login", {"username":"k","password":"pw12345678"}, format="json").data["access"]
    api.credentials(HTTP_AUTHORIZATION=f"Bearer {t}")

def test_create_family_and_list_own_only(api):
    auth(api)
    r = api.post("/api/families", {"name":"我家"}, format="json")
    assert r.status_code == 201
    fid = r.data["id"]
    r = api.get("/api/families")
    assert r.status_code == 200 and len(r.data) == 1
    r = api.post(f"/api/families/{fid}/contacts", {"name":"儿子","phone":"13800000000","order":1}, format="json")
    assert r.status_code == 201

def test_family_requires_auth(api):
    assert api.get("/api/families").status_code == 401
