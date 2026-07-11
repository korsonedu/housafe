import pytest
pytestmark = pytest.mark.django_db

def test_register_then_login(api):
    r = api.post("/api/auth/register", {"username":"kid","password":"pw12345678"}, format="json")
    assert r.status_code == 201
    r = api.post("/api/auth/login", {"username":"kid","password":"pw12345678"}, format="json")
    assert r.status_code == 200 and "access" in r.data

def test_login_wrong_password(api):
    api.post("/api/auth/register", {"username":"kid","password":"pw12345678"}, format="json")
    r = api.post("/api/auth/login", {"username":"kid","password":"wrong"}, format="json")
    assert r.status_code == 401
