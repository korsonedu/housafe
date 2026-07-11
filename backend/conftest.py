import pytest

@pytest.fixture
def api(db):
    from rest_framework.test import APIClient
    return APIClient()
