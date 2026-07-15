import pytest


def pytest_configure():
    """Override channel layers to use in-memory layer for tests."""
    from django.conf import settings
    settings.CHANNEL_LAYERS = {
        "default": {
            "BACKEND": "channels.layers.InMemoryChannelLayer",
        },
    }


@pytest.fixture
def api(db):
    from rest_framework.test import APIClient
    return APIClient()
