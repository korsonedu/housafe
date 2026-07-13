import pytest
pytestmark = pytest.mark.django_db
from events.store import store_event
from events.models import PostureRow
from housafe_contracts.events import parse_event

def test_store_and_idempotent():
    m = parse_event("posture", {"ts":1700000000000,"radar_id":"d1","room":"bed","seq":1,"posture":"walk","confidence":0.9})
    store_event("d1","posture",m,ts_recv=1700000000050)
    store_event("d1","posture",m,ts_recv=1700000000060)  # 重复 seq
    rows = PostureRow.objects.filter(device_id="d1")
    assert rows.count() == 1 and rows.first().ts == 1700000000000
