import pytest
from pydantic import ValidationError
from housafe_contracts.events import parse_event, PostureEvent, POSTURES

def test_posture_valid():
    e = parse_event("posture", {"ts":1700000000000,"radar_id":"r1","room":"bedroom","seq":5,"posture":"walk","confidence":0.9})
    assert isinstance(e, PostureEvent) and e.posture == "walk"

def test_posture_rejects_bad_enum():
    with pytest.raises(ValidationError):
        parse_event("posture", {"ts":1,"radar_id":"r1","room":"b","seq":1,"posture":"jump","confidence":0.5})

def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        parse_event("nope", {})

def test_postures_frozen():
    assert POSTURES == ("stand","sit","lie","walk","fall")
