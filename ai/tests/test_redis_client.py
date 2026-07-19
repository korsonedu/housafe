"""测试 Redis Stream 消息解析（不需要真实 Redis）"""
import json
import numpy as np
from ai.shared.redis_client import parse_pointcloud_message, parse_vital_message


def test_parse_pointcloud_message():
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"frame_id": b"f-0042",
        b"n_points": b"3",
        b"points": json.dumps([
            [1.0, 2.0, 0.5, 0.1, 0.8],
            [1.1, 2.1, 1.5, 0.2, 0.9],
            [0.9, 1.9, 1.0, 0.0, 0.7],
        ]).encode(),
    }
    frame = parse_pointcloud_message(data)
    assert frame.device_id == "rad_01"
    assert frame.family_id == "1"
    assert frame.room == "bedroom"
    assert frame.ts == 1700000000000
    assert frame.frame_id == "f-0042"
    assert isinstance(frame.points, np.ndarray)
    assert frame.points.shape == (3, 5)
    assert frame.points.dtype == np.float32
    # 验证点坐标
    assert frame.points[0, 0] == 1.0  # x
    assert frame.points[1, 2] == 1.5  # z


def test_parse_vital_message():
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"resp_rate": b"16.5",
        b"heart_rate": b"72.0",
        b"quality": b"0.85",
    }
    frame = parse_vital_message(data)
    assert frame.device_id == "rad_01"
    assert frame.family_id == "1"
    assert frame.resp_rate == 16.5
    assert frame.heart_rate == 72.0
    assert frame.quality == 0.85


def test_parse_vital_message_none_values():
    """部分生命体征字段可能为空"""
    data = {
        b"family_id": b"1",
        b"device_id": b"rad_01",
        b"ts": b"1700000000000",
        b"room": b"bedroom",
        b"resp_rate": b"",
        b"heart_rate": b"",
        b"quality": b"0.5",
    }
    frame = parse_vital_message(data)
    assert frame.resp_rate is None
    assert frame.heart_rate is None
    assert frame.quality == 0.5
