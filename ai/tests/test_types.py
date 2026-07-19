"""验证 FeatureVector 构造、时间编码计算、AnomalyResult 字段约束"""
import math
import pytest
import numpy as np
from ai.shared.types import (
    FeatureVector, AnomalyResult, NotificationDecision,
    time_encode, posture_to_onehot,
)


def test_time_encode_noon():
    """正午 12:00 → sin≈0, cos≈-1 (或 1 取决于角度定义)"""
    s, c = time_encode(12.0)
    # 12:00 = π in sin/cos cycle (0=midnight, 12h=π)
    assert abs(s) < 1e-9
    assert c == pytest.approx(-1.0, abs=1e-9)


def test_time_encode_midnight():
    s, c = time_encode(0.0)
    assert abs(s) < 1e-9
    assert c == pytest.approx(1.0, abs=1e-9)


def test_time_encode_symmetry():
    """6:00 和 18:00 的 cos 应该相同（循环对称）"""
    s6, c6 = time_encode(6.0)
    s18, c18 = time_encode(18.0)
    assert c6 == pytest.approx(c18, abs=1e-9)
    assert s6 == pytest.approx(-s18, abs=1e-9)


def test_posture_onehot():
    assert posture_to_onehot("stand") == [1, 0, 0, 0, 0]
    assert posture_to_onehot("lie")   == [0, 0, 1, 0, 0]
    assert posture_to_onehot("fall")  == [0, 0, 0, 0, 1]
    # unknown → all zeros
    assert posture_to_onehot("unknown") == [0, 0, 0, 0, 0]


def test_feature_vector_to_array():
    fv = FeatureVector(
        ts=1000, device_id="r1", room="bedroom",
        posture="lie", posture_confidence=0.9, presence=True, moving=False,
        centroid=(1.0, 2.0, 0.3), height=0.2, n_points=12, occupancy_estimate=1,
        resp_rate=16.0, heart_rate=72.0, vital_quality=0.85,
        hour_sin=0.0, hour_cos=1.0, weekday=0,
        velocity_variance=0.01,
    )
    arr = fv.to_array()
    # 5 (posture onehot) + 3 (confidence/moving/presence) + 3 (centroid)
    # + 1 (height) + 1 (n_points) + 1 (occupancy)
    # + 3 (vitals) + 2 (time) + 7 (weekday) + 1 (vel_var) = 27
    assert arr.shape == (27,)
    assert arr.dtype == np.float64
    # posture onehot: lie = [0,0,1,0,0]
    assert arr[2] == 1.0


def test_anomaly_result_fields():
    ar = AnomalyResult(
        ts=1000, device_id="r1", room="bathroom",
        anomaly_score=0.85, anomaly_type="fall",
        severity="critical", source="fallback",
        details={"height_drop_m": 1.2},
    )
    assert 0 <= ar.anomaly_score <= 1


def test_notification_decision():
    nd = NotificationDecision(
        level="sms", reason="fall detected in bathroom",
        anomaly=AnomalyResult(
            ts=1000, device_id="r1", room="bathroom",
            anomaly_score=0.95, anomaly_type="fall",
            severity="critical", source="fallback", details={},
        ),
    )
    assert nd.level in ("none", "push", "sms", "call")
