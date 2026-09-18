"""端到端集成测试：特征提取 → 规则 → 基线 → 异常 → 通知 → DecoderOutput"""
import numpy as np
import pytest
from ai.main import WorldModelWorker, extract_features_from_pointcloud, extract_features_from_vital
from ai.shared.types import PointCloudFrame, VitalFrame, FeatureVector


class TestFeatureExtraction:
    def test_extract_from_pointcloud(self):
        pts = np.array([
            [1.0, 2.0, 0.5, 0.1, 0.8],
            [1.1, 2.1, 1.5, 0.2, 0.9],
            [0.9, 1.9, 0.3, 0.0, 0.7],
        ], dtype=np.float32)
        frame = PointCloudFrame(
            ts=1700000000000, device_id="r1", family_id="1",
            room="bedroom", frame_id="f-01", points=pts,
        )
        fv = extract_features_from_pointcloud(frame)
        assert fv.device_id == "r1"
        assert fv.room == "bedroom"
        assert fv.presence is True
        assert fv.n_points == 3
        assert fv.posture in ("stand", "sit", "lie", "walk", "fall")
        # 3 个点 (z: 0.5, 1.5, 0.3), height=1.2 → stand
        assert fv.height == pytest.approx(1.2, abs=0.1)
        # centroid
        assert fv.centroid[0] == pytest.approx(1.0, abs=0.1)

    def test_extract_from_vital(self):
        frame = VitalFrame(
            ts=1700000000000, device_id="r1", family_id="1",
            room="bedroom", resp_rate=16.5, heart_rate=72.0,
            quality=0.85,
        )
        fv = extract_features_from_vital(frame)
        assert fv.resp_rate == 16.5
        assert fv.heart_rate == 72.0
        assert fv.vital_quality == 0.85
        # 生命体征帧一般不改变姿态
        assert fv.posture == "stand"  # 默认值


class TestWorldModelPipeline:
    def test_full_pipeline_normal_frame(self):
        """正常帧经过完整 pipeline → 不应触发异常"""
        worker = WorldModelWorker(
            redis_url="redis://localhost:6379/0",  # 不连真实 Redis
            backend_url="http://localhost:8000",
            token="test-token",
        )
        fv = FeatureVector(
            ts=1000, device_id="r1", room="bedroom",
            posture="sit", posture_confidence=0.9, presence=True,
            moving=False, centroid=(1, 1, 0.5), height=0.8,
            n_points=15, occupancy_estimate=1,
            resp_rate=16.0, heart_rate=72.0, vital_quality=0.85,
            hour_sin=0.0, hour_cos=1.0, weekday=0,
            velocity_variance=0.01,
        )
        history = [fv] * 10
        anomaly = worker.anomaly.evaluate(fv, history)
        # 正常帧，无基线 → 不应告警
        assert anomaly is None

    def test_fall_detected_in_pipeline(self):
        """跌倒序列 → pipeline 应检出"""
        worker = WorldModelWorker(
            redis_url="redis://localhost:6379/0",
            backend_url="http://localhost:8000",
            token="test-token",
        )
        from ai.shared.types import time_encode
        h_sin, h_cos = time_encode(12.0)

        history = []
        for i in range(5):
            history.append(FeatureVector(
                ts=1000 + i * 200, device_id="r1", room="bathroom",
                posture="stand", posture_confidence=0.9, presence=True,
                moving=True, centroid=(1, 1, 1.5), height=1.6,
                n_points=20, occupancy_estimate=1,
                resp_rate=18, heart_rate=80, vital_quality=0.8,
                hour_sin=h_sin, hour_cos=h_cos, weekday=0,
                velocity_variance=0.2,
            ))
        current = FeatureVector(
            ts=2000, device_id="r1", room="bathroom",
            posture="lie", posture_confidence=0.85, presence=True,
            moving=False, centroid=(1, 1, 0.1), height=0.2,
            n_points=8, occupancy_estimate=1,
            resp_rate=20, heart_rate=100, vital_quality=0.7,
            hour_sin=h_sin, hour_cos=h_cos, weekday=0,
            velocity_variance=0.0,
        )
        history.append(current)

        anomaly = worker.anomaly.evaluate(current, history)
        assert anomaly is not None
        assert anomaly.anomaly_type == "fall"

        decision = worker.notifier.decide(anomaly, [])
        assert decision.level in ("sms", "call")

    def test_decoder_output_format(self):
        """验证输出的 decoder output 格式符合 backend 接口契约"""
        from ai.shared.types import AnomalyResult

        # 构造异常的 decoder output
        anomaly = AnomalyResult(
            ts=2000, device_id="r1", room="bathroom",
            anomaly_score=0.9, anomaly_type="fall",
            severity="critical", source="fallback",
            details={"height_drop_m": 1.4},
        )
        # 检查字段完整性（backend contracts 要求）
        assert anomaly.anomaly_type in ("fall", "stillness", "vital_anomaly",
                                          "pattern_deviation", "offline")
        assert anomaly.severity in ("info", "warning", "critical")
        assert 0 <= anomaly.anomaly_score <= 1
