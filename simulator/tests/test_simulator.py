"""仿真系统集成测试"""
import tempfile
import os
import numpy as np
from simulator.types import FrameGroup, GroundTruth, VitalRecord, RoomConfig
from simulator.room_vitals import RoomModel, VitalSignsGen
from simulator.anomaly import AnomalyInjector
from simulator.scenario import ScenarioEngine
from simulator.validate import validate_simulation


class TestVitalSignsGen:
    def test_generate_returns_valid_vitals(self):
        gen = VitalSignsGen(resting_hr=72, resting_rr=16)
        v = gen.generate("sit", 0)
        assert 50 < v.heart_rate < 100, f"HR out of range: {v.heart_rate}"
        assert 10 < v.resp_rate < 25, f"RR out of range: {v.resp_rate}"
        assert 0 < v.quality <= 1.0, f"Quality out of range: {v.quality}"

    def test_activity_affects_vitals(self):
        gen = VitalSignsGen(resting_hr=72, resting_rr=16, seed=42)
        active_hrs = [gen.generate("walk", i * 0.1).heart_rate for i in range(50)]
        rest_hrs = [gen.generate("sit", i * 0.1).heart_rate for i in range(50)]
        assert sum(active_hrs) / len(active_hrs) > sum(rest_hrs) / len(rest_hrs), \
            "Walking HR should be higher than sitting HR"

    def test_rsa_oscillation(self):
        gen = VitalSignsGen(resting_hr=72, resting_rr=16, seed=42)
        samples = [gen.generate("sit", i * 0.1).heart_rate for i in range(100)]
        # Should see some variation due to RSA
        assert max(samples) - min(samples) > 2, "Expected RSA oscillation > 2 bpm"


class TestRoomModel:
    def test_apply_preserves_shape(self):
        cfg = RoomConfig("test", size=(4, 3, 2.8), radar_pos=(2, 1.5, 1.4))
        model = RoomModel(cfg)
        points = np.random.randn(20, 5).astype(np.float32)
        points[:, :3] += np.array([2, 1.5, 1.0])  # center of room
        result = model.apply(points)
        assert result is not None
        assert result.shape[1] == 5

    def test_apply_clips_outside_points(self):
        cfg = RoomConfig("test", size=(4, 3, 2.8), radar_pos=(2, 1.5, 1.4))
        model = RoomModel(cfg)
        points = np.array([[100, 100, 100, 0, 0]], dtype=np.float32)  # far outside
        result = model.apply(points)
        assert len(result) == 0, "Points outside room should be clipped"

    def test_apply_noise_adds_jitter(self):
        cfg = RoomConfig("test", size=(10, 10, 5), radar_pos=(5, 5, 2.5),
                         noise_sigma=0.02)
        model = RoomModel(cfg)
        points = np.tile(np.array([[5, 5, 1.0, 0, 0]], dtype=np.float32), (100, 1))
        result = model.apply(points)
        assert len(result) > 0
        # All points should have some noise applied
        assert not np.allclose(result[:, :3], points[:len(result), :3], atol=0.001)


class TestAnomalyInjector:
    def test_inject_fall_adds_ground_truth(self):
        frames = [
            FrameGroup(frame_id="f1", ts=1000, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
            FrameGroup(frame_id="f2", ts=1100, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
            FrameGroup(frame_id="f3", ts=1200, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
        ]
        injector = AnomalyInjector()
        injections = [{"at_s": 1.0, "type": "fall", "room": "living",
                       "params": {"transition_duration": 0.2},
                       "ground_truth": {"severity": "critical", "desc": "test fall"}}]
        result = injector.inject(frames, injections)
        assert len(result) == 3
        assert any(f.gt and f.gt.anomaly_type == "fall" for f in result)

    def test_inject_offline_clears_points(self):
        frames = [
            FrameGroup(frame_id="f1", ts=1000, device_id="d1", room="living",
                       points=np.random.randn(10, 5).astype(np.float32)),
            FrameGroup(frame_id="f2", ts=1100, device_id="d1", room="living",
                       points=np.random.randn(10, 5).astype(np.float32)),
        ]
        injector = AnomalyInjector()
        injections = [{"at_s": 1.0, "type": "offline", "room": "living",
                       "params": {"gap_duration_s": 1},
                       "ground_truth": {"severity": "critical"}}]
        result = injector.inject(frames, injections)
        assert result[0].points is None

    def test_inject_vital_anomaly_modifies_vitals(self):
        frames = [
            FrameGroup(frame_id="f1", ts=1000, device_id="d1", room="living",
                       vitals=VitalRecord(heart_rate=72, resp_rate=16, quality=0.95)),
            FrameGroup(frame_id="f2", ts=1100, device_id="d1", room="living",
                       vitals=VitalRecord(heart_rate=72, resp_rate=16, quality=0.95)),
        ]
        injector = AnomalyInjector()
        injections = [{"at_s": 1.0, "type": "vital_anomaly", "room": "living",
                       "params": {"ramp_duration_s": 0.2, "hr_range": (40, 130)},
                       "ground_truth": {"severity": "warning"}}]
        result = injector.inject(frames, injections)
        assert result[0].vitals is not None
        assert result[0].gt.anomaly_type == "vital_anomaly"


class TestTypes:
    def test_ground_truth_to_dict_excludes_none(self):
        gt = GroundTruth(
            frame_id="f1", ts=1000, posture="walk",
            posture_confidence=1.0, heart_rate_true=75.0,
            resp_rate_true=16.0, anomaly_type=None, anomaly_severity=None,
            scenario_name="test", generator="synthetic",
        )
        d = gt.to_dict()
        assert d["frame_id"] == "f1"
        assert d["posture"] == "walk"
        assert "anomaly_type" not in d  # None 被过滤

    def test_ground_truth_to_dict_includes_anomaly(self):
        gt = GroundTruth(
            frame_id="f2", ts=2000, posture="fall",
            posture_confidence=1.0, heart_rate_true=90.0,
            resp_rate_true=20.0, anomaly_type="fall",
            anomaly_severity="critical", anomaly_start=True,
            scenario_name="test", generator="synthetic",
        )
        d = gt.to_dict()
        assert d["anomaly_type"] == "fall"
        assert d["anomaly_start"] == 1  # converted to int


class TestScenarioEngine:
    def test_run_generates_frames_with_gt(self):
        yaml_content = '''
name: "test"
version: 1
device_id: rad_test
rooms:
  room1:
    size: [4.0, 3.0, 2.8]
    radar_pos: [2.0, 1.5, 1.4]
generator_defaults:
  method: synthetic
  fps: 10
timeline:
  - at: 0s
    room: room1
    activity: sit
    duration: 2s
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        engine = ScenarioEngine(yaml_path, speed=1.0)
        frames = engine.run()
        os.unlink(yaml_path)

        assert len(frames) == 20  # 2s * 10fps
        for f in frames:
            assert f.device_id == "rad_test"
            assert f.gt is not None
            assert f.gt.posture == "sit"
            assert f.vitals is not None

    def test_run_with_injection(self):
        yaml_content = '''
name: "test_inject"
version: 1
device_id: rad_test
rooms:
  room1:
    size: [4.0, 3.0, 2.8]
    radar_pos: [2.0, 1.5, 1.4]
generator_defaults:
  method: synthetic
  fps: 10
timeline:
  - at: 0s
    room: room1
    activity: walk
    duration: 5s
injections:
  - at: 2s
    type: fall
    room: room1
    params:
      height_drop_rate: 0.8
    ground_truth:
      severity: critical
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        engine = ScenarioEngine(yaml_path, speed=1.0)
        frames = engine.run()
        os.unlink(yaml_path)

        has_anomaly = any(f.gt and f.gt.anomaly_type == "fall" for f in frames)
        assert has_anomaly, "Should have fall anomaly injected"

    def test_heartbeat_insertion(self):
        yaml_content = '''
name: "test_hb"
version: 1
device_id: rad_test
rooms:
  room1:
    size: [4.0, 3.0, 2.8]
    radar_pos: [2.0, 1.5, 1.4]
generator_defaults:
  method: synthetic
  fps: 10
timeline:
  - at: 0s
    room: room1
    activity: sit
    duration: 3s
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        engine = ScenarioEngine(yaml_path, speed=1.0)
        frames = engine.run()
        os.unlink(yaml_path)

        # 30 frames: heartbeat at indices 9, 19, 29
        assert frames[9].heartbeat
        assert frames[19].heartbeat
        assert frames[29].heartbeat
        assert not frames[0].heartbeat


class TestValidate:
    def test_validate_passes_for_clean_data(self):
        frames = []
        rng = np.random.RandomState(42)
        for i in range(20):
            pts = np.zeros((15, 5), dtype=np.float32)
            pts[:, 0] = rng.uniform(-1, 1, 15)
            pts[:, 1] = rng.uniform(-1, 1, 15)
            pts[:, 2] = rng.uniform(0.5, 1.7, 15)  # standing height
            pts[:, 3] = rng.uniform(0, 1, 15)
            pts[:, 4] = rng.uniform(0.3, 1.0, 15)
            frames.append(FrameGroup(
                frame_id=f"f{i}", ts=1000 + i * 100, device_id="d1", room="r1",
                points=pts,
                gt=GroundTruth(
                    frame_id=f"f{i}", ts=1000 + i * 100,
                    posture="stand", posture_confidence=1.0,
                    heart_rate_true=72.0, resp_rate_true=16.0,
                    anomaly_type=None, anomaly_severity=None,
                    scenario_name="test", generator="synthetic",
                ),
            ))
        checks = validate_simulation(frames)
        assert checks["all_passed"], f"Should pass: {checks}"
