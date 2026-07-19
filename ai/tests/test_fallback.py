"""测试 5 条确定性规则：跌倒/静止/生命体征阈值/设备离线/规则开关"""
import numpy as np
from ai.shared.types import FeatureVector, AnomalyResult
from ai.fallback.rules import FallbackEngine


def make_fv(ts=1000, posture="stand", height=1.5, moving=True,
            centroid=(1.0, 2.0, 1.0), room="bedroom", resp_rate=16.0,
            heart_rate=72.0, vel_var=0.1, **kw):
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=moving,
        centroid=centroid, height=height, n_points=15,
        occupancy_estimate=1, resp_rate=resp_rate, heart_rate=heart_rate,
        vital_quality=0.85, hour_sin=0.0, hour_cos=1.0,
        weekday=0, velocity_variance=vel_var, **kw
    )


class TestFallDetection:
    def test_height_drop_triggers_fall(self):
        engine = FallbackEngine()
        # 模拟跌倒序列：站立(height=1.6) → 3帧后躺下(height=0.2)
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6, moving=True)
            for i in range(5)
        ]
        # 跌倒帧
        current = make_fv(ts=2000, posture="lie", height=0.2, moving=False,
                          centroid=(1.0, 2.0, 0.1))
        history.append(current)

        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 1
        assert falls[0].severity == "critical"
        assert falls[0].anomaly_score > 0.8

    def test_slow_lie_down_no_fall(self):
        """缓慢躺下(height 渐变)不应触发跌倒"""
        engine = FallbackEngine()
        history = [
            make_fv(ts=1000 + i*500, posture="stand", height=1.6 - i*0.1, moving=True)
            for i in range(10)
        ]
        current = make_fv(ts=6000, posture="lie", height=0.6, moving=False)
        history.append(current)
        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 0


class TestStillnessDetection:
    def test_stillness_in_bathroom_triggers(self):
        engine = FallbackEngine()
        # 60 帧 (~2min) 卫生间静止=lie
        history = [
            make_fv(ts=1000 + i*2000, posture="lie", room="bathroom",
                    height=0.3, moving=False, centroid=(1,1,0.1), vel_var=0.0)
            for i in range(60)
        ]
        current = history[-1]
        results = engine.evaluate(current, history)
        stills = [r for r in results if r.anomaly_type == "stillness"]
        assert len(stills) >= 1

    def test_stillness_in_bedroom_ignored(self):
        """卧室长时间静止 = 睡眠，不告警"""
        engine = FallbackEngine()
        history = [
            make_fv(ts=1000 + i*2000, posture="lie", room="bedroom",
                    height=0.3, moving=False, centroid=(1,1,0.1), vel_var=0.0)
            for i in range(60)
        ]
        current = history[-1]
        results = engine.evaluate(current, history)
        stills = [r for r in results if r.anomaly_type == "stillness"]
        assert len(stills) == 0  # 卧室躺着正常


class TestVitalThresholds:
    def test_high_heart_rate(self):
        engine = FallbackEngine()
        fv = make_fv(heart_rate=140, resp_rate=20)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 1
        assert vitals[0].severity in ("warning", "critical")

    def test_normal_vitals_no_alert(self):
        engine = FallbackEngine()
        fv = make_fv(heart_rate=72, resp_rate=16)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 0

    def test_no_vitals_data(self):
        """生命体征数据缺失 → 不告警（可能是安静态条件未满足）"""
        engine = FallbackEngine()
        fv = make_fv(resp_rate=None, heart_rate=None)
        results = engine.evaluate(fv, [fv])
        vitals = [r for r in results if r.anomaly_type == "vital_anomaly"]
        assert len(vitals) == 0


class TestRuleToggles:
    def test_disable_fall_rule(self):
        engine = FallbackEngine(enable_fall=False)
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6)
            for i in range(5)
        ]
        current = make_fv(ts=2000, posture="lie", height=0.2)
        history.append(current)
        results = engine.evaluate(current, history)
        falls = [r for r in results if r.anomaly_type == "fall"]
        assert len(falls) == 0
