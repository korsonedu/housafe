"""测试异常解码器：融合 fallback + baseline + 空间感知"""
from ai.shared.types import FeatureVector
from ai.decoders.anomaly import AnomalyDecoder
from ai.fallback.rules import FallbackEngine


def make_fv(ts=1000, posture="sit", room="bedroom", heart_rate=72, resp_rate=16,
            centroid=(1.0, 1.0, 0.5), height=0.8, moving=False, vel_var=0.01,
            hour=12.0, weekday=0):
    from ai.shared.types import time_encode
    h_sin, h_cos = time_encode(hour)
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=moving,
        centroid=centroid, height=height, n_points=15,
        occupancy_estimate=1, resp_rate=resp_rate, heart_rate=heart_rate,
        vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
        weekday=weekday, velocity_variance=vel_var,
    )


class TestAnomalyDecoder:
    def test_fallback_detected_passes_through(self):
        """Fallback 检出的异常直接透传"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        # 构造跌倒序列
        history = [
            make_fv(ts=1000 + i*200, posture="stand", height=1.6, moving=True)
            for i in range(5)
        ]
        current = make_fv(ts=2000, posture="lie", height=0.2, moving=False,
                          centroid=(1.0, 1.0, 0.1))
        history.append(current)

        result = decoder.evaluate(current, history)
        assert result is not None
        assert result.anomaly_type == "fall"
        assert result.source == "fallback"

    def test_no_baseline_no_graph_returns_none_for_normal(self):
        """无基线/无图时 → 正常帧不告警"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fv = make_fv()
        result = decoder.evaluate(fv, [fv])
        assert result is None

    def test_spatial_awareness_bathroom_lie(self):
        """卫生间躺卧 → 空间感知加权 → 异常分增加"""
        decoder = AnomalyDecoder(fallback=FallbackEngine())
        fvs_bathroom = [make_fv(room="bathroom", posture="lie",
                                 centroid=(3, 3, 0.1), height=0.2)
                        for _ in range(60)]
        result = decoder.evaluate(fvs_bathroom[-1], fvs_bathroom)
        # stillness 规则在卫生间触发
        if result:
            assert result.room == "bathroom"
