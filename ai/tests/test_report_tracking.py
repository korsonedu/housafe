"""报告和追踪解码器测试"""
from ai.shared.types import FeatureVector
from ai.decoders.report import ReportDecoder
from ai.decoders.tracking import TrackingDecoder


def make_fv(ts=1000, posture="sit", centroid=(1.0, 1.0, 0.5),
            room="bedroom", **kw):
    from ai.shared.types import time_encode
    h_sin, h_cos = time_encode(12.0)
    return FeatureVector(
        ts=ts, device_id="r1", room=room, posture=posture,
        posture_confidence=0.9, presence=True, moving=False,
        centroid=centroid, height=0.8, n_points=15,
        occupancy_estimate=1, resp_rate=16.0, heart_rate=72.0,
        vital_quality=0.85, hour_sin=h_sin, hour_cos=h_cos,
        weekday=0, velocity_variance=0.01, **kw
    )


class TestReportDecoder:
    def test_daily_summary(self):
        """24h 特征 → 活动摘要"""
        decoder = ReportDecoder()
        fvs = [
            make_fv(ts=3600_000 * i, posture="lie", room="bedroom")
            for i in range(8)  # 8h 睡眠
        ] + [
            make_fv(ts=3600_000 * (8 + i), posture="sit", room="living_room")
            for i in range(4)  # 4h 坐着
        ] + [
            make_fv(ts=3600_000 * (12 + i), posture="walk", room="kitchen")
            for i in range(2)  # 2h 活动
        ]

        summary = decoder.generate_daily_summary(fvs)
        assert "posture_distribution" in summary
        assert "hours_active" in summary
        assert summary["posture_distribution"]["lie"] > 0.3  # ~57% lie
        assert summary["hours_active"] > 0

    def test_empty_input(self):
        decoder = ReportDecoder()
        summary = decoder.generate_daily_summary([])
        assert summary["hours_active"] == 0


class TestTrackingDecoder:
    def test_trajectory(self):
        decoder = TrackingDecoder()
        fvs = [
            make_fv(ts=i*2000, centroid=(float(i)*0.1, 0.0, 1.0))
            for i in range(10)
        ]
        traj = decoder.build_trajectory(fvs)
        assert len(traj["positions"]) == 10
        assert "total_distance_m" in traj
        assert "duration_s" in traj

    def test_heatmap(self):
        decoder = TrackingDecoder()
        fvs = [
            make_fv(ts=i*2000, centroid=(1.0, 1.0, 0.5)) for i in range(50)
        ] + [
            make_fv(ts=100_000 + i*2000, centroid=(3.0, 3.0, 1.5))
            for i in range(30)
        ]
        hm = decoder.heatmap(fvs)
        assert "grid" in hm
        assert hm["grid"].shape[0] > 0
