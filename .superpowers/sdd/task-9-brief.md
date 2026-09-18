### Task 9: 报告 + 追踪 Decoders

**Files:**
- Create: `ai/decoders/report.py`
- Create: `ai/decoders/tracking.py`
- Create: `ai/tests/test_report_tracking.py`

**Interfaces:**
- Consumes: `FeatureVector` from Task 1, `SpatialGraph` from Task 5
- Produces: `ReportDecoder` — `generate_daily_summary(features) -> dict`
- Produces: `TrackingDecoder` — `build_trajectory(features) -> dict`, `heatmap(features, graph) -> dict`

- [ ] **Step 1: Write tests**

`ai/tests/test_report_tracking.py`:

```python
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
```

- [ ] **Step 2: Implement `ai/decoders/report.py`**

```python
"""报告 Decoder — 从隐状态轨迹生成结构化统计数据"""
from collections import Counter
from ai.shared.types import FeatureVector


class ReportDecoder:
    """MVP 版本：统计聚合。P2 接 LLM 生成自然语言报告。"""

    def generate_daily_summary(self, features: list[FeatureVector]) -> dict:
        """
        生成单日摘要。
        Args:
            features: 约 1440 帧（1/min 采样）的当天特征
        Returns:
            dict with posture_distribution, hours_active, avg_vitals, room_usage
        """
        if not features:
            return {
                "posture_distribution": {},
                "hours_active": 0,
                "avg_heart_rate": None,
                "avg_resp_rate": None,
                "room_usage": {},
                "n_frames": 0,
            }

        # 姿态分布
        posture_counts = Counter(f.posture for f in features)
        total = len(features)
        posture_dist = {k: round(v / total, 3) for k, v in posture_counts.items()}

        # 活跃时长（moving=True 且有走动）
        active_frames = sum(1 for f in features if f.moving or f.posture == "walk")
        hours_active = round(active_frames / max(total, 1) * 24, 1)

        # 平均生命体征
        hrs = [f.heart_rate for f in features if f.heart_rate is not None]
        rrs = [f.resp_rate for f in features if f.resp_rate is not None]
        avg_hr = round(sum(hrs) / len(hrs), 1) if hrs else None
        avg_rr = round(sum(rrs) / len(rrs), 1) if rrs else None

        # 房间使用
        room_counts = Counter(f.room for f in features)
        room_usage = {k: round(v / total, 3) for k, v in room_counts.items()}

        return {
            "posture_distribution": posture_dist,
            "hours_active": hours_active,
            "avg_heart_rate": avg_hr,
            "avg_resp_rate": avg_rr,
            "room_usage": room_usage,
            "n_frames": total,
        }

    def weekly_comparison(self, this_week: dict, last_week: dict) -> dict:
        """本周 vs 上周趋势对比"""
        deltas = {}
        for key in ("hours_active", "avg_heart_rate", "avg_resp_rate"):
            if this_week.get(key) is not None and last_week.get(key) is not None:
                deltas[key] = round(this_week[key] - last_week[key], 2)
        return deltas
```

- [ ] **Step 3: Implement `ai/decoders/tracking.py`**

```python
"""追踪 Decoder — 位置轨迹、热力图、区域停留统计"""
import numpy as np
from ai.shared.types import FeatureVector


class TrackingDecoder:
    """位置追踪。后续可接入图结构做房间级追踪。"""

    def build_trajectory(self, features: list[FeatureVector]) -> dict:
        """
        构建轨迹线。
        Returns: positions[[x,y,z]], timestamps, total_distance_m, duration_s
        """
        if not features:
            return {"positions": [], "timestamps": [], "total_distance_m": 0, "duration_s": 0}

        sorted_f = sorted(features, key=lambda f: f.ts)
        positions = [list(f.centroid) for f in sorted_f]
        timestamps = [f.ts for f in sorted_f]

        total_dist = 0.0
        for i in range(1, len(positions)):
            p1, p2 = np.array(positions[i - 1]), np.array(positions[i])
            total_dist += float(np.linalg.norm(p2 - p1))

        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0 if len(timestamps) > 1 else 0

        return {
            "positions": positions,
            "timestamps": timestamps,
            "total_distance_m": round(total_dist, 2),
            "duration_s": round(duration_s, 1),
        }

    def heatmap(self, features: list[FeatureVector],
                grid_size: float = 0.5, graph=None) -> dict:
        """
        2D 热力图（xy 平面）。
        Returns: grid (2D array), x_edges, y_edges, peak_region
        """
        if not features:
            return {"grid": np.zeros((1, 1)), "x_edges": [], "y_edges": [], "peak_region": None}

        xs = [f.centroid[0] for f in features]
        ys = [f.centroid[1] for f in features]

        if not xs:
            return {"grid": np.zeros((1, 1)), "x_edges": [], "y_edges": [], "peak_region": None}

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_bins = max(int((x_max - x_min) / grid_size), 2)
        y_bins = max(int((y_max - y_min) / grid_size), 2)

        grid, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_bins, y_bins])

        peak_idx = np.unravel_index(grid.argmax(), grid.shape)
        peak_center = (
            round(float((x_edges[peak_idx[0]] + x_edges[peak_idx[0] + 1]) / 2), 2),
            round(float((y_edges[peak_idx[1]] + y_edges[peak_idx[1] + 1]) / 2), 2),
        )

        return {
            "grid": grid,
            "x_edges": [round(float(e), 2) for e in x_edges],
            "y_edges": [round(float(e), 2) for e in y_edges],
            "peak_region": peak_center,
        }
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_report_tracking.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ai/decoders/report.py ai/decoders/tracking.py ai/tests/test_report_tracking.py
git commit -m "feat(ai): add ReportDecoder + TrackingDecoder

Daily summary stats, weekly comparison, trajectory builder, 2D heatmap.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

