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
