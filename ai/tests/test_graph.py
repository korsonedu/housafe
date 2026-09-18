"""测试空间图：聚类、节点属性、定位、周期性更新"""
import numpy as np
from ai.latent_space.graph import SpatialGraph


def make_trajectory(
    centroids: list[tuple[float, float, float]],
    interval_ms: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """生成轨迹数据"""
    n = len(centroids)
    ts = np.arange(n) * interval_ms
    pos = np.array(centroids, dtype=np.float32)
    return pos, ts


class TestSpatialGraph:
    def test_builds_nodes_from_distinct_regions(self):
        """3 个明显分离的区域 + 各停留足够时长 → 应产生 3 个节点"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +   # A: 床，长时间停留
            [(0.5, 3.0, 1.5)] * 100 +   # B: 门口，中等停留
            [(3.0, 0.5, 1.0)] * 150     # C: 卫生间
        )
        pos, ts = make_trajectory(centroids, interval_ms=1000)
        g = SpatialGraph()
        g.update(pos, ts)

        assert len(g.nodes) == 3
        # 按 stay_ratio 排序验证分布合理性
        sorted_nodes = sorted(g.nodes.values(), key=lambda n: n.stay_ratio, reverse=True)
        assert sorted_nodes[0].stay_ratio > sorted_nodes[1].stay_ratio

    def test_filters_transient_points(self):
        """短暂经过的区域不应成为独立节点"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +   # A: 床
            [(2.0, 2.0, 1.0)] * 3 +     # 路过（3帧）
            [(3.0, 3.0, 1.5)] * 150     # C: 另一区域
        )
        pos, ts = make_trajectory(centroids, interval_ms=1000)
        g = SpatialGraph(min_stay_frames=60)
        g.update(pos, ts)
        assert len(g.nodes) == 2  # 路过点被过滤

    def test_locate_returns_correct_node(self):
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +
            [(3.0, 3.0, 1.5)] * 150
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        n1 = g.locate((0.6, 0.4, 0.3))  # 靠近区域 A
        n2 = g.locate((3.1, 2.9, 1.4))  # 靠近区域 B
        assert n1 != n2
        assert n1 is not None
        assert n2 is not None

    def test_builds_edges_from_transitions(self):
        """A→B→A→B 频繁 → 应产生边"""
        centroids = []
        for _ in range(10):
            centroids += [(0.5, 0.5, 0.2)] * 100   # A
            centroids += [(3.0, 3.0, 1.5)] * 100   # B
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        assert len(g.edges) >= 2  # A→B, B→A

    def test_incremental_update(self):
        """增量更新不应重建节点"""
        pos_a = np.tile(np.array([0.5, 0.5, 0.2]), (200, 1))
        ts_a = np.arange(200) * 1000
        g = SpatialGraph()
        g.update(pos_a, ts_a)
        n_before = len(g.nodes)

        # 次日新数据
        pos_b = np.tile(np.array([3.0, 3.0, 1.5]), (150, 1))
        ts_b = np.arange(150) * 1000 + 86400_000
        g.update(pos_b, ts_b)
        assert len(g.nodes) >= n_before  # 不会丢掉旧节点

    def test_node_risk_score(self):
        """区域高度变化大 → risk_score 更高（更可能是浴室等危险区域）"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +     # A: 低高度、低变化
            [(3.0, 3.0, 1.5)] * 50 +      # B: 中高度
            [(3.0, 3.0, 0.1)] * 50 +      # B: 低高度（高度变化大）
            [(3.0, 3.0, 1.6)] * 50        # B: 高高度
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        node_a = g.locate((0.5, 0.5, 0.2))
        node_b = g.locate((3.0, 3.0, 1.5))
        assert g.nodes[node_a].risk_score < g.nodes[node_b].risk_score

    def test_save_and_load(self, tmp_path):
        """持久化 + 恢复"""
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +
            [(3.0, 3.0, 1.5)] * 150
        )
        pos, ts = make_trajectory(centroids)
        g = SpatialGraph()
        g.update(pos, ts)

        path = str(tmp_path / "graph.pkl")
        g.save(path)

        g2 = SpatialGraph.load(path)
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.edges) == len(g.edges)
