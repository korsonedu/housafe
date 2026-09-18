"""空间图结构 — 从轨迹数据自动学习房间功能区域拓扑"""
import pickle
import numpy as np
from sklearn.cluster import HDBSCAN
from collections import defaultdict
from ai.shared.types import GraphNode, SpatialEdge


class SpatialGraph:
    """从人的位置轨迹中自动学出的空间拓扑图"""

    def __init__(self, min_stay_frames: int = 60):
        """
        Args:
            min_stay_frames: 成为独立节点的最少停留帧数（过滤路过）
                默认 60 帧 ≈ 1min @ 1fps（样本点）或 ≈ 3s @ 20fps
        """
        self.nodes: dict[int, GraphNode] = {}
        self.edges: dict[tuple[int, int], SpatialEdge] = {}
        self._clusterer = HDBSCAN(min_cluster_size=max(min_stay_frames, 5))
        self._node_centroids: np.ndarray | None = None  # (K, 3) 用于快速 locate
        self._total_frames: int = 0
        self._point_buffer: list[tuple[np.ndarray, int]] = []  # [(xyz, ts)]

    def update(self, positions: np.ndarray, timestamps: np.ndarray):
        """
        增量更新图结构。
        Args:
            positions: (N, 3) float32, 每帧人的质心坐标
            timestamps: (N,) int64, 每帧的时间戳 (ms)
        """
        if len(positions) == 0:
            return

        # 累积缓冲
        for i in range(len(positions)):
            self._point_buffer.append((positions[i], timestamps[i]))
        self._total_frames += len(positions)

        # 限制缓冲区大小（保留最近 14 天，按 1fps 采样 ≈ 1.2M 点）
        max_buffer = 1_200_000
        if len(self._point_buffer) > max_buffer:
            self._point_buffer = self._point_buffer[-max_buffer:]

        # 重新聚类（HDBSCAN 不支持增量，替换为全量；数据量小时可接受）
        self._recluster()

    def _recluster(self):
        """从缓冲数据重新聚类"""
        if len(self._point_buffer) < 10:
            return

        all_pos = np.array([p[0] for p in self._point_buffer])
        all_ts = np.array([p[1] for p in self._point_buffer])

        # HDBSCAN 聚类
        labels = self._clusterer.fit_predict(all_pos)

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 噪声点不建节点

        if len(unique_labels) < 1:
            return

        # 构建节点
        new_nodes = {}
        for label in unique_labels:
            mask = labels == label
            cluster_pos = all_pos[mask]
            cluster_ts = all_ts[mask]
            centroid = tuple(cluster_pos.mean(axis=0).astype(np.float32))
            avg_height = float(cluster_pos[:, 2].mean())
            stay_ratio = len(cluster_pos) / len(all_pos)

            # 24 时段分布
            hourly_prob = np.zeros(24)
            hours = (cluster_ts / 3600_000).astype(int) % 24
            for h in hours:
                hourly_prob[h] += 1
            hourly_prob = (hourly_prob / hourly_prob.sum()).tolist()

            # 危险等级：高度方差标准化
            height_std = float(cluster_pos[:, 2].std())
            risk_score = min(height_std / 1.0, 1.0)  # 归一化

            node = GraphNode(
                node_id=int(label),
                centroid=centroid,
                avg_height=avg_height,
                stay_ratio=stay_ratio,
                hourly_prob=hourly_prob,
                risk_score=risk_score,
            )
            new_nodes[int(label)] = node

        self.nodes = new_nodes

        # 更新 centroid 缓存
        if new_nodes:
            self._node_centroids = np.array([
                n.centroid for n in new_nodes.values()
            ], dtype=np.float32)

        # 构建边（共现统计）
        self._build_edges(labels, all_pos, all_ts)

    def _build_edges(self, labels: np.ndarray, positions: np.ndarray, timestamps: np.ndarray):
        """从标签序列统计节点间转移"""
        transitions = defaultdict(lambda: {"count": 0, "total_time": 0.0})

        for i in range(len(labels) - 1):
            a, b = labels[i], labels[i + 1]
            if a == -1 or b == -1 or a == b:
                continue
            dt = (timestamps[i + 1] - timestamps[i]) / 1000.0
            if dt < 0 or dt > 300:  # 超过 5 分钟不算连续过渡
                continue
            key = (int(a), int(b))
            transitions[key]["count"] += 1
            transitions[key]["total_time"] += dt

        total = sum(t["count"] for t in transitions.values())
        self.edges = {}
        for (a, b), t in transitions.items():
            if total > 0 and t["count"] / total >= 0.005:  # 过滤低频过渡
                self.edges[(a, b)] = SpatialEdge(
                    from_node=a, to_node=b,
                    transition_freq=t["count"] / total,
                    avg_transition_s=t["total_time"] / t["count"],
                )

    def locate(self, position: tuple[float, float, float]) -> int | None:
        """给定位置 → 返回最近的 node_id"""
        if self._node_centroids is None or len(self._node_centroids) == 0:
            return None
        pos = np.array(position, dtype=np.float32)
        dists = np.linalg.norm(self._node_centroids - pos, axis=1)
        return int(np.argmin(dists))

    def get_node_attrs(self, node_id: int) -> dict:
        """获取节点属性"""
        if node_id not in self.nodes:
            return {}
        n = self.nodes[node_id]
        return {
            "node_id": n.node_id,
            "centroid": n.centroid,
            "avg_height": n.avg_height,
            "stay_ratio": n.stay_ratio,
            "hourly_prob": n.hourly_prob,
            "risk_score": n.risk_score,
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)

    @classmethod
    def load(cls, path: str) -> "SpatialGraph":
        g = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        g.nodes = data["nodes"]
        g.edges = data["edges"]
        if g.nodes:
            g._node_centroids = np.array([n.centroid for n in g.nodes.values()], dtype=np.float32)
        return g
