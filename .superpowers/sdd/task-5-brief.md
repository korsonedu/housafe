### Task 5: 空间图结构学习

**Files:**
- Create: `ai/latent_space/__init__.py`
- Create: `ai/latent_space/graph.py`
- Create: `ai/tests/test_graph.py`

**Interfaces:**
- Consumes: `GraphNode`, `SpatialEdge` from Task 1
- Produces: `SpatialGraph` class — `update(centroids, timestamps)`, `locate(position) -> int`, `get_node_attrs(node_id) -> dict`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ai/latent_space
touch ai/latent_space/__init__.py
```

- [ ] **Step 2: Write tests**

`ai/tests/test_graph.py`:

```python
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
        # 区域 A: 床 (0.5, 0.5, 0.2)
        # 区域 B: 门口 (0.5, 3.0, 1.5)
        # 区域 C: 卫生间 (3.0, 0.5, 1.0)
        centroids = (
            [(0.5, 0.5, 0.2)] * 200 +   # A: 床，长时间停留
            [(0.5, 3.0, 1.5)] * 100 +   # B: 门口，中等停留
            [(3.0, 0.5, 1.0)] * 150     # C: 卫生间
        )
        pos, ts = make_trajectory(centroids, interval_ms=1000)
        g = SpatialGraph()
        g.update(pos, ts)

        assert len(g.nodes) == 3
        # 节点应按停留时长排序（节点0=停留最多）
        assert g.nodes[0].stay_ratio > g.nodes[1].stay_ratio

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
```

- [ ] **Step 3: Run test (verify failure)** then implement

- [ ] **Step 4: Implement `ai/latent_space/graph.py`**

```python
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
```

- [ ] **Step 5: Check scikit-learn availability**

```bash
cd /Users/eular/Desktop/housafe && python -c "import sklearn; print(sklearn.__version__)"
```

If missing, add `"scikit-learn>=1.3,<2"` to `backend/pyproject.toml`.

- [ ] **Step 6: Run tests**

```bash
cd /Users/eular/Desktop/housafe && python -m pytest ai/tests/test_graph.py -v
```
Expected: all passed

- [ ] **Step 7: Commit**

```bash
git add ai/latent_space/ ai/tests/test_graph.py
git commit -m "feat(ai): add SpatialGraph — auto-learn room topology from trajectories

HDBSCAN clustering on position centroids → nodes, co-occurrence stats
→ edges. Supports incremental update, save/load, locate, risk scoring.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

