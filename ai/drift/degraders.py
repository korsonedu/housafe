"""传感器退化模拟器 — Experiment 6。

模拟真实毫米波雷达的不完美：点稀疏、噪声、断线、多径。
"""

import numpy as np


def degrade_sparsity(points: np.ndarray, level: float) -> np.ndarray:
    """减少点数。level: 0=不变, 1=只剩 25%。"""
    keep_ratio = 1.0 - level * 0.75  # 1.0 → 0.25
    n = len(points)
    if n == 0:
        return points
    keep_n = max(1, int(n * keep_ratio))
    indices = np.random.choice(n, keep_n, replace=False)
    return points[indices]


def degrade_noise(points: np.ndarray, level: float) -> np.ndarray:
    """位置加高斯噪声。level: 0=不变, 1=σ=10cm。"""
    sigma = level * 0.1  # meters, 0 → 0.1m
    noise = np.random.randn(*points.shape).astype(np.float32) * sigma
    # 只加噪声到位置坐标 (x,y,z)，不加速度和强度
    result = points.copy()
    result[:, :3] += noise[:, :3]
    return result


def degrade_dropout(points: np.ndarray, level: float) -> np.ndarray | None:
    """模拟断线。level: 概率。返回 None 表示丢帧。"""
    if np.random.random() < level:
        return None
    return points


def degrade_multipath(points: np.ndarray, level: float) -> np.ndarray:
    """多径伪影：随机复制 + 位移一些点。level: 0=不变, 1=20% 点被复制。"""
    n = len(points)
    if n == 0:
        return points
    n_ghost = int(n * level * 0.2)
    if n_ghost == 0:
        return points
    ghost_indices = np.random.choice(n, n_ghost, replace=True)
    ghosts = points[ghost_indices].copy()
    ghosts[:, :3] += np.random.randn(n_ghost, 3).astype(np.float32) * 0.2
    return np.vstack([points, ghosts])


# ── 复合破化器 ────────────────────────────────────────

class PointCloudDegrader:
    """组合多个退化类型。"""

    def __init__(self,
                 sparsity: float = 0.0,
                 noise: float = 0.0,
                 dropout_prob: float = 0.0,
                 multipath: float = 0.0):
        self.sparsity = sparsity
        self.noise = noise
        self.dropout_prob = dropout_prob
        self.multipath = multipath

    def apply(self, points: np.ndarray) -> np.ndarray | None:
        """返回 degraded points 或 None（dropout）。"""
        if points is None or len(points) == 0:
            return points

        if self.dropout_prob > 0:
            points = degrade_dropout(points, self.dropout_prob)
            if points is None:
                return None

        if self.sparsity > 0:
            points = degrade_sparsity(points, self.sparsity)

        if self.noise > 0:
            points = degrade_noise(points, self.noise)

        if self.multipath > 0:
            points = degrade_multipath(points, self.multipath)

        return points


# ── 预定义退化等级 ────────────────────────────────────

DEGRADATION_LEVELS = {
    "sparsity": [
        ("sparsity_mild", PointCloudDegrader(sparsity=0.25)),
        ("sparsity_moderate", PointCloudDegrader(sparsity=0.5)),
        ("sparsity_severe", PointCloudDegrader(sparsity=0.75)),
    ],
    "noise": [
        ("noise_mild", PointCloudDegrader(noise=0.1)),    # σ=1cm
        ("noise_moderate", PointCloudDegrader(noise=0.5)),  # σ=5cm
        ("noise_severe", PointCloudDegrader(noise=1.0)),    # σ=10cm
    ],
    "dropout": [
        ("dropout_5s", PointCloudDegrader(dropout_prob=0.05)),
        ("dropout_30s", PointCloudDegrader(dropout_prob=0.3)),
        ("dropout_60s", PointCloudDegrader(dropout_prob=0.6)),
    ],
    "multipath": [
        ("multipath_mild", PointCloudDegrader(multipath=0.3)),
        ("multipath_moderate", PointCloudDegrader(multipath=0.6)),
        ("multipath_severe", PointCloudDegrader(multipath=1.0)),
    ],
}
