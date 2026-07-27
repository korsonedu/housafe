"""点云生成器 — GMM 合成 + 3DPCHM 回放"""
import numpy as np
from collections import defaultdict
from simulator.calibrate import GMMParams, load_params, ACTION_NAMES

RANDOM_SEED = 42

# 回退：无校准参数时每姿态的 z 范围和点云数
_FALLBACK_Z = {
    "stand": (0.0, 1.7), "sit": (0.0, 1.0), "lie": (0.0, 0.3),
    "walk": (0.0, 1.8), "fall": (0.0, 0.2), "squat": (0.0, 1.1),
}


class SyntheticGenerator:
    """基于校准 GMM 参数的运行时点云生成"""

    def __init__(self, params: dict[str, GMMParams] | None = None,
                 params_path: str | None = None):
        if params is not None:
            self._params = params
        elif params_path is not None:
            self._params = load_params(params_path)
        else:
            self._params = {}
        self._rng = np.random.RandomState(RANDOM_SEED)
        self._prev_points: np.ndarray | None = None

    def generate(self, posture: str, prev_centroid: np.ndarray | None = None,
                 dt: float = 0.1) -> np.ndarray:
        """
        生成单帧点云 (N, 5) [x, y, z, velocity, intensity]
        """
        if posture not in self._params:
            return _fallback_uniform(posture, self._rng)

        p = self._params[posture]

        # 1. 点数 = Poisson 采样（限制 5-100）
        n = max(5, min(100, self._rng.poisson(p.point_count_lambda)))

        # 2. GMM 采样 (n, 3)
        component = self._rng.choice(len(p.gmm_weights), size=n, p=p.gmm_weights)
        raw_xyz = np.zeros((n, 3), dtype=np.float32)
        for k in range(len(p.gmm_weights)):
            mask = component == k
            nk = mask.sum()
            if nk == 0:
                continue
            raw_xyz[mask] = self._rng.multivariate_normal(
                p.gmm_means[k], p.gmm_covariances[k], size=nk,
            )

        # 3. 时间平滑：限制质心位移
        centroid = raw_xyz.mean(axis=0)
        if prev_centroid is not None:
            disp = centroid - prev_centroid
            dist = np.linalg.norm(disp)
            max_d = p.max_displacement * dt
            if dist > max_d and dist > 1e-6:
                centroid = prev_centroid + (disp / dist) * max_d
            raw_xyz += (centroid - raw_xyz.mean(axis=0))

        # 4. 速度：点对点位移估算
        if self._prev_points is not None and len(self._prev_points) > 0:
            velocity = _estimate_velocity(raw_xyz, self._prev_points, dt)
        else:
            velocity = np.abs(self._rng.normal(p.velocity_mean,
                               max(p.velocity_std, 0.01), size=n))

        # 5. 强度：Beta 采样
        intensity = self._rng.beta(
            max(p.intensity_alpha, 0.01), max(p.intensity_beta, 0.01), size=n,
        )

        self._prev_points = raw_xyz.copy()
        return np.column_stack([raw_xyz, velocity, intensity]).astype(np.float32)


class ReplayGenerator:
    """从 3DPCHM 回放真实帧序列"""

    def __init__(self, dataset_path: str):
        data = np.load(dataset_path, allow_pickle=True)
        self._pts = data["points"]
        self._actions = data["action_labels"].astype(np.int32)
        self._subjects = data["subject_ids"].astype(np.int32)
        self._frames = data["frame_indices"].astype(np.int32)

        # 按 (action, subject) 索引帧序列
        self._index: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in range(len(self._pts)):
            key = (int(self._actions[i]), int(self._subjects[i]))
            self._index[key].append(i)

        self._rng = np.random.RandomState(RANDOM_SEED)

    def generate_sequence(self, action: str, duration_s: float,
                          subject_ids: list[int] | None = None) -> list[np.ndarray]:
        """返回按时间排序的帧列表"""
        action_id = ACTION_NAMES.index(action) if action in ACTION_NAMES else 0
        subjects = set(subject_ids) if subject_ids else set()

        candidates = []
        for (aid, sid), indices in self._index.items():
            if aid == action_id and (not subjects or sid in subjects):
                candidates.extend(indices)

        if not candidates:
            return []

        candidates.sort(key=lambda i: self._frames[i])

        fps = 10
        n_frames = max(1, int(duration_s * fps))
        n_available = len(candidates)

        if n_available <= n_frames:
            seq_indices = candidates
        else:
            start = self._rng.randint(0, n_available - n_frames)
            seq_indices = candidates[start:start + n_frames]

        return [self._pts[i].astype(np.float32) for i in seq_indices]


def _fallback_uniform(posture: str, rng: np.random.RandomState) -> np.ndarray:
    """无校准参数时的回退：均匀包围盒（范围适配典型房间）"""
    z_min, z_max = _FALLBACK_Z.get(posture, (0.0, 1.5))
    n = rng.randint(10, 25)
    xyz = np.column_stack([
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(-0.6, 0.6, n),
        rng.uniform(z_min, z_max, n),
    ])
    vel = np.abs(rng.normal(0, 0.5, n))
    itn = rng.uniform(0.3, 1.0, n)
    return np.column_stack([xyz, vel, itn]).astype(np.float32)


def _estimate_velocity(curr_xyz: np.ndarray, prev_xyz: np.ndarray,
                       dt: float) -> np.ndarray:
    """通过最近邻匹配估算逐点速度"""
    n = len(curr_xyz)
    vel = np.zeros(n, dtype=np.float32)
    dt_safe = max(dt, 0.01)
    for i in range(n):
        dists = np.linalg.norm(prev_xyz[:, :3] - curr_xyz[i, :3], axis=1)
        j = int(np.argmin(dists))
        vel[i] = np.linalg.norm(curr_xyz[i, :3] - prev_xyz[j, :3]) / dt_safe
    return vel
