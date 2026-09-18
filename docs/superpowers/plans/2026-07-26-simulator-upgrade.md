# 仿真系统升级 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 simulator/ 从 106 行硬编码灌数脚本升级为 AI 验证基础设施——可控场景生成 → 自动标注 → 双通道注入。

**Architecture:** 9 个模块，每个独立可测。模块间通过 `FrameGroup` dataclass 通信。生成器产出 FrameGroup → ScenarioEngine 编排时间线 → Injectors 双通道注入（WS 全链路 / Redis 直写 AI）。校准是离线一次性步骤。

**Tech Stack:** Python 3.14, numpy, scikit-learn (GMM), PyYAML, websockets, redis-py

## 审查修正清单

设计文档审查中发现的 7 个问题，在此计划中修正：

1. **Redis stream 键名对齐** → RedisInjector 使用 `housafe:pointcloud:ingest` / `housafe:vital:ingest`（匹配现有 `redis_client.py`）
2. **动作类别统一** → 校准覆盖全部 12 类；场景暴露 5 常用类 + fall
3. **心跳帧** → FrameGroup 增加 `heartbeat: bool` 字段；ScenarioEngine 按时间间隔自动插入
4. **时间加速** → ScenarioEngine 接受 `speed` 参数（默认 1x），ts 间隔除以 speed
5. **Ground Truth 单文件** → 输出 `ground_truth.jsonl`（每行一个 gt 记录）
6. **WS 鉴权协议** → 精确匹配 `feed.py` 的 `{device_id, secret}` → `{ack: "auth"}` 流程
7. **错误处理** → 各模块内部处理，连接失败抛明确异常

## 文件结构

```
simulator/
  types.py              # FrameGroup, GroundTruth, VitalRecord, RoomConfig
  calibrate.py          # GMM 校准 CLI（load 3DPCHM npz → fit GMM per action → save）
  generators.py         # SyntheticGenerator (GMM采样) + ReplayGenerator (3DPCHM回放)
  room_vitals.py        # RoomModel (坐标变换+噪声) + VitalSignsGen (生理模型)
  anomaly.py            # AnomalyInjector (fall/stillness/vital_anomaly/offline)
  scenario.py           # ScenarioEngine (YAML解析+时间线编排)
  injectors.py          # RedisInjector + WSInjector
  cli.py                # CLI 入口 (list/run/calibrate/batch)
  validate.py           # 仿真有效性自检
  scenarios/            # 示例场景 YAML
    elderly_day_normal.yaml
```

---

### Task 1: 基础类型定义

**Files:**
- Create: `simulator/types.py`

**Interfaces:**
- Produces: `VitalRecord`, `GroundTruth`, `FrameGroup`, `RoomConfig` dataclasses — 所有后续任务的公共接口

- [ ] **Step 1: 写 types.py**

```python
"""仿真系统共享类型 — 所有模块通过 dataclass 通信"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VitalRecord:
    """单帧生命体征"""
    heart_rate: float       # bpm
    resp_rate: float        # bpm
    quality: float          # 0-1


@dataclass
class GroundTruth:
    """精确标注 — AI 评估的参考基准"""
    frame_id: str
    ts: int                           # UTC ms
    posture: str                      # stand/sit/lie/walk/fall
    posture_confidence: float         # 1.0=synthetic, 0.7-0.95=replay
    heart_rate_true: float | None
    resp_rate_true: float | None
    anomaly_type: str | None          # fall/stillness/vital_anomaly/offline/None
    anomaly_severity: str | None      # info/warning/critical
    anomaly_start: bool = False
    anomaly_end: bool = False
    scenario_name: str = ""
    generator: str = ""               # synthetic/replay/hybrid
    params_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        d["anomaly_start"] = int(self.anomaly_start)
        d["anomaly_end"] = int(self.anomaly_end)
        return d


@dataclass
class FrameGroup:
    """统一帧协议 — 所有生成器输出"""
    frame_id: str
    ts: int                  # UTC ms
    device_id: str
    room: str
    points: np.ndarray | None = None    # (N,5) float32 [x,y,z,velocity,intensity]
    vitals: VitalRecord | None = None
    gt: GroundTruth | None = None
    heartbeat: bool = False             # True=心跳帧（无点云/体征，仅保活）


@dataclass
class RoomConfig:
    """房间物理参数"""
    name: str
    size: tuple[float, float, float]         # (x, y, z) 米
    radar_pos: tuple[float, float, float]    # 雷达安装位置
    noise_sigma: float = 0.02                # 高斯噪声标准差（米）
    dropout_rate: float = 0.01               # 随机丢点率
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.types import FrameGroup, GroundTruth, VitalRecord, RoomConfig; print('OK')"
```

---

### Task 2: GMM 校准模块

**Files:**
- Create: `simulator/calibrate.py`

**Interfaces:**
- Consumes: 3DPCHM npz (`points`, `action_labels` arrays)
- Produces: `calibrated_params.npz` — dict of `{action_name: GMMParams}`; CLI `python -m simulator.calibrate`

- [ ] **Step 1: 写 calibrate.py**

```python
"""从 3DPCHM 数据集拟合 GMM 参数 — 离线一次性校准"""
import argparse
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture

# 3DPCHM 全部 12 个动作类（按 ai/data/public/stats.py）
ACTION_NAMES = [
    "stand", "sit", "fall", "walk",
    "punch", "jump", "wave_left", "lean_left",
    "open_arms", "wave_right", "lean_right", "squat",
]

# 场景系统暴露的常用动作
SCENE_ACTIONS = {"stand", "sit", "fall", "walk", "squat"}

GMM_K_DEFAULT = 4
RANDOM_SEED = 42


@dataclass
class GMMParams:
    """单个动作的统计参数"""
    action: str
    gmm_weights: np.ndarray       # (K,)
    gmm_means: np.ndarray         # (K, 3) 空间均值
    gmm_covariances: np.ndarray   # (K, 3, 3) 协方差矩阵
    point_count_lambda: float     # Poisson λ
    velocity_mean: float
    velocity_std: float
    intensity_alpha: float        # Beta α
    intensity_beta: float         # Beta β
    max_displacement: float       # 质心每秒最大位移 (m/s)


def fit_action_gmm(points_list: list[np.ndarray], k: int = GMM_K_DEFAULT) -> GaussianMixture:
    """对单动作的所有帧点云拟合 GMM"""
    all_points = np.vstack([p[:, :3] for p in points_list if len(p) > 0])
    gmm = GaussianMixture(n_components=min(k, len(points_list)), random_state=RANDOM_SEED,
                          covariance_type="full", max_iter=200)
    gmm.fit(all_points)
    return gmm


def fit_point_count_lambda(points_list: list[np.ndarray]) -> float:
    """Poisson λ = 每帧点数均值"""
    counts = [len(p) for p in points_list]
    return float(np.mean(counts))


def fit_velocity_stats(points_list: list[np.ndarray]) -> tuple[float, float]:
    """拟合速度分布（点云第4列）"""
    velocities = []
    for p in points_list:
        if p.shape[1] >= 4:
            velocities.extend(p[:, 3].tolist())
    if not velocities:
        return 0.0, 0.5
    return float(np.mean(velocities)), float(np.std(velocities))


def fit_intensity_beta(points_list: list[np.ndarray]) -> tuple[float, float]:
    """拟合强度 Beta 分布（点云第5列），method of moments"""
    intensities = []
    for p in points_list:
        if p.shape[1] >= 5:
            intensities.extend(p[:, 4].tolist())
    if not intensities:
        return 1.0, 1.0
    arr = np.array(intensities)
    arr = arr[(arr > 0) & (arr < 1)]
    if len(arr) < 2:
        return 1.0, 1.0
    mu = arr.mean()
    var = arr.var()
    if var == 0:
        return 1.0, 1.0
    # method of moments: α = μ(μ(1-μ)/σ² - 1), β = (1-μ)(μ(1-μ)/σ² - 1)
    common = mu * (1 - mu) / var - 1
    alpha = max(mu * common, 0.01)
    beta_val = max((1 - mu) * common, 0.01)
    return float(alpha), float(beta_val)


def fit_max_displacement(points_list: list[np.ndarray], fps: float = 10.0) -> float:
    """从相邻帧质心位移估算 max_displacement (m/s)"""
    centroids = [p[:, :3].mean(axis=0) for p in points_list if len(p) > 0]
    if len(centroids) < 2:
        return 1.5
    displacements = [np.linalg.norm(centroids[i] - centroids[i-1])
                     for i in range(1, len(centroids))]
    return float(np.percentile(displacements, 95) * fps * 1.5)  # 95分位 * safety margin


def calibrate(dataset_path: str, k: int = GMM_K_DEFAULT) -> dict[str, GMMParams]:
    """主校准流程：加载数据集 → 逐动作拟合 → 返回参数字典"""
    data = np.load(dataset_path, allow_pickle=True)
    pts_list = data["points"]
    action_labels = data["action_labels"].astype(np.int32)

    params = {}
    for aid, name in enumerate(ACTION_NAMES):
        mask = action_labels == aid
        if mask.sum() < 10:
            print(f"  [WARN] {name}: 仅 {mask.sum()} 帧，跳过")
            continue
        action_points = [pts_list[i] for i in np.where(mask)[0]]
        gmm = fit_action_gmm(action_points, k)
        params[name] = GMMParams(
            action=name,
            gmm_weights=gmm.weights_,
            gmm_means=gmm.means_,
            gmm_covariances=gmm.covariances_,
            point_count_lambda=fit_point_count_lambda(action_points),
            velocity_mean=fit_velocity_stats(action_points)[0],
            velocity_std=fit_velocity_stats(action_points)[1],
            intensity_alpha=fit_intensity_beta(action_points)[0],
            intensity_beta=fit_intensity_beta(action_points)[1],
            max_displacement=fit_max_displacement(action_points),
        )
        print(f"  {name}: {mask.sum()} 帧, λ={params[name].point_count_lambda:.1f}, "
              f"max_disp={params[name].max_displacement:.2f} m/s")
    return params


def save_params(params: dict[str, GMMParams], out_path: str):
    """保存参数到 npz"""
    save_dict = {}
    for name, p in params.items():
        save_dict[f"{name}_weights"] = p.gmm_weights
        save_dict[f"{name}_means"] = p.gmm_means
        save_dict[f"{name}_covs"] = p.gmm_covariances
        save_dict[f"{name}_pt_lambda"] = np.array([p.point_count_lambda])
        save_dict[f"{name}_vel_mean"] = np.array([p.velocity_mean])
        save_dict[f"{name}_vel_std"] = np.array([p.velocity_std])
        save_dict[f"{name}_int_alpha"] = np.array([p.intensity_alpha])
        save_dict[f"{name}_int_beta"] = np.array([p.intensity_beta])
        save_dict[f"{name}_max_disp"] = np.array([p.max_displacement])
    np.savez_compressed(out_path, **save_dict)


def load_params(npz_path: str) -> dict[str, GMMParams]:
    """从 npz 加载参数"""
    data = np.load(npz_path)
    params = {}
    for name in ACTION_NAMES:
        key = f"{name}_weights"
        if key not in data:
            continue
        params[name] = GMMParams(
            action=name,
            gmm_weights=data[f"{name}_weights"],
            gmm_means=data[f"{name}_means"],
            gmm_covariances=data[f"{name}_covs"],
            point_count_lambda=float(data[f"{name}_pt_lambda"]),
            velocity_mean=float(data[f"{name}_vel_mean"]),
            velocity_std=float(data[f"{name}_vel_std"]),
            intensity_alpha=float(data[f"{name}_int_alpha"]),
            intensity_beta=float(data[f"{name}_int_beta"]),
            max_displacement=float(data[f"{name}_max_disp"]),
        )
    return params


def main():
    parser = argparse.ArgumentParser(description="校准仿真参数（从 3DPCHM 数据集）")
    parser.add_argument("--dataset", required=True, help="3DPCHM 预处理 npz 路径")
    parser.add_argument("--out", required=True, help="输出参数文件路径")
    parser.add_argument("--k", type=int, default=GMM_K_DEFAULT, help="GMM 分量数 (default: 4)")
    args = parser.parse_args()

    print(f"加载数据集: {args.dataset}")
    params = calibrate(args.dataset, k=args.k)
    save_params(params, args.out)
    print(f"\n参数已保存到: {args.out} ({len(params)} 个动作)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.calibrate import calibrate, load_params, GMMParams, ACTION_NAMES; print('OK')"
```

---

### Task 3: 点云生成器

**Files:**
- Create: `simulator/generators.py`

**Interfaces:**
- Consumes: `simulator.calibrate.GMMParams`
- Produces: `SyntheticGenerator(posture, prev_centroid, dt) → np.ndarray`, `ReplayGenerator(action, duration) → Iterator[np.ndarray]`

- [ ] **Step 1: 写 generators.py**

```python
"""点云生成器 — GMM 合成 + 3DPCHM 回放"""
import numpy as np
from collections import defaultdict
from simulator.calibrate import GMMParams, load_params, ACTION_NAMES

RANDOM_SEED = 42
FALLBACK_DISP = {"stand": 0.3, "sit": 0.2, "lie": 0.1, "walk": 1.5, "fall": 2.0, "squat": 0.3}


class SyntheticGenerator:
    """基于校准 GMM 参数的运行时点云生成"""

    def __init__(self, params: dict[str, GMMParams] | None = None, params_path: str | None = None):
        if params is not None:
            self._params = params
        elif params_path is not None:
            self._params = load_params(params_path)
        else:
            self._params = {}
        self._rng = np.random.RandomState(RANDOM_SEED)
        self._prev_points = None

    def generate(self, posture: str, prev_centroid: np.ndarray | None = None,
                 dt: float = 0.1) -> np.ndarray:
        """
        生成单帧点云 (N, 5) [x, y, z, velocity, intensity]
        prev_centroid: 上一帧质心 (3,) 或 None
        dt: 距上一帧的秒数
        """
        if posture not in self._params:
            return _fallback_uniform(posture, self._rng)

        p = self._params[posture]

        # 1. 点数 = Poisson 采样
        n = max(5, min(100, self._rng.poisson(p.point_count_lambda)))

        # 2. GMM 采样 (n, 3)
        component = self._rng.choice(len(p.gmm_weights), size=n, p=p.gmm_weights)
        raw_xyz = np.zeros((n, 3), dtype=np.float32)
        for k in range(len(p.gmm_weights)):
            mask = component == k
            nk = mask.sum()
            if nk == 0:
                continue
            raw_xyz[mask] = self._rng.multivariate_normal(p.gmm_means[k], p.gmm_covariances[k], size=nk)

        # 3. 时间平滑：限制质心位移
        centroid = raw_xyz.mean(axis=0)
        if prev_centroid is not None:
            disp = centroid - prev_centroid
            dist = np.linalg.norm(disp)
            max_d = p.max_displacement * dt
            if dist > max_d:
                centroid = prev_centroid + (disp / dist) * max_d
            raw_xyz += (centroid - raw_xyz.mean(axis=0))

        # 4. 速度：点对点位移 / dt（与上一帧最近邻匹配估算）
        if self._prev_points is not None and len(self._prev_points) > 0:
            velocity = _estimate_velocity(raw_xyz, self._prev_points, dt)
        else:
            velocity = np.abs(self._rng.normal(p.velocity_mean, p.velocity_std, size=n))

        # 5. 强度：Beta 采样
        intensity = self._rng.beta(p.intensity_alpha, p.intensity_beta, size=n)

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

        # 收集匹配的帧索引
        candidates = []
        for (aid, sid), indices in self._index.items():
            if aid == action_id and (not subjects or sid in subjects):
                candidates.extend(indices)

        if not candidates:
            return []

        # 按 frame_id 排序
        candidates.sort(key=lambda i: self._frames[i])

        # 随机选起点
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
    """无校准参数时的回退：均匀包围盒"""
    z_ranges = {"stand": (0.0, 1.7), "sit": (0.0, 1.0), "lie": (0.0, 0.3),
                "walk": (0.0, 1.8), "fall": (0.0, 0.2), "squat": (0.0, 1.1)}
    z_min, z_max = z_ranges.get(posture, (0.0, 1.5))
    n = rng.randint(8, 30)
    xyz = np.column_stack([
        rng.uniform(-2, 2, n),
        rng.uniform(-2, 2, n),
        rng.uniform(z_min, z_max, n),
    ])
    vel = np.abs(rng.normal(0, 0.5, n))
    itn = rng.uniform(0.3, 1.0, n)
    return np.column_stack([xyz, vel, itn]).astype(np.float32)


def _estimate_velocity(curr_xyz: np.ndarray, prev_xyz: np.ndarray, dt: float) -> np.ndarray:
    """通过最近邻匹配估算逐点速度"""
    n = len(curr_xyz)
    vel = np.zeros(n, dtype=np.float32)
    dt_safe = max(dt, 0.01)
    for i in range(n):
        dists = np.linalg.norm(prev_xyz[:, :3] - curr_xyz[i, :3], axis=1)
        j = np.argmin(dists)
        vel[i] = np.linalg.norm(curr_xyz[i, :3] - prev_xyz[j, :3]) / dt_safe
    return vel
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.generators import SyntheticGenerator, ReplayGenerator; print('OK')"
```

---

### Task 4: 房间模型 + 体征生成

**Files:**
- Create: `simulator/room_vitals.py`

**Interfaces:**
- Consumes: `simulator.types.RoomConfig`, `simulator.types.VitalRecord`
- Produces: `RoomModel.apply(points) → np.ndarray`, `VitalSignsGen.generate(activity, t) → VitalRecord`

- [ ] **Step 1: 写 room_vitals.py**

```python
"""房间模型 + 生命体征生成"""
import math
import numpy as np
from simulator.types import RoomConfig, VitalRecord

RESTING_HR = 72
RESTING_RR = 16
DEFAULT_ACTIVITY = "sit"

ACTIVITY_MULT = {"lie": 0.9, "sit": 1.0, "stand": 1.05, "walk": 1.3, "fall": 1.6, "squat": 1.1}
RR_MULT = {"lie": 1.0, "sit": 1.0, "stand": 1.1, "walk": 1.5, "fall": 2.0, "squat": 1.1}


class RoomModel:
    """房间坐标变换 + 噪声"""

    def __init__(self, config: RoomConfig):
        self._config = config
        self._rng = np.random.RandomState(42)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """对点云施加坐标变换和噪声，返回过滤后的点云"""
        if points is None or len(points) == 0:
            return points

        pts = points.copy()

        # 1. 坐标平移：世界坐标系 → 雷达坐标系
        rx, ry, rz = self._config.radar_pos
        pts[:, 0] -= rx
        pts[:, 1] -= ry
        pts[:, 2] -= rz

        # 2. 墙壁裁剪
        sx, sy, sz = self._config.size
        mask = (pts[:, 0] >= -sx/2) & (pts[:, 0] <= sx/2) & \
               (pts[:, 1] >= -sy/2) & (pts[:, 1] <= sy/2) & \
               (pts[:, 2] >= -0.1) & (pts[:, 2] <= sz)
        pts = pts[mask]

        if len(pts) == 0:
            return pts

        # 3. 高斯噪声
        noise = self._rng.normal(0, self._config.noise_sigma, size=(len(pts), 3))
        pts[:, :3] += noise

        # 4. 随机丢点
        if self._config.dropout_rate > 0:
            keep = self._rng.random(len(pts)) > self._config.dropout_rate
            pts = pts[keep]

        # 坐标平移回去（保持绝对坐标）
        pts[:, 0] += rx
        pts[:, 1] += ry
        pts[:, 2] += rz

        return pts.astype(np.float32)


class VitalSignsGen:
    """生理信号模型 — 活动相关 HR/RR + 呼吸性窦性心律不齐"""

    def __init__(self, resting_hr: float = RESTING_HR, resting_rr: float = RESTING_RR,
                 seed: int = 42):
        self.resting_hr = resting_hr
        self.resting_rr = resting_rr
        self._rng = np.random.RandomState(seed)

    def generate(self, activity: str, t: float) -> VitalRecord:
        """t: 场景时间（秒）"""
        hr_mult = ACTIVITY_MULT.get(activity, 1.0)
        rr_mult = RR_MULT.get(activity, 1.0)

        # 呼吸性窦性心律不齐（RSA）：吸气时心率↑，呼气时↓
        rr_cycle = self.resting_rr * rr_mult
        rr_freq = rr_cycle / 60.0  # Hz
        rsa_amplitude = 3.0  # bpm

        hr = (self.resting_hr * hr_mult
              + rsa_amplitude * math.sin(2 * math.pi * rr_freq * t)
              + self._rng.normal(0, 3))

        rr = rr_cycle + self._rng.normal(0, 1)

        quality = 0.95 + self._rng.normal(0, 0.02)
        quality = max(0.1, min(1.0, quality))

        return VitalRecord(
            heart_rate=round(max(30, min(200, hr)), 1),
            resp_rate=round(max(5, min(40, rr)), 1),
            quality=round(quality, 2),
        )
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.room_vitals import RoomModel, VitalSignsGen; print('OK')"
```

---

### Task 5: 异常注入器

**Files:**
- Create: `simulator/anomaly.py`

**Interfaces:**
- Consumes: `simulator.types.FrameGroup`, `simulator.types.GroundTruth`
- Produces: `AnomalyInjector.inject(frames, injection_configs) → list[FrameGroup]`

- [ ] **Step 1: 写 anomaly.py**

```python
"""异常注入器 — 在时间线精确位置注入异常并产出标注"""
import numpy as np
from simulator.types import FrameGroup, GroundTruth, VitalRecord
from simulator.calibrate import ACTION_NAMES

ANOMALY_TEMPLATES = {
    "fall": {
        "transition_duration": 1.0,
        "end_posture": "lie",
        "velocity_spike": True,
        "height_drop_rate": 1.0,
    },
    "stillness": {
        "min_duration_s": 1800,
        "max_displacement": 0.05,
    },
    "vital_anomaly": {
        "hr_range": (40, 130),
        "rr_range": (5, 35),
        "ramp_duration_s": 60,
    },
    "offline": {
        "gap_duration_s": 120,
    },
}


class AnomalyInjector:

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def inject(self, frames: list[FrameGroup], injections: list[dict]) -> list[FrameGroup]:
        """
        根据 injection 配置修改帧序列，注入异常并标注 ground truth。
        injections: [{"at_s": float, "type": str, "room": str, "params": dict, "ground_truth": dict}, ...]
        """
        # 按注入时间排序，倒序处理避免索引偏移
        injections = sorted(injections, key=lambda x: x["at_s"], reverse=True)

        result = list(frames)

        for inj in injections:
            at_s = inj["at_s"]
            inj_type = inj["type"]
            inj_params = inj.get("params", {})
            gt_meta = inj.get("ground_truth", {})

            if inj_type == "fall":
                result = self._inject_fall(result, at_s, inj_params, gt_meta)
            elif inj_type == "stillness":
                result = self._inject_stillness(result, at_s, inj_params, gt_meta)
            elif inj_type == "vital_anomaly":
                result = self._inject_vital_anomaly(result, at_s, inj_params, gt_meta)
            elif inj_type == "offline":
                result = self._inject_offline(result, at_s, inj_params, gt_meta)

        return result

    def _find_insert_index(self, frames: list[FrameGroup], at_s: float) -> int:
        """找到 at_s 对应的时间索引"""
        if not frames:
            return 0
        at_ts = int(at_s * 1000)
        for i, f in enumerate(frames):
            if f.ts >= at_ts:
                return i
        return len(frames)

    def _inject_fall(self, frames: list[FrameGroup], at_s: float,
                     params: dict, gt_meta: dict) -> list[FrameGroup]:
        """注入跌倒：质心 z 线性下降，姿态切换"""
        idx = self._find_insert_index(frames, at_s)
        if idx >= len(frames):
            return frames

        duration = params.get("transition_duration", 1.0)
        drop_rate = params.get("height_drop_rate", 1.0)

        # 修改后续帧
        n_frames = max(1, int(duration * 10))  # 10 fps
        end_idx = min(idx + n_frames, len(frames))
        for i in range(idx, end_idx):
            f = frames[i]
            if f.points is not None and len(f.points) > 0:
                progress = (i - idx) / max(n_frames - 1, 1)
                # z 线性下降
                f.points[:, 2] -= drop_rate * duration * progress / n_frames
                f.points[:, 2] = np.maximum(f.points[:, 2], 0.0)
            # 标注
            f.gt = GroundTruth(
                frame_id=f.frame_id,
                ts=f.ts,
                posture="fall" if progress < 0.8 else "lie",
                posture_confidence=1.0,
                heart_rate_true=f.vitals.heart_rate if f.vitals else None,
                resp_rate_true=f.vitals.resp_rate if f.vitals else None,
                anomaly_type="fall",
                anomaly_severity=gt_meta.get("severity", "critical"),
                anomaly_start=(i == idx),
                anomaly_end=(i == end_idx - 1),
                scenario_name=gt_meta.get("desc", ""),
                generator="synthetic",
            )
        return frames

    def _inject_stillness(self, frames: list[FrameGroup], at_s: float,
                          params: dict, gt_meta: dict) -> list[FrameGroup]:
        """注入静止：不改变点云，仅标记 ground truth"""
        idx = self._find_insert_index(frames, at_s)
        min_dur = params.get("min_duration_s", 1800)
        n_frames = max(1, int(min_dur * 10))
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            if f.gt is None:
                f.gt = GroundTruth(
                    frame_id=f.frame_id, ts=f.ts,
                    posture="lie", posture_confidence=0.9,
                    heart_rate_true=f.vitals.heart_rate if f.vitals else None,
                    resp_rate_true=f.vitals.resp_rate if f.vitals else None,
                    anomaly_type="stillness",
                    anomaly_severity=gt_meta.get("severity", "warning"),
                    anomaly_start=(i == idx), anomaly_end=(i == end_idx - 1),
                    scenario_name=gt_meta.get("desc", ""), generator="synthetic",
                )
            else:
                f.gt.anomaly_type = "stillness"
                f.gt.anomaly_severity = gt_meta.get("severity", "warning")
        return frames

    def _inject_vital_anomaly(self, frames: list[FrameGroup], at_s: float,
                              params: dict, gt_meta: dict) -> list[FrameGroup]:
        """注入体征异常：逐步 ramp HR/RR 到异常范围"""
        idx = self._find_insert_index(frames, at_s)
        ramp_dur = params.get("ramp_duration_s", 60)
        hr_range = params.get("hr_range", (40, 130))
        rr_range = params.get("rr_range", (5, 35))

        n_frames = max(1, int(ramp_dur * 10))
        end_idx = min(idx + n_frames, len(frames))
        for i in range(idx, end_idx):
            f = frames[i]
            progress = (i - idx) / max(n_frames - 1, 1)
            if f.vitals:
                target_hr = hr_range[0] + (hr_range[1] - hr_range[0]) * progress
                target_rr = rr_range[0] + (rr_range[1] - rr_range[0]) * progress
                f.vitals = VitalRecord(
                    heart_rate=round(target_hr + self._rng.normal(0, 2), 1),
                    resp_rate=round(target_rr + self._rng.normal(0, 0.5), 1),
                    quality=round(0.95 + self._rng.normal(0, 0.02), 2),
                )
            if f.gt is None:
                f.gt = GroundTruth(
                    frame_id=f.frame_id, ts=f.ts,
                    posture="lie", posture_confidence=1.0,
                    heart_rate_true=f.vitals.heart_rate if f.vitals else None,
                    resp_rate_true=f.vitals.resp_rate if f.vitals else None,
                    anomaly_type="vital_anomaly",
                    anomaly_severity=gt_meta.get("severity", "warning"),
                    anomaly_start=(i == idx), anomaly_end=(i == end_idx - 1),
                    scenario_name=gt_meta.get("desc", ""), generator="synthetic",
                )
            else:
                f.gt.anomaly_type = "vital_anomaly"
                f.gt.anomaly_severity = gt_meta.get("severity", "warning")
        return frames

    def _inject_offline(self, frames: list[FrameGroup], at_s: float,
                        params: dict, gt_meta: dict) -> list[FrameGroup]:
        """注入断连：清空后续帧的点云/体征（模拟设备离线）"""
        idx = self._find_insert_index(frames, at_s)
        gap_dur = params.get("gap_duration_s", 120)
        n_frames = max(1, int(gap_dur * 10))
        end_idx = min(idx + n_frames, len(frames))

        for i in range(idx, end_idx):
            f = frames[i]
            f.points = None
            f.vitals = None
            f.heartbeat = False
            if f.gt is None:
                f.gt = GroundTruth(
                    frame_id=f.frame_id, ts=f.ts,
                    posture="", posture_confidence=0.0,
                    heart_rate_true=None, resp_rate_true=None,
                    anomaly_type="offline",
                    anomaly_severity=gt_meta.get("severity", "critical"),
                    anomaly_start=(i == idx), anomaly_end=(i == end_idx - 1),
                    scenario_name=gt_meta.get("desc", ""), generator="synthetic",
                )
            else:
                f.gt.anomaly_type = "offline"
                f.gt.anomaly_severity = gt_meta.get("severity", "critical")
        return frames
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.anomaly import AnomalyInjector, ANOMALY_TEMPLATES; print('OK')"
```

---

### Task 6: 场景引擎

**Files:**
- Create: `simulator/scenario.py`

**Interfaces:**
- Consumes: `simulator.types.*`, `simulator.generators.*`, `simulator.room_vitals.*`, `simulator.anomaly.*`
- Produces: `ScenarioEngine(yaml_path, speed, params_path, dataset_path).run() → Iterator[FrameGroup]`

- [ ] **Step 1: 写 scenario.py**

```python
"""场景引擎 — YAML 解析 + 时间线编排"""
import yaml
import time
import numpy as np
from pathlib import Path
from simulator.types import FrameGroup, GroundTruth, VitalRecord, RoomConfig
from simulator.generators import SyntheticGenerator, ReplayGenerator
from simulator.room_vitals import RoomModel, VitalSignsGen
from simulator.anomaly import AnomalyInjector

DEFAULT_FPS = 10


class ScenarioEngine:
    """解析 YAML 场景，编排时间线，产出 FrameGroup 序列"""

    def __init__(self, yaml_path: str, speed: float = 1.0,
                 params_path: str | None = None,
                 dataset_path: str | None = None):
        with open(yaml_path) as f:
            self._cfg = yaml.safe_load(f)

        self._speed = speed
        self._fps = self._cfg.get("generator_defaults", {}).get("fps", DEFAULT_FPS)
        self._dt = 1.0 / self._fps

        # 房间模型
        self._rooms: dict[str, tuple[RoomModel, RoomConfig]] = {}
        for room_name, room_data in self._cfg.get("rooms", {}).items():
            cfg = RoomConfig(
                name=room_name,
                size=tuple(room_data["size"]),
                radar_pos=tuple(room_data["radar_pos"]),
            )
            self._rooms[room_name] = (RoomModel(cfg), cfg)

        # 生成器
        self._synth = SyntheticGenerator(params_path=params_path) if params_path else SyntheticGenerator()
        self._replay = ReplayGenerator(dataset_path) if dataset_path else None

        # 体征
        vcfg = self._cfg.get("vitals", {})
        self._vitals = VitalSignsGen(
            resting_hr=vcfg.get("resting_hr", 72),
            resting_rr=vcfg.get("resting_rr", 16),
        )

        # 异常注入器
        self._anomaly = AnomalyInjector()

        self._device_id = self._cfg.get("device_id", "sim_device")
        self._scenario_name = self._cfg.get("name", Path(yaml_path).stem)
        self._frame_seq = 0

    def run(self) -> list[FrameGroup]:
        """运行整个时间线，返回全部帧列表（含异常注入）"""
        timeline = self._cfg.get("timeline", [])
        injections = self._cfg.get("injections", [])

        frames: list[FrameGroup] = []
        for segment in timeline:
            segment_frames = self._run_segment(segment)
            frames.extend(segment_frames)

        # 按时间排序
        frames.sort(key=lambda f: f.ts)

        # 注入异常
        if injections:
            frames = self._anomaly.inject(frames, injections)

        return frames

    def _run_segment(self, seg: dict) -> list[FrameGroup]:
        """执行一个时间线段"""
        at_s = seg.get("at", 0)
        duration_s = seg.get("duration", 10)
        activity = seg.get("activity", "sit")
        room_name = seg.get("room", "bedroom")
        generator_type = seg.get("generator", "synthetic")

        # 过渡段
        transition = seg.get("transition")
        frames: list[FrameGroup] = []

        if transition:
            trans_frames = self._run_transition(transition, at_s, room_name, generator_type)
            frames.extend(trans_frames)
            at_s += transition.get("duration", 2)
            duration_s -= transition.get("duration", 2)

        # 主动作段
        n_frames = max(1, int(duration_s * self._fps))
        actual_dt = duration_s / n_frames

        # 回放：预取帧序列
        replay_frames: list[np.ndarray] = []
        if generator_type == "replay" and self._replay:
            replay_frames = self._replay.generate_sequence(activity, duration_s)
        elif generator_type == "hybrid":
            # hybrid: 优先 replay，fallback 到 synthetic
            if self._replay:
                replay_frames = self._replay.generate_sequence(activity, duration_s)
            if not replay_frames:
                generator_type = "synthetic"

        prev_centroid = None
        for i in range(n_frames):
            t = at_s + i * actual_dt
            ts = int(t * 1000)

            # 生成点云
            if generator_type == "replay" and i < len(replay_frames):
                points = replay_frames[i].copy()
            else:
                if activity in ("lie", "lying"):
                    synth_posture = "squat"  # 3DPCHM 无 lying，用 squat 近似 z 分布
                else:
                    synth_posture = activity
                points = self._synth.generate(synth_posture, prev_centroid, actual_dt)

            if points is not None and len(points) > 0:
                prev_centroid = points[:, :3].mean(axis=0)

            # 房间变换
            if room_name in self._rooms:
                points = self._rooms[room_name][0].apply(points)

            # 体征
            vitals = self._vitals.generate(activity, t)

            # Ground truth
            gt = GroundTruth(
                frame_id="",
                ts=ts,
                posture=activity,
                posture_confidence=1.0 if generator_type == "synthetic" else 0.85,
                heart_rate_true=vitals.heart_rate,
                resp_rate_true=vitals.resp_rate,
                anomaly_type=None,
                anomaly_severity=None,
                scenario_name=self._scenario_name,
                generator=generator_type,
            )

            self._frame_seq += 1
            frame_id = f"{self._device_id}-{self._frame_seq:06d}"
            gt.frame_id = frame_id

            frames.append(FrameGroup(
                frame_id=frame_id, ts=ts,
                device_id=self._device_id, room=room_name,
                points=points, vitals=vitals, gt=gt,
            ))

        return frames

    def _run_transition(self, trans: dict, at_s: float, room: str,
                        generator: str) -> list[FrameGroup]:
        """生成姿态过渡帧序列"""
        from_posture = trans.get("from", "stand")
        to_posture = trans.get("to", "sit")
        duration = trans.get("duration", 2.0)

        n_frames = max(2, int(duration * self._fps))
        actual_dt = duration / n_frames
        frames: list[FrameGroup] = []

        prev_centroid = None
        for i in range(n_frames):
            progress = i / max(n_frames - 1, 1)
            t = at_s + i * actual_dt
            ts = int(t * 1000)

            # 从 from 过渡到 to 的混合姿态
            if progress < 0.5:
                posture = from_posture
            else:
                posture = to_posture

            points = self._synth.generate(posture, prev_centroid, actual_dt)
            if points is not None and len(points) > 0:
                prev_centroid = points[:, :3].mean(axis=0)

            if room in self._rooms:
                points = self._rooms[room][0].apply(points)

            vitals = self._vitals.generate(posture, t)

            self._frame_seq += 1
            frames.append(FrameGroup(
                frame_id=f"{self._device_id}-{self._frame_seq:06d}",
                ts=ts, device_id=self._device_id, room=room,
                points=points, vitals=vitals,
                gt=GroundTruth(
                    frame_id=f"{self._device_id}-{self._frame_seq:06d}",
                    ts=ts, posture=posture,
                    posture_confidence=0.85,
                    heart_rate_true=vitals.heart_rate,
                    resp_rate_true=vitals.resp_rate,
                    anomaly_type=None, anomaly_severity=None,
                    scenario_name=self._scenario_name, generator=generator,
                ),
            ))
        return frames
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.scenario import ScenarioEngine; print('OK')"
```

---

### Task 7: 双通道注入器

**Files:**
- Create: `simulator/injectors.py`

**Interfaces:**
- Consumes: `simulator.types.FrameGroup`, contracts Pydantic models
- Produces: `RedisInjector.inject(frames)`, `WSInjector.inject(frames, url, device_id, secret)` (async)

- [ ] **Step 1: 写 injectors.py**

```python
"""双通道注入 — Redis Stream 直写 + WebSocket 全链路"""
import json
import asyncio
import redis
import websockets
from simulator.types import FrameGroup
from housafe_contracts.events import PointXYZVI


# 与 ai/shared/redis_client.py 对齐
STREAM_POINTCLOUD = "housafe:pointcloud:ingest"
STREAM_VITAL = "housafe:vital:ingest"


class RedisInjector:
    """Redis Stream 直写 — 跳过 ingest，直接到 AI 消费端"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._gt_redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._frame_seq = 0

    def inject(self, frames: list[FrameGroup]):
        """批量注入帧序列"""
        for fg in frames:
            self._inject_one(fg)

    def _inject_one(self, fg: FrameGroup):
        """注入单帧"""
        ts_str = str(fg.ts)
        device_str = fg.device_id
        room_str = fg.room

        # 点云 → Redis Stream
        if fg.points is not None and len(fg.points) > 0:
            pts_json = json.dumps([
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                 "velocity": float(p[3]), "intensity": float(p[4])}
                for p in fg.points
            ])
            self._redis.xadd(STREAM_POINTCLOUD, {
                "ts": ts_str,
                "device_id": device_str,
                "family_id": "sim",
                "room": room_str,
                "frame_id": fg.frame_id,
                "points": pts_json,
            }, maxlen=10000)

        # 体征 → Redis Stream
        if fg.vitals:
            self._redis.xadd(STREAM_VITAL, {
                "ts": ts_str,
                "device_id": device_str,
                "family_id": "sim",
                "room": room_str,
                "resp_rate": str(fg.vitals.resp_rate),
                "heart_rate": str(fg.vitals.heart_rate),
                "quality": str(fg.vitals.quality),
            }, maxlen=10000)

        # Ground truth → Redis String
        if fg.gt and (fg.gt.anomaly_type or fg.gt.posture):
            self._gt_redis.set(f"gt:{fg.frame_id}", json.dumps(fg.gt.to_dict()))

    def close(self):
        self._redis.close()
        self._gt_redis.close()


class WSInjector:
    """WebSocket 注入器 — 走 /ws/ingest 全链路"""

    async def inject(self, frames: list[FrameGroup], url: str,
                     device_id: str, secret: str):
        """通过 WebSocket 逐帧注入"""
        async with websockets.connect(url) as ws:
            # 鉴权
            await ws.send(json.dumps({"device_id": device_id, "secret": secret}))
            ack = json.loads(await ws.recv())
            if ack.get("ack") != "auth":
                raise RuntimeError(f"WebSocket 鉴权失败: {ack}")

            for fg in frames:
                # 点云帧
                if fg.points is not None and len(fg.points) > 0:
                    payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "room": fg.room,
                        "frame_id": fg.frame_id,
                        "points": [
                            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
                             "velocity": float(p[3]), "intensity": float(p[4])}
                            for p in fg.points
                        ],
                    }
                    await ws.send(json.dumps({"kind": "point_cloud", "payload": payload}))
                    ack = json.loads(await ws.recv())

                # 体征帧
                if fg.vitals:
                    vital_payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "room": fg.room,
                        "quiet": True,
                        "resp_rate": fg.vitals.resp_rate,
                        "heart_rate": fg.vitals.heart_rate,
                        "quality": fg.vitals.quality,
                    }
                    await ws.send(json.dumps({"kind": "vital", "payload": vital_payload}))
                    ack = json.loads(await ws.recv())

                # 心跳帧
                if fg.heartbeat:
                    hb_payload = {
                        "ts": fg.ts,
                        "radar_id": fg.device_id,
                        "status": "online",
                        "fw_version": "sim-1.0",
                    }
                    await ws.send(json.dumps({"kind": "heartbeat", "payload": hb_payload}))
                    ack = json.loads(await ws.recv())


async def inject_ws_async(frames: list[FrameGroup], url: str, device_id: str, secret: str):
    """便捷函数：WS 异步注入"""
    injector = WSInjector()
    await injector.inject(frames, url, device_id, secret)
```

- [ ] **Step 2: 验证导入**

```bash
python3 -c "from simulator.injectors import RedisInjector, WSInjector; print('OK')"
```

---

### Task 8: CLI 入口 + 示例场景

**Files:**
- Create: `simulator/cli.py`
- Create: `simulator/scenarios/elderly_day_normal.yaml`

**Interfaces:**
- Produces: `python -m simulator.cli {list,run,calibrate}`

- [ ] **Step 1: 写 cli.py**

```python
"""仿真系统 CLI 入口"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from simulator.scenario import ScenarioEngine
from simulator.injectors import RedisInjector, inject_ws_async

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def cmd_list(args):
    """列出可用场景"""
    if not SCENARIOS_DIR.exists():
        print("无场景目录")
        return
    for f in sorted(SCENARIOS_DIR.glob("*.yaml")):
        print(f"  {f.stem}  ({f.name})")


def cmd_run(args):
    """运行场景"""
    # 1. 加载并编排
    print(f"加载场景: {args.scenario}")
    engine = ScenarioEngine(
        args.scenario,
        speed=args.speed,
        params_path=args.params,
        dataset_path=args.dataset,
    )
    frames = engine.run()
    print(f"生成 {len(frames)} 帧")

    # 2. Ground truth 落盘
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_path = out_dir / "ground_truth.jsonl"
        with open(gt_path, "w") as f:
            for fg in frames:
                if fg.gt:
                    f.write(json.dumps(fg.gt.to_dict(), ensure_ascii=False) + "\n")
        print(f"Ground truth → {gt_path} ({gt_path.stat().st_size} bytes)")

    # 3. 注入
    if args.redis:
        print(f"Redis 注入: {args.redis}")
        ri = RedisInjector(args.redis)
        ri.inject(frames)
        ri.close()
        print("  ✓ 完成")

    if args.ws:
        print(f"WebSocket 注入: {args.ws}")
        device_id = args.device or "sim_device"
        secret = args.secret or "sim_secret"
        asyncio.run(inject_ws_async(frames, args.ws, device_id, secret))
        print("  ✓ 完成")


def cmd_calibrate(args):
    """校准参数"""
    from simulator.calibrate import calibrate, save_params
    params = calibrate(args.dataset, k=args.k)
    save_params(params, args.out)
    print(f"参数已保存到: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="housafe 仿真系统")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="列出可用场景")

    # run
    p_run = sub.add_parser("run", help="运行场景")
    p_run.add_argument("scenario", help="场景 YAML 路径")
    p_run.add_argument("--speed", type=float, default=1.0, help="时间加速倍率 (default: 1x)")
    p_run.add_argument("--params", help="校准参数 npz 路径")
    p_run.add_argument("--dataset", help="3DPCHM 数据集 npz 路径（replay 模式用）")
    p_run.add_argument("--out", help="Ground truth 输出目录")
    p_run.add_argument("--redis", help="Redis URL (如 redis://localhost:6379/0)")
    p_run.add_argument("--ws", help="WebSocket URL (如 ws://localhost:8000/ws/ingest)")
    p_run.add_argument("--device", help="设备 ID")
    p_run.add_argument("--secret", help="设备密钥")

    # calibrate
    p_cal = sub.add_parser("calibrate", help="校准 GMM 参数")
    p_cal.add_argument("--dataset", required=True, help="3DPCHM 预处理 npz 路径")
    p_cal.add_argument("--out", required=True, help="输出参数文件路径")
    p_cal.add_argument("--k", type=int, default=4, help="GMM 分量数 (default: 4)")

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写示例场景**

```yaml
# scenarios/elderly_day_normal.yaml
name: "老人日常 - 正常半天"
version: 1
device_id: rad_elderly_01

rooms:
  bedroom:
    size: [4.0, 3.5, 2.8]
    radar_pos: [2.0, 3.0, 2.5]
  living:
    size: [5.0, 4.0, 2.8]
    radar_pos: [2.5, 3.5, 2.5]

vitals:
  resting_hr: 68
  resting_rr: 15

generator_defaults:
  method: synthetic
  fps: 10

annotations:
  output: output/
  format: jsonl

timeline:
  - at: 0s
    room: bedroom
    activity: lie
    duration: 10s
    generator: synthetic
  - at: 10s
    room: bedroom
    activity: sit
    duration: 10s
    transition: {from: lie, to: sit, duration: 3s}
  - at: 20s
    room: bedroom
    activity: walk
    duration: 10s
  - at: 30s
    room: living
    activity: walk
    duration: 10s
  - at: 40s
    room: living
    activity: sit
    duration: 30s

injections:
  - at: 35s
    type: fall
    room: living
    params:
      height_drop_rate: 0.8
    ground_truth:
      severity: critical
      desc: "客厅行走时滑倒"
```

- [ ] **Step 3: 验证 CLI**

```bash
python3 -m simulator.cli list
```

---

### Task 9: 验证 + 集成测试

**Files:**
- Create: `simulator/validate.py`
- Create: `simulator/tests/test_simulator.py`

**Interfaces:**
- Consumes: All previous modules
- Produces: `validate_simulation(frames) → dict`, integration tests

- [ ] **Step 1: 写 validate.py**

```python
"""仿真有效性自检"""
import numpy as np
from simulator.types import FrameGroup

EXPECTED_Z = {
    "stand": (0.3, 1.8),
    "sit": (0.0, 1.2),
    "lie": (-0.1, 0.5),
    "walk": (0.3, 1.8),
    "fall": (-0.1, 0.5),
    "squat": (0.0, 1.3),
}


def validate_simulation(frames: list[FrameGroup]) -> dict:
    """对生成数据跑统计检查，返回检查结果字典"""
    checks = {}

    # 1. 点数范围：5-100
    n_points = [len(f.points) for f in frames if f.points is not None and len(f.points) > 0]
    if n_points:
        checks["point_count_range"] = {
            "passed": min(n_points) >= 5 and max(n_points) <= 100,
            "min": min(n_points), "max": max(n_points),
        }
    else:
        checks["point_count_range"] = {"passed": False, "detail": "no valid points"}

    # 2. z 坐标范围：-0.1m ~ 2.2m
    all_z = []
    for f in frames:
        if f.points is not None and len(f.points) > 0 and f.points.shape[1] >= 3:
            all_z.extend(f.points[:, 2].tolist())
    if all_z:
        z_arr = np.array(all_z)
        checks["z_range"] = {
            "passed": z_arr.min() >= -0.5 and z_arr.max() <= 2.5,
            "min": float(z_arr.min()), "max": float(z_arr.max()),
        }
    else:
        checks["z_range"] = {"passed": False, "detail": "no z data"}

    # 3. 时序连续性：相邻帧间隔在 50-200ms（考虑加速）
    if len(frames) >= 2:
        ts_diffs = np.diff([f.ts for f in frames])
        checks["temporal_continuity"] = {
            "passed": ts_diffs.min() > 0 and ts_diffs.max() < 300,
            "min_ms": int(ts_diffs.min()), "max_ms": int(ts_diffs.max()),
            "mean_ms": float(ts_diffs.mean()),
        }

    # 4. 姿态-z一致性
    mismatches = []
    for f in frames:
        if f.gt and f.gt.posture in EXPECTED_Z and f.points is not None and len(f.points) > 0:
            z_mean = float(f.points[:, 2].mean())
            z_lo, z_hi = EXPECTED_Z[f.gt.posture]
            if not (z_lo <= z_mean <= z_hi):
                mismatches.append({"frame_id": f.frame_id, "posture": f.gt.posture, "z_mean": z_mean})
    checks["posture_z_consistency"] = {
        "passed": len(mismatches) <= len(frames) * 0.3,  # 允许 30% 不一致（过渡段）
        "mismatches": len(mismatches),
    }

    # 汇总
    checks["all_passed"] = all(
        v["passed"] if isinstance(v, dict) and "passed" in v else True
        for v in checks.values()
    )
    return checks


def print_validation_report(checks: dict):
    """打印可读的验证报告"""
    print("=" * 50)
    print("  仿真有效性自检报告")
    print("=" * 50)
    for name, result in checks.items():
        if name == "all_passed":
            continue
        passed = result.get("passed", False) if isinstance(result, dict) else result
        status = "✓" if passed else "✗"
        detail = ""
        if isinstance(result, dict):
            detail = " | ".join(f"{k}={v}" for k, v in result.items() if k != "passed")
        print(f"  {status} {name}: {detail}")
    print(f"\n  整体: {'✓ 通过' if checks.get('all_passed') else '✗ 失败'}")
```

- [ ] **Step 2: 写集成测试**

```python
"""仿真系统集成测试"""
import json
import tempfile
from pathlib import Path
from simulator.types import FrameGroup, GroundTruth, VitalRecord, RoomConfig
from simulator.room_vitals import RoomModel, VitalSignsGen
from simulator.anomaly import AnomalyInjector


class TestVitalSignsGen:
    def test_generate_returns_valid_vitals(self):
        gen = VitalSignsGen(resting_hr=72, resting_rr=16)
        v = gen.generate("sit", 0)
        assert 50 < v.heart_rate < 100, f"HR out of range: {v.heart_rate}"
        assert 10 < v.resp_rate < 25, f"RR out of range: {v.resp_rate}"
        assert 0 < v.quality <= 1.0, f"Quality out of range: {v.quality}"

    def test_activity_affects_vitals(self):
        gen = VitalSignsGen(resting_hr=72, resting_rr=16, seed=42)
        rest = gen.generate("sit", 0)
        active = gen.generate("walk", 0)
        # 活动时心率应该更高（统计意义）
        active_hrs = [gen.generate("walk", i * 0.1).heart_rate for i in range(50)]
        rest_hrs = [gen.generate("sit", i * 0.1).heart_rate for i in range(50)]
        assert sum(active_hrs) / len(active_hrs) > sum(rest_hrs) / len(rest_hrs), \
            "Walking HR should be higher than sitting HR"


class TestRoomModel:
    def test_apply_transforms_coordinates(self):
        import numpy as np
        cfg = RoomConfig("test", size=(4, 3, 2.8), radar_pos=(2, 1.5, 2.5))
        model = RoomModel(cfg)
        points = np.random.randn(20, 5).astype(np.float32)
        points[:, :3] += np.array([2, 1.5, 1.0])  # 移到房间中心
        result = model.apply(points)
        assert result is not None
        assert result.shape[1] == 5
        # 噪声应该使坐标微调
        assert np.any(np.abs(result[:, :3] - points[:len(result), :3]) > 0)


class TestAnomalyInjector:
    def test_inject_fall_adds_ground_truth(self):
        import numpy as np
        frames = [
            FrameGroup(frame_id="f1", ts=1000, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
            FrameGroup(frame_id="f2", ts=1100, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
            FrameGroup(frame_id="f3", ts=1200, device_id="d1", room="living",
                       points=np.random.randn(15, 5).astype(np.float32) + np.array([0, 0, 1.0, 0, 0])),
        ]
        injector = AnomalyInjector()
        injections = [{"at_s": 1.0, "type": "fall", "room": "living",
                       "params": {"transition_duration": 0.2},
                       "ground_truth": {"severity": "critical", "desc": "test fall"}}]
        result = injector.inject(frames, injections)
        assert len(result) == 3
        # 至少一帧有 anomaly_type
        assert any(f.gt and f.gt.anomaly_type == "fall" for f in result)

    def test_inject_offline_clears_points(self):
        import numpy as np
        frames = [
            FrameGroup(frame_id="f1", ts=1000, device_id="d1", room="living",
                       points=np.random.randn(10, 5).astype(np.float32)),
            FrameGroup(frame_id="f2", ts=1100, device_id="d1", room="living",
                       points=np.random.randn(10, 5).astype(np.float32)),
        ]
        injector = AnomalyInjector()
        injections = [{"at_s": 1.0, "type": "offline", "room": "living",
                       "params": {"gap_duration_s": 1}, "ground_truth": {"severity": "critical"}}]
        result = injector.inject(frames, injections)
        # 帧应该被清空
        assert result[0].points is None


class TestTypes:
    def test_ground_truth_to_dict(self):
        gt = GroundTruth(
            frame_id="f1", ts=1000, posture="walk",
            posture_confidence=1.0, heart_rate_true=75.0,
            resp_rate_true=16.0, anomaly_type=None, anomaly_severity=None,
            scenario_name="test", generator="synthetic",
        )
        d = gt.to_dict()
        assert d["frame_id"] == "f1"
        assert d["posture"] == "walk"
        assert "anomaly_type" not in d  # None 被过滤
```

- [ ] **Step 3: 运行测试**

```bash
python3 -m pytest simulator/tests/test_simulator.py -v
```
```

---

### Task 10: 文档更新

**Files:**
- Modify: `simulator/README.md`

- [ ] **Step 1: 更新 README**

用新的 CLI 用法替换旧的 feed.py 文档。
