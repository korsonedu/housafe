"""漂移实验共享工具 — 仿真数据生成 + S_t 提取 + 评估。

从 fit_and_eval.py 抽取，供 experiments.py 复用。
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.generators import ReplayGenerator
from simulator.calibrate import ACTION_NAMES as _SIM_ACTION_NAMES
from ai.encoder.encoder_model import EncoderModel

# ── 常量 ──────────────────────────────────────────────

FPS = 10
WINDOW = 32
MAX_POINTS = 64

# 每日活动模板（正常基线）
NORMAL_PATTERN = [
    ("lie", 0, 6),
    ("stand", 6, 6.5), ("walk", 6.5, 6.7), ("sit", 6.7, 7.2),
    ("stand", 7.2, 7.3), ("walk", 7.3, 7.5),
    ("sit", 7.5, 9), ("stand", 9, 9.1), ("walk", 9.1, 9.3),
    ("squat", 9.3, 9.35), ("stand", 9.35, 9.5), ("sit", 9.5, 11.5),
    ("stand", 11.5, 11.6), ("walk", 11.6, 11.8), ("sit", 11.8, 13),
    ("stand", 13, 13.1), ("walk", 13.1, 13.3), ("sit", 13.3, 15),
    ("stand", 15, 15.1), ("walk", 15.1, 15.3),
    ("lean_left", 15.3, 15.33), ("lean_right", 15.33, 15.36),
    ("sit", 15.36, 17.5),
    ("stand", 17.5, 17.6), ("walk", 17.6, 17.8), ("sit", 17.8, 18.5),
    ("stand", 18.5, 18.6), ("walk", 18.6, 18.7), ("sit", 18.7, 20),
    ("stand", 20, 20.1), ("walk", 20.1, 20.2), ("sit", 20.2, 22),
    ("stand", 22, 22.1), ("walk", 22.1, 22.2), ("lie", 22.2, 24),
]

# 默认漂移修饰符（duration 变化）
DRIFT_MODIFIERS = {
    "walk": -0.5,
    "squat": -0.8,
    "lean_left": -0.7,
    "lean_right": -0.7,
    "stand": -0.3,
    "sit": 0.5,
}


# ── 点云工具 ──────────────────────────────────────────

def pad_to_64(pts: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return np.zeros((MAX_POINTS, 5), dtype=np.float32)
    if n >= MAX_POINTS:
        return pts[rng.choice(n, MAX_POINTS, replace=False)]
    extra = MAX_POINTS - n
    idx = rng.randint(0, n, size=extra)
    jitter = rng.randn(extra, 5).astype(np.float32) * 0.01
    return np.vstack([pts, pts[idx] + jitter])


# ── 漂移注入 ──────────────────────────────────────────

def apply_drift(pattern: list, drift_level: float,
                modifiers: dict | None = None) -> list:
    """对活动模板施加漂移。drift_level 0-1, 0=正常 1=重度漂移。"""
    if drift_level <= 0:
        return pattern
    if modifiers is None:
        modifiers = DRIFT_MODIFIERS

    modified = []
    for action, start_h, end_h in pattern:
        dur = end_h - start_h
        m = modifiers.get(action, 0.0)
        new_dur = dur * (1.0 + m * drift_level)
        new_dur = max(new_dur, 0.02)
        new_end = start_h + new_dur
        if new_end > 24:
            new_end = 24
        modified.append((action, start_h, new_end))
    return modified


# ── 数据生成 ──────────────────────────────────────────

def generate_day(point_frames_out: list, timestamps_out: list,
                 day_index: int, subjects: list[int], replay: ReplayGenerator,
                 seconds_per_day: float, drift_level: float = 0.0,
                 drift_modifiers: dict | None = None,
                 rng: np.random.RandomState | None = None,
                 subject_filter: int | None = None):
    """生成一天的点云帧和时间戳。

    Args:
        point_frames_out, timestamps_out: 追加到这些列表
        day_index: day 0 = baseline day 1
        subjects: 可用被试 ID 列表
        seconds_per_day: 每天仿真时长（秒）
        drift_level: 0.0 = 正常, 1.0 = 最大漂移
        drift_modifiers: 自定义漂移修饰符
        subject_filter: 如果指定，只用这个被试的点云
    """
    if rng is None:
        rng = np.random.RandomState(42 + day_index)

    pattern = apply_drift(NORMAL_PATTERN, drift_level, drift_modifiers)

    base_day_ts = 1753891200000 + day_index * 86400 * 1000
    total_pattern_hours = 24.0
    time_scale = seconds_per_day / total_pattern_hours

    sim_time = 0.0
    pattern_idx = 0
    max_iterations = len(pattern) * 10  # 安全上限，防止死循环

    for _iter in range(max_iterations):
        if sim_time >= seconds_per_day - 0.05:  # 接近完成，退出
            break

        if pattern_idx >= len(pattern):
            pattern_idx = 0

        action, start_h, end_h = pattern[pattern_idx]
        dur_h = end_h - start_h
        dur_s = dur_h * time_scale
        dur_s = min(dur_s, seconds_per_day - sim_time)
        if dur_s < 0.05:  # 小于 0.05 秒的片段跳过，sim_time 不增加但继续下一个
            pattern_idx += 1
            continue

        # subject 选择
        if subject_filter is not None:
            subject_id = subject_filter
        else:
            subject_id = rng.choice(subjects)

        if action == "lie":
            frames = replay.generate_sequence("squat", dur_s, subject_ids=[subject_id])
        else:
            frames = replay.generate_sequence(action, dur_s, subject_ids=[subject_id])

        if not frames:
            pattern_idx += 1
            continue

        n_frames = len(frames)
        for i, pts in enumerate(frames):
            point_frames_out.append(pts.astype(np.float32))
            frac = i / max(n_frames, 1)
            frame_hour = start_h + frac * dur_h
            frame_hour = frame_hour % 24
            timestamps_out.append(int(base_day_ts + frame_hour * 3600 * 1000))

        sim_time += dur_s
        pattern_idx += 1


# ── S_t 提取 ──────────────────────────────────────────

def extract_S_sequence(encoder: EncoderModel, point_frames: list[np.ndarray],
                       device: torch.device, stride: int = 4,
                       timestamps_ms: list[int] | None = None,
                       ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """高效逐帧提取 S_t。使用 backbone+buffer+TCN。

    返回: (M, 256) where M ≈ len(point_frames) / stride
    如果提供 timestamps_ms，返回 (S, ts_aligned)
    """
    N = len(point_frames)
    if N < WINDOW:
        if timestamps_ms is not None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        return np.array([], dtype=np.float32)

    rng = np.random.RandomState(42)
    S_list = []
    ts_list = []
    bb_buffer = []

    for i in range(N):
        pts = pad_to_64(point_frames[i], rng)
        pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
        xyz, feats = pts_t[:, :, :3], pts_t[:, :, 3:]

        with torch.no_grad():
            bb = encoder.backbone(xyz, feats)

        bb_buffer.append(bb.squeeze(0))
        if len(bb_buffer) > WINDOW:
            bb_buffer = bb_buffer[-WINDOW:]

        if len(bb_buffer) < WINDOW:
            continue
        if i % stride != 0:
            continue

        seq = torch.stack(list(bb_buffer)).unsqueeze(0)
        with torch.no_grad():
            S_seq = encoder.tcn(seq)
        S_list.append(S_seq[0, -1].cpu().numpy())
        if timestamps_ms is not None and i < len(timestamps_ms):
            ts_list.append(timestamps_ms[i])

    if not S_list:
        if timestamps_ms is not None:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        return np.array([], dtype=np.float32)

    S_arr = np.array(S_list, dtype=np.float32)
    if timestamps_ms is not None:
        return S_arr, np.array(ts_list, dtype=np.int64)
    return S_arr


# ── 时间戳对齐 ────────────────────────────────────────

def align_timestamps_and_boundaries(
    all_timestamps: list[int],
    day_boundaries: list[int],
    W: int = WINDOW,
    stride: int = 4,
    S_len: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """将原始时间戳和日边界对齐到 S_t 采样网格。

    Returns:
        ts_aligned: (S_len,) 对齐后的时间戳
        day_S_boundaries: S_t 索引的日边界
    """
    ts_raw = np.array(all_timestamps[W - 1:], dtype=np.int64)
    ts_aligned = ts_raw[::stride]

    day_S_boundaries = []
    for b in day_boundaries:
        if b < W:
            day_S_boundaries.append(0)
        else:
            s_idx = max(0, (b - W) // stride)
            day_S_boundaries.append(min(s_idx, S_len if S_len else len(ts_aligned)))

    if S_len is not None:
        ts_aligned = ts_aligned[:S_len]

    return ts_aligned, day_S_boundaries


# ── Encoder 加载 ──────────────────────────────────────

ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "encoder_best.pt",
)


def load_encoder(device: torch.device) -> EncoderModel:
    encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
    ckpt = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder


# ── 分析性 drift 评分 ─────────────────────────────────

def pattern_to_action_dist(pattern: list) -> dict[str, float]:
    """从活动模板计算归一化动作时间占比。"""
    dist: dict[str, float] = {}
    for action, start, end in pattern:
        dur = end - start
        dist[action] = dist.get(action, 0.0) + dur
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two action distributions."""
    actions = sorted(set(list(p.keys()) + list(q.keys())))
    p_arr = np.array([p.get(a, 0.0) for a in actions])
    q_arr = np.array([q.get(a, 0.0) for a in actions])
    eps = 1e-10
    p_arr = np.clip(p_arr, eps, 1.0)
    q_arr = np.clip(q_arr, eps, 1.0)
    m = (p_arr + q_arr) / 2.0
    return float((np.sum(p_arr * np.log(p_arr / m)) +
                  np.sum(q_arr * np.log(q_arr / m))) / 2.0)


def analytical_drift_score(drift_level: float,
                           baseline_pattern: list | None = None,
                           drift_modifiers: dict | None = None) -> float:
    """分析性 drift 评分：baseline pattern vs drifted pattern 的 JS divergence。

    这是任何基于动作分布分类的检测器的理论上限。
    """
    if baseline_pattern is None:
        baseline_pattern = NORMAL_PATTERN
    if drift_modifiers is None:
        drift_modifiers = DRIFT_MODIFIERS

    baseline_dist = pattern_to_action_dist(baseline_pattern)
    drifted_pattern = apply_drift(baseline_pattern, drift_level, drift_modifiers)
    drifted_dist = pattern_to_action_dist(drifted_pattern)
    return js_divergence(baseline_dist, drifted_dist)


def estimate_noise_floor(encoder, replay, subjects: list[int],
                         subject_filter: int,
                         seconds_per_day: float = 60.0,
                         n_days: int = 14,
                         device=None) -> float:
    """通过仿真标定日常随机波动导致的 JS divergence 噪声水平。

    生成 n_days 个 dl=0 的天，两两比较得到 JS divergence 的经验分布。
    返回 95 分位数作为检测阈值建议。
    """
    import torch
    from ai.drift.detector import ExplainableDriftDetector
    if device is None:
        device = torch.device("cpu")

    edd = ExplainableDriftDetector(encoder)
    pf, ts = [], []
    rng = np.random.RandomState(42)
    for d in range(n_days):
        generate_day(pf, ts, day_index=d, subjects=subjects, replay=replay,
                     seconds_per_day=seconds_per_day, drift_level=0.0,
                     rng=rng, subject_filter=subject_filter)

    S_all, ts_all = extract_S_sequence(encoder, pf, device, stride=4,
                                        timestamps_ms=ts)
    day_S_bounds = compute_day_boundaries_from_ts(ts_all) if 'compute_day_boundaries_from_ts' in dir() else None

    # 需要 compute_day_boundaries_from_ts
    from ai.drift.experiments import compute_day_boundaries_from_ts as _cb
    day_S_bounds = _cb(ts_all)

    # 每天 vs 所有其他天（pooled baseline）
    edd.fit_baseline(S_all, ts_all)
    js_values = []
    for d in range(n_days):
        s, e = day_S_bounds[d], day_S_bounds[d + 1] if d + 1 < len(day_S_bounds) else len(S_all)
        if e <= s:
            continue
        r = edd.compare(S_all[s:e], ts_all[s:e])
        js_values.append(r.overall_drift)

    return float(np.percentile(js_values, 95)) if js_values else 0.001
