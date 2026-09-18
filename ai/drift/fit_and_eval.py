#!/usr/bin/env python3
"""
事前健康管理 — 漂移检测离线验证。

流程:
  1. 生成多天正常 + 漂移仿真数据（7天基线 + 14天渐进漂移）
  2. Encoder 提取 S_t
  3. 在基线天拟合 LatentBaseline
  4. 运行 DistributionDriftDetector（Direction A）和 ExplainableDriftDetector（Direction C）
  5. 评估漂移检测效果

用法:
  python -m ai.drift.fit_and_eval --baseline-days 7 --drift-days 14

数据生成方案:
  - 每个模拟天 = 2min 仿真时间（DAILY_PATTERN 1h→30s，一天 ≈ 12min real）
  - 实际用 --day-minutes 控制每天时长
  - 漂移注入: 从第 8 天开始逐渐减少 walk/stand，增加 sit
"""

import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.generators import ReplayGenerator
from ai.encoder.encoder_model import EncoderModel
from ai.baseline.latent_baseline import LatentBaseline, context_key, _utc_ms_to_datetime
from ai.drift.detector import (
    DistributionDriftDetector,
    ExplainableDriftDetector,
    DayDriftResult,
    ActionDriftResult,
    ACTION_LABELS,
)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "public", "processed", "3dpchm_frames.npz",
)
ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "encoder_best.pt",
)
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
)

FPS = 10
WINDOW = 32
MAX_POINTS = 64
STRIDE = 4  # S_t 采样间隔

# ── 每日活动模板（正常基线）─────────────────────────
# 压缩版：1h → 30s 仿真时间

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

# 漂移模式：逐渐减少 walk/stand/squat/lean，增加 sit
# drift_level: 0.0=正常, 0.5=中度, 1.0=重度
DRIFT_MODIFIERS = {
    "walk": -0.5,     # 每级漂移 walk 时长 -50%
    "squat": -0.8,    # squat -80%
    "lean_left": -0.7,
    "lean_right": -0.7,
    "stand": -0.3,
    "sit": 0.5,       # sit 时长补偿 +50%
}


def apply_drift(pattern: list, drift_level: float) -> list:
    """对活动模板施加漂移。drift_level 0-1, 0=正常 1=重度漂移。"""
    if drift_level <= 0:
        return pattern

    modified = []
    for action, start_h, end_h in pattern:
        dur = end_h - start_h
        modifier = DRIFT_MODIFIERS.get(action, 0.0)
        # 漂移效果 = modifier * drift_level
        new_dur = dur * (1.0 + modifier * drift_level)
        new_dur = max(new_dur, 0.02)  # 最少 0.02h ≈ 1min
        new_end = start_h + new_dur
        # 不超过 24h
        if new_end > 24:
            new_end = 24
        modified.append((action, start_h, new_end))

    return modified


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


def generate_day(point_frames_out: list, timestamps_out: list,
                 day_index: int, subjects: list[int], replay: ReplayGenerator,
                 seconds_per_day: float, drift_level: float = 0.0,
                 rng: np.random.RandomState | None = None):
    """生成一天的点云帧和时间戳。

    Args:
        point_frames_out, timestamps_out: 追加到这些列表
        day_index: day 0 = baseline day 1
        seconds_per_day: 每天仿真时长（秒）
        drift_level: 0.0 = 正常, 1.0 = 最大漂移
    """
    if rng is None:
        rng = np.random.RandomState(42 + day_index)

    pattern = apply_drift(NORMAL_PATTERN, drift_level)

    # 每天 base timestamp: 2026-07-(31+day) 00:00 Beijing
    base_day_ts = 1753891200000 + day_index * 86400 * 1000  # ms

    total_pattern_hours = 24.0
    time_scale = seconds_per_day / total_pattern_hours  # 1 pattern hour → X real seconds

    sim_time = 0.0
    pattern_idx = 0
    max_iterations = len(pattern) * 10

    for _iter in range(max_iterations):
        if sim_time >= seconds_per_day - 0.05:
            break
        if pattern_idx >= len(pattern):
            pattern_idx = 0

        action, start_h, end_h = pattern[pattern_idx]
        dur_h = end_h - start_h
        dur_s = dur_h * time_scale
        dur_s = min(dur_s, seconds_per_day - sim_time)
        if dur_s < 0.05:
            pattern_idx += 1
            continue

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
            # 时间戳映射
            frac = i / max(n_frames, 1)
            frame_hour = start_h + frac * dur_h
            frame_hour = frame_hour % 24
            timestamps_out.append(int(base_day_ts + frame_hour * 3600 * 1000))

        sim_time += dur_s
        pattern_idx += 1


def extract_S_sequence(encoder: EncoderModel, point_frames: list[np.ndarray],
                       device: torch.device, stride: int = STRIDE) -> np.ndarray:
    """高效逐帧提取 S_t。"""
    N = len(point_frames)
    if N < WINDOW:
        return np.array([])

    rng = np.random.RandomState(42)
    S_list = []
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

    return np.array(S_list, dtype=np.float32)


# ── 评估指标 ──────────────────────────────────────────


@dataclass
class DriftEvalResult:
    baseline_days: int
    drift_days: int
    day_results: list[dict] = field(default_factory=list)
    drift_score_correlation: float = 0.0       # drift_level vs drift_score Spearman
    action_drift_correlation: float = 0.0
    baseline_vs_drift_auc: float = 0.0         # 二分类: baseline day vs drifted day
    mild_vs_severe_auc: float = 0.0

    def report(self) -> str:
        lines = [
            "=" * 64,
            "  事前健康管理 — 漂移检测评估",
            "=" * 64,
            f"  基线天数: {self.baseline_days}  |  漂移天数: {self.drift_days}",
            f"  ─────────────────────────────────────",
            f"  Distribution Drift (A):",
            f"    漂移分数 vs 注入级别 Spearman R: {self.drift_score_correlation:.3f}",
            f"    基线 vs 漂移日 AUC:              {self.baseline_vs_drift_auc:.3f}",
            f"    轻 vs 重漂移 AUC:                {self.mild_vs_severe_auc:.3f}",
            f"  ─────────────────────────────────────",
            f"  Explainable Drift (C):",
            f"    JS 距离 vs 注入级别 Spearman R:   {self.action_drift_correlation:.3f}",
            "=" * 64,
        ]

        if self.day_results:
            lines.append("\n  逐日详情:")
            lines.append(f"  {'Day':>4s} {'DriftLv':>7s} {'DriftScore':>10s} {'JSDist':>7s}  Contexts")
            lines.append("  " + "-" * 56)
            for r in self.day_results:
                ctx_str = " ".join(
                    f"{c}={s:.2f}" for c, s in sorted(r.get("per_context", {}).items())
                    if s > 0.1
                )[:40]
                lines.append(
                    f"  {r['day']:>4d} {r['drift_level']:>7.2f} "
                    f"{r['drift_score']:>10.4f} {r['js_distance']:>7.4f}  {ctx_str}"
                )

        return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="漂移检测离线验证")
    parser.add_argument("--baseline-days", type=int, default=7,
                        help="正常基线天数")
    parser.add_argument("--drift-days", type=int, default=14,
                        help="漂移天数（渐进注入）")
    parser.add_argument("--day-seconds", type=float, default=60.0,
                        help="每天仿真时长（秒），默认 60s/天")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--output", default=None,
                        help="LatentBaseline 保存路径")
    parser.add_argument("--save-data", action="store_true",
                        help="保存生成的 S_t 数据供后续分析")
    args = parser.parse_args()

    device = torch.device(args.device)
    total_days = args.baseline_days + args.drift_days
    rng = np.random.RandomState(42)

    # ── 加载 Encoder ──────────────────────────────────
    print(f"加载 Encoder: {ENCODER_PATH}")
    encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
    ckpt = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()

    # ── 加载 ReplayGenerator ─────────────────────────
    print(f"加载数据: {DATASET_PATH}")
    replay = ReplayGenerator(DATASET_PATH)

    # ── Step 1: 生成多天仿真数据 ─────────────────────
    print(f"\n{'='*60}")
    print(f"Step 1: 生成 {total_days} 天仿真数据 ({args.day_seconds}s/天)")
    print(f"  baseline days 1-{args.baseline_days}: drift_level=0")
    print(f"  drift days {args.baseline_days+1}-{total_days}: drift_level 0→1 线性递增")
    print(f"{'='*60}")

    all_point_frames: list[np.ndarray] = []
    all_timestamps: list[int] = []
    day_boundaries: list[int] = [0]  # 每天起始帧索引
    day_drift_levels: list[float] = []

    t0 = time.time()
    for day in range(total_days):
        if day < args.baseline_days:
            drift_level = 0.0
        else:
            # 线性递增 0 → 1
            progress = (day - args.baseline_days) / max(args.drift_days - 1, 1)
            drift_level = progress  # 0 → 1

        generate_day(all_point_frames, all_timestamps,
                     day_index=day, subjects=args.subjects, replay=replay,
                     seconds_per_day=args.day_seconds, drift_level=drift_level,
                     rng=rng)
        day_boundaries.append(len(all_point_frames))
        day_drift_levels.append(drift_level)

    print(f"  点云帧总数: {len(all_point_frames):,}  ({time.time() - t0:.0f}s)")

    # ── Step 2: 提取 S_t ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2: Encoder 提取 S_t")
    print(f"{'='*60}")
    t1 = time.time()

    S_all = extract_S_sequence(encoder, all_point_frames, device, stride=STRIDE)
    # 时间戳对齐
    ts_raw = np.array(all_timestamps[WINDOW - 1:], dtype=np.int64)
    ts_all = ts_raw[::STRIDE]
    ts_all = ts_all[:len(S_all)]

    # 日边界也要对齐
    day_S_boundaries = []
    for b in day_boundaries:
        if b < WINDOW:
            day_S_boundaries.append(0)
            continue
        s_idx = max(0, (b - WINDOW) // STRIDE)
        day_S_boundaries.append(min(s_idx, len(S_all)))

    print(f"  S_t: {S_all.shape}  ({time.time() - t1:.0f}s)")

    if len(S_all) < 500:
        print("ERROR: S_t 太少，请增加 --day-seconds")
        sys.exit(1)

    # ── Step 3: 拟合 LatentBaseline（仅基线天）───────
    print(f"\n{'='*60}")
    print(f"Step 3: 在基线天 (1-{args.baseline_days}) 上拟合 LatentBaseline")
    print(f"{'='*60}")
    t2 = time.time()

    baseline_end_idx = day_S_boundaries[args.baseline_days]
    S_baseline = S_all[:baseline_end_idx]
    ts_baseline = ts_all[:baseline_end_idx]

    print(f"  基线帧数: {len(S_baseline):,}")

    lb = LatentBaseline()
    lb.fit(S_baseline, ts_baseline)

    if not lb.is_ready:
        print("ERROR: LatentBaseline 未就绪（基线数据不足）")
        print(f"  请增加 --day-seconds（当前 {args.day_seconds}s/天）或 --baseline-days")
        sys.exit(1)

    n_contexts = len(lb._gmms)
    print(f"  Contexts: {n_contexts}")
    for ctx in sorted(lb._gmms.keys()):
        print(f"    {ctx}: {len(lb._scores.get(ctx, [])):,} samples, "
              f"NLL mean={lb._nll_stats.get(ctx, {}).get('mean', 0):.2f}")

    # 保存基线
    output_path = args.output or os.path.join(CHECKPOINT_DIR, "latent_baseline_drift.pkl")
    lb.save(output_path)
    print(f"  保存到: {output_path}  ({time.time() - t2:.0f}s)")

    # ── Step 4: 逐日评估漂移 ─────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: 逐日漂移评估")
    print(f"{'='*60}")

    drift_detector = DistributionDriftDetector(lb)
    explainer = ExplainableDriftDetector(encoder, device=str(device))

    # 用基线天 S_t 拟合解释器基线
    explainer.fit_baseline(S_baseline, ts_baseline)
    print(f"  ExplainableDriftDetector 基线已拟合 ({len(S_baseline):,} frames)")

    day_results = []
    for day in range(total_days):
        start_idx = day_S_boundaries[day]
        end_idx = day_S_boundaries[day + 1] if day + 1 < len(day_S_boundaries) else len(S_all)
        if end_idx <= start_idx:
            continue

        S_day = S_all[start_idx:end_idx]
        ts_day = ts_all[start_idx:end_idx]

        # A: 分布漂移
        drift_result = drift_detector.score_day(S_day, ts_day)

        # C: 可解释漂移
        action_result = explainer.compare(S_day, ts_day)

        day_results.append({
            "day": day + 1,
            "drift_level": day_drift_levels[day],
            "drift_score": drift_result.overall_score,
            "mean_nll": drift_result.mean_nll,
            "n_frames": drift_result.n_total_frames,
            "js_distance": action_result.overall_drift,
            "dominant_context": drift_result.dominant_context,
            "per_context": drift_result.per_context,
            "action_summary": action_result.summary,
            "action_changes": action_result.top_changes,
        })

        # 打印
        marker = "🟢" if day < args.baseline_days else ("🟡" if day_drift_levels[day] < 0.5 else "🔴")
        print(f"  {marker} Day {day+1:>2d}  "
              f"drift_lv={day_drift_levels[day]:.2f}  "
              f"drift_score={drift_result.overall_score:.4f}  "
              f"JS={action_result.overall_drift:.4f}  "
              f"nll={drift_result.mean_nll:.2f}  "
              f"ctx={drift_result.dominant_context}  "
              f"frames={drift_result.n_total_frames}")

    # ── Step 5: 汇总指标 ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 5: 汇总评估")
    print(f"{'='*60}")

    drift_levels = np.array([r["drift_level"] for r in day_results])
    drift_scores = np.array([r["drift_score"] for r in day_results])
    js_distances = np.array([r["js_distance"] for r in day_results])

    # Spearman 相关：drift_score vs drift_level
    from scipy.stats import spearmanr as _spearmanr
    try:
        corr_a, p_a = _spearmanr(drift_levels, drift_scores)
    except Exception:
        corr_a, p_a = 0.0, 1.0

    try:
        corr_c, p_c = _spearmanr(drift_levels, js_distances)
    except Exception:
        corr_c, p_c = 0.0, 1.0

    # AUC: baseline days vs drift days
    from sklearn.metrics import roc_auc_score as _roc_auc
    is_baseline = np.array([r["drift_level"] == 0 for r in day_results])
    if is_baseline.any() and (~is_baseline).any():
        auc_bl = _roc_auc(is_baseline.astype(int), drift_scores)
        # invert: baseline should have low drift score
        auc_bl = max(auc_bl, 1 - auc_bl)
    else:
        auc_bl = 0.0

    # AUC: mild drift vs severe drift (split at drift_level 0.5)
    drift_days_mask = ~is_baseline
    if drift_days_mask.sum() >= 4:
        drift_day_levels = drift_levels[drift_days_mask]
        drift_day_scores = drift_scores[drift_days_mask]
        is_severe = drift_day_levels >= 0.5
        if is_severe.any() and (~is_severe).any():
            auc_ms = _roc_auc(is_severe.astype(int), drift_day_scores)
            auc_ms = max(auc_ms, 1 - auc_ms)
        else:
            auc_ms = 0.0
    else:
        auc_ms = 0.0

    result = DriftEvalResult(
        baseline_days=args.baseline_days,
        drift_days=args.drift_days,
        day_results=day_results,
        drift_score_correlation=corr_a,
        action_drift_correlation=corr_c,
        baseline_vs_drift_auc=auc_bl,
        mild_vs_severe_auc=auc_ms,
    )

    print(result.report())

    # ── 可解释性展示 ─────────────────────────────────
    if day_results:
        # 找漂移最大的一天展示详细分解
        worst_day = max(day_results, key=lambda r: r["drift_score"])
        print(f"\n  ── 最严重漂移日 (Day {worst_day['day']}, drift_lv={worst_day['drift_level']:.2f}) ──")
        print(f"  A. 分布漂移: {worst_day['drift_score']:.4f}")
        print(f"  C. 可解释分解:")
        print(f"     {worst_day['action_summary']}")
        if worst_day["action_changes"]:
            print(f"     Top changes:")
            for c in worst_day["action_changes"]:
                direction = "↑" if c["direction"] == "increase" else "↓"
                print(f"       {c['action']:>15s}: {c['baseline_pct']:>5.1f}% → "
                      f"{c['current_pct']:>5.1f}% ({direction}{abs(c['delta_pct']):.1f}pp)")

    # ── 可选: 保存数据 ───────────────────────────────
    if args.save_data:
        data_path = os.path.join(CHECKPOINT_DIR, "drift_eval_data.npz")
        np.savez_compressed(data_path,
                           S_all=S_all, ts_all=ts_all,
                           day_boundaries=np.array(day_S_boundaries),
                           day_drift_levels=np.array(day_drift_levels))
        print(f"\n  数据保存到: {data_path}")


if __name__ == "__main__":
    main()
