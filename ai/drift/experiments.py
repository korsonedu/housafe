#!/usr/bin/env python3
"""事前健康管理 — 仿真实验主入口。

用法:
  python3 -m ai.drift.experiments --exp 1   # 跨人泛化
  python3 -m ai.drift.experiments --exp 2   # 动作质量退化
  python3 -m ai.drift.experiments --exp 3   # 异常动作检测
  python3 -m ai.drift.experiments --exp 4   # Context 时段敏感性
  python3 -m ai.drift.experiments --exp 5   # 消融实验
  python3 -m ai.drift.experiments --exp 6   # 传感器鲁棒性
  python3 -m ai.drift.experiments --exp all # 全部

公共参数:
  --day-seconds N     每天仿真秒数 (default 60)
  --seeds N           随机种子数 (default 3)
  --device cpu/cuda   设备 (default cpu)
  --subjects ...      被试列表 (default 0-6)
"""

import argparse
import os
import sys
import time
import json
import hashlib
import numpy as np
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.generators import ReplayGenerator
from ai.baseline.latent_baseline import LatentBaseline
from ai.drift.common import (
    load_encoder, generate_day, extract_S_sequence, apply_drift,
    align_timestamps_and_boundaries, NORMAL_PATTERN, DRIFT_MODIFIERS,
    FPS, WINDOW, MAX_POINTS,
)
from ai.drift.detector import (
    DistributionDriftDetector,
    ExplainableDriftDetector,
)
from ai.drift.report import (
    write_report,
    report_exp1_cross_person,
    report_exp2_gait,
    report_exp3_anomaly,
    report_exp4_context,
    report_exp5_ablation,
    report_exp6_robustness,
)

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "public", "processed", "3dpchm_frames.npz",
)
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reports", "checkpoints",
)


# ── 断点续传 ──────────────────────────────────────────

class Checkpoint:
    """每次 combo 完成后写入，恢复时跳过已完成组合。"""

    def __init__(self, exp_id: str):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.path = os.path.join(CHECKPOINT_DIR, f"exp{exp_id}.json")
        self.completed: set[str] = set()
        self._results: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.completed = set(data.get("completed", []))
                self._results = data.get("results", [])
                if self.completed:
                    print(f"  [断点] 已恢复 {len(self.completed)} 个已完成项, "
                          f"{len(self._results)} 条结果")
            except Exception:
                pass

    def is_done(self, *keys) -> bool:
        return self.key(*keys) in self.completed

    def key(self, *keys) -> str:
        raw = "|".join(str(k) for k in keys)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def save_one(self, result: dict, *keys):
        self._results.append(result)
        self.completed.add(self.key(*keys))
        with open(self.path, "w") as f:
            json.dump({
                "completed": list(self.completed),
                "results": self._results,
            }, f)

    def get_results(self) -> list[dict]:
        return list(self._results)

# ── 共享工具 ──────────────────────────────────────────


def get_subjects():
    """从数据集加载被试 ID 列表。"""
    data = np.load(DATASET_PATH, allow_pickle=True)
    return sorted(set(int(s) for s in data["subject_ids"]))


def compute_day_boundaries_from_ts(ts_all: np.ndarray, base_ts: int = 1753891200000,
                                    ms_per_day: int = 86400 * 1000) -> list[int]:
    """从对齐后的时间戳计算日边界（S_t 索引）。"""
    boundaries = [0]
    for day in range(60):  # max 60 days
        threshold = base_ts + (day + 1) * ms_per_day
        idx = int(np.searchsorted(ts_all, threshold))
        if idx > boundaries[-1]:
            boundaries.append(idx)
        if idx >= len(ts_all):
            break
    if boundaries[-1] < len(ts_all):
        boundaries.append(len(ts_all))
    return boundaries


def _run_daily_simulation(total_days: int, baseline_days: int,
                         day_seconds: float, subjects: list[int],
                         drift_fn=None,
                         subject_filter: int | None = None,
                         seeds: int = 1) -> list[dict]:
    """运行多天仿真，返回逐日数据。

    Args:
        total_days: 总天数
        baseline_days: 前 N 天为基线（drift_level=0）
        day_seconds: 每天仿真秒数
        subjects: 可用被试列表
        drift_fn: callable(day_index) → drift_level，None=线性 0→1
        subject_filter: 限定被试
        seeds: 随机种子数

    Returns:
        [{"day": int, "S": ndarray, "ts": ndarray, "drift_level": float}, ...]
    """
    all_results = []
    base_seed = 42
    replay = ReplayGenerator(DATASET_PATH)  # 全局复用

    for seed_idx in range(seeds):
        seed = base_seed + seed_idx
        rng = np.random.RandomState(seed)

        point_frames: list = []
        timestamps: list = []
        day_boundaries = [0]
        day_drift_levels = []

        for day in range(total_days):
            if day < baseline_days:
                dl = 0.0
            elif drift_fn is not None:
                dl = drift_fn(day - baseline_days, total_days - baseline_days)
            else:
                progress = (day - baseline_days) / max(total_days - baseline_days - 1, 1)
                dl = progress  # linear 0→1

            generate_day(point_frames, timestamps,
                         day_index=day, subjects=subjects,
                         replay=replay,
                         seconds_per_day=day_seconds,
                         drift_level=dl,
                         rng=rng,
                         subject_filter=subject_filter)
            day_boundaries.append(len(point_frames))
            day_drift_levels.append(dl)

        # 提取 S_t（返回与 S 同步的时间戳）
        encoder = load_encoder(torch.device("cpu"))
        S_all, ts_all = extract_S_sequence(encoder, point_frames, torch.device("cpu"),
                                            stride=4, timestamps_ms=timestamps)
        day_S_bounds = compute_day_boundaries_from_ts(ts_all)

        if len(S_all) < 500:
            print(f"  [WARN] seed={seed}: only {len(S_all)} S_t frames, skipping")
            continue

        # 按天切分
        day_data = []
        for day in range(total_days):
            start = day_S_bounds[day]
            end = day_S_bounds[day + 1] if day + 1 < len(day_S_bounds) else len(S_all)
            if end <= start:
                continue
            day_data.append({
                "day": day + 1,
                "S": S_all[start:end],
                "ts": ts_all[start:end],
                "drift_level": day_drift_levels[day],
                "seed": seed,
            })

        all_results.extend(day_data)
        print(f"  seed={seed}: {len(day_data)} days, {len(S_all)} S_t total")

    return all_results


# ═══════════════════════════════════════════════════════
# Experiment 1: 跨人泛化
# ═══════════════════════════════════════════════════════

def exp1_cross_person(args):
    """留一被试交叉验证 + 冷启动分析。"""
    print("=" * 60)
    print("Experiment 1: 跨人泛化 (Leave-One-Subject-Out)")
    print("=" * 60)

    all_subjects = get_subjects()
    print(f"  被试: {all_subjects}")

    ckpt = Checkpoint("1")
    results = ckpt.get_results()
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)

    # 冷启动时长：个人数据积累天数
    cold_start_days = [0, 1, 3, 7, 14]

    for test_subj in all_subjects:
        if ckpt.is_done(test_subj):
            print(f"  被试={test_subj} [已跳过]")
            continue
        train_subjs = [s for s in all_subjects if s != test_subj]
        print(f"\n  测试被试={test_subj}, 训练被试={train_subjs}")

        # 在训练被试上生成基线数据
        rng = np.random.RandomState(42)
        point_frames, timestamps = [], []
        for day in range(5):
            generate_day(point_frames, timestamps,
                         day_index=day, subjects=train_subjs,
                         replay=replay, seconds_per_day=args.day_seconds,
                         drift_level=0.0, rng=rng)

        S_cross, ts_cross = extract_S_sequence(encoder, point_frames,
                                                  torch.device(args.device), stride=4,
                                                  timestamps_ms=timestamps)

        if len(S_cross) < 200:
            print(f"    [WARN] 跨人训练数据不足 ({len(S_cross)} S_t), 跳过")
            continue

        # 生成测试被试的 fall 帧（所有冷启动级别共用）
        fall_frames_raw = replay.generate_sequence("fall", 10.0, subject_ids=[test_subj])
        pt_fall, ts_fall = [], []
        base_fall_ts = 1753891200000 + 200 * 86400 * 1000
        for i, pts in enumerate(fall_frames_raw):
            pt_fall.append(pts.astype(np.float32))
            ts_fall.append(int(base_fall_ts + 12 * 3600 * 1000 + i * 100))
        S_fall, ts_fall_aligned = extract_S_sequence(encoder, pt_fall,
                                                       torch.device(args.device), stride=4,
                                                       timestamps_ms=ts_fall)

        # 测试被试的独立测试正常帧（day 200，远离训练数据天数，不参与训练）
        pt_test_normal, ts_test_normal = [], []
        generate_day(pt_test_normal, ts_test_normal, day_index=200,
                     subjects=all_subjects, replay=replay,
                     seconds_per_day=args.day_seconds, drift_level=0.0,
                     rng=rng, subject_filter=test_subj)
        S_test_normal, ts_test_normal = extract_S_sequence(
            encoder, pt_test_normal, torch.device(args.device), stride=4,
            timestamps_ms=ts_test_normal)

        # 生成该被试的"个人数据池"（用于训练，day 100+）
        max_personal_days = max(cold_start_days)
        pt_personal, ts_personal = [], []
        for day in range(max_personal_days):
            generate_day(pt_personal, ts_personal,
                         day_index=100 + day, subjects=all_subjects,
                         replay=replay, seconds_per_day=args.day_seconds,
                         drift_level=0.0, rng=rng, subject_filter=test_subj)
        S_personal_pool, ts_personal_pool = extract_S_sequence(
            encoder, pt_personal, torch.device(args.device), stride=4,
            timestamps_ms=ts_personal)

        # 冷启动曲线
        cold_start_aucs = {}
        for cs_days in cold_start_days:
            if cs_days == 0:
                S_cs = S_cross
                ts_cs = ts_cross
            else:
                cutoff_ts = 1753891200000 + (100 + cs_days) * 86400 * 1000
                n_personal = int(np.searchsorted(ts_personal_pool, cutoff_ts))
                S_personal = S_personal_pool[:n_personal]
                ts_personal = ts_personal_pool[:n_personal]
                S_cs = np.vstack([S_cross, S_personal])
                ts_cs = np.concatenate([ts_cross, ts_personal])

            if len(S_cs) < 200:
                cold_start_aucs[f"d{cs_days}"] = 0.0
                continue

            lb_cs = LatentBaseline()
            lb_cs.fit(S_cs, ts_cs)

            if not lb_cs.is_ready:
                cold_start_aucs[f"d{cs_days}"] = 0.0
                continue

            try:
                from sklearn.metrics import roc_auc_score
                n_fall = len(S_fall)
                # 用独立测试正常帧（不参与训练）
                n_normal = min(len(S_test_normal), len(ts_test_normal))
                normal_scores = [lb_cs.score(S_test_normal[i], ts_test_normal[i]) for i in range(n_normal)]
                fall_scores = [lb_cs.score(S_fall[i], ts_fall_aligned[i]) for i in range(n_fall)]

                all_scores = np.array(normal_scores + fall_scores)
                all_labels = np.array([0] * len(normal_scores) + [1] * len(fall_scores))
                auc_val = roc_auc_score(all_labels, all_scores)
                cold_start_aucs[f"d{cs_days}"] = auc_val
            except Exception as e:
                cold_start_aucs[f"d{cs_days}"] = 0.0

        auc_d0 = cold_start_aucs.get("d0", 0.0)
        auc_d14 = cold_start_aucs.get("d14", 0.0)
        cs_str = " → ".join(f"d{d}={cold_start_aucs.get(f'd{d}', 0):.3f}" for d in cold_start_days)
        print(f"    冷启动: {cs_str}")

        results.append({
            "subject": int(test_subj),
            "same_auc": auc_d14,       # 14 天个人数据 = 同人上限
            "cross_auc": auc_d0,       # 0 天 = 纯跨人
            "cold_start": cold_start_aucs,
            "n_train_frames": len(S_cross),
        })
        ckpt.save_one(results[-1], test_subj)

    report = report_exp1_cross_person(results)
    write_report("exp1_cross_person.md", report)
    print(report)
    return results


# ═══════════════════════════════════════════════════════
# Experiment 2: 前置时间分析
# ═══════════════════════════════════════════════════════

def exp2_gait_degradation(args):
    """动作质量退化检测：walk 点云渐增噪声 → NLL 单调上升。"""
    from ai.drift.degraders import PointCloudDegrader

    print("=" * 60)
    print("Experiment 2: 动作质量退化 (Gait Degradation)")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    subject = args.subject if args.subject is not None else subjects[0]
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)

    noise_levels = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25]  # σ in meters
    ckpt = Checkpoint("2")
    results = ckpt.get_results()

    # 基线：14 天正常数据
    print("  构建基线...", end="", flush=True)
    point_frames, timestamps = [], []
    for day in range(14):
        generate_day(point_frames, timestamps,
                     day_index=day, subjects=subjects,
                     replay=replay, seconds_per_day=args.day_seconds,
                     drift_level=0.0, rng=np.random.RandomState(42),
                     subject_filter=subject)
    S_all, ts_all = extract_S_sequence(encoder, point_frames,
                                        torch.device(args.device), stride=4,
                                        timestamps_ms=timestamps)
    lb = LatentBaseline()
    lb.fit(S_all, ts_all)
    print(f" {len(S_all)} S_t, {len(lb._gmms)} contexts")

    if not lb.is_ready:
        print("  [ERROR] LatentBaseline 未就绪")
        return []

    # 正常帧 NLL 分布（用于对比）
    detector = DistributionDriftDetector(lb)
    raw_baseline = detector.score_frames(S_all, ts_all)
    baseline_nlls = []
    for ctx, info in raw_baseline["per_context"].items():
        # 需要逐帧 NLL，这里先收集 baseline stats
        pass

    # 每个噪声等级生成 walk-only 数据并评分
    print(f"\n  {'噪声σ':>8s}  {'mean NLL':>10s}  {'Δ baseline':>10s}  {'检出':>6s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*6}")

    # 统计 baseline 逐帧 NLL（用 baseline 自身评分）
    bl_scores = []
    day_S_bounds = compute_day_boundaries_from_ts(ts_all)
    for day in range(14):
        s, e = day_S_bounds[day], day_S_bounds[day + 1] if day + 1 < len(day_S_bounds) else len(S_all)
        if e > s:
            r = detector.score_day(S_all[s:e], ts_all[s:e])
            bl_scores.append(r.mean_nll)
    bl_mean_nll = np.mean(bl_scores)
    bl_std_nll = np.std(bl_scores)

    for noise_sigma in noise_levels:
        if ckpt.is_done(f"noise_{noise_sigma}"):
            print(f"  {noise_sigma:8.3f}  [已跳过]")
            continue

        # 生成 1 天数据（含噪声）。用新的 ReplayGenerator 确保
        # 和基线天从相同的 RNG 起点采样，唯一差异是点云噪声。
        replay_test = ReplayGenerator(DATASET_PATH)
        degrader = PointCloudDegrader(noise=noise_sigma) if noise_sigma > 0 else PointCloudDegrader(noise=0)
        pf_test, ts_test = [], []
        _generate_day_with_degrader(
            pf_test, ts_test, day_index=0, subjects=subjects,
            replay=replay_test, seconds_per_day=args.day_seconds,
            pattern=NORMAL_PATTERN, rng=np.random.RandomState(42),
            degrader=degrader, drift_level=0.0, subject_filter=subject)

        S_test, ts_test_arr = extract_S_sequence(
            encoder, pf_test, torch.device(args.device), stride=4,
            timestamps_ms=ts_test)

        if len(S_test) < 10:
            continue

        r_test = detector.score_day(S_test, ts_test_arr)
        delta = r_test.mean_nll - bl_mean_nll
        detected = "✅" if r_test.overall_score > 0.15 else "  —"
        print(f"  {noise_sigma:8.3f}  {r_test.mean_nll:10.2f}  {delta:+10.2f}  {detected:>6s}")

        entry = {
            "noise_sigma": noise_sigma,
            "mean_nll": r_test.mean_nll,
            "delta_nll": delta,
            "overall_score": r_test.overall_score,
            "baseline_mean_nll": bl_mean_nll,
            "baseline_std_nll": bl_std_nll,
        }
        results.append(entry)
        ckpt.save_one(entry, f"noise_{noise_sigma}")

    report = report_exp2_gait(results)
    write_report("exp2_gait_degradation.md", report)
    print(f"\n{report}")
    return results


# ═══════════════════════════════════════════════════════
# Experiment 3: 异常动作检测
# ═══════════════════════════════════════════════════════

def exp3_anomaly_action(args):
    """异常动作检测：插入 fall 帧 → S_t 进入 GMM 盲区 → NLL 尖峰。"""
    print("=" * 60)
    print("Experiment 3: 异常动作检测 (Anomaly Action)")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    subject = args.subject if args.subject is not None else subjects[0]
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)

    ckpt = Checkpoint("3")
    results = ckpt.get_results()

    # 基线：14 天正常数据
    print("  构建基线...", end="", flush=True)
    point_frames, timestamps = [], []
    for day in range(14):
        generate_day(point_frames, timestamps,
                     day_index=day, subjects=subjects,
                     replay=replay, seconds_per_day=args.day_seconds,
                     drift_level=0.0, rng=np.random.RandomState(42),
                     subject_filter=subject)
    S_all, ts_all = extract_S_sequence(encoder, point_frames,
                                        torch.device(args.device), stride=4,
                                        timestamps_ms=timestamps)
    lb = LatentBaseline()
    lb.fit(S_all, ts_all)
    print(f" {len(S_all)} S_t, {len(lb._gmms)} contexts")

    if not lb.is_ready:
        print("  [ERROR] LatentBaseline 未就绪")
        return []

    detector = DistributionDriftDetector(lb)

    # 正常帧 NLL 分布
    raw_bl = detector.score_frames(S_all, ts_all)
    normal_scores_all = []
    for ctx, info in raw_bl.get("per_context", {}).items():
        pass  # per-context stats are in info

    # 逐帧 NLL：重新跑一遍 score_frames 拿 all_nlls
    # score_frames 内部构建 all_nlls，但返回值里没暴露。这里用 score_day 的 mean_nll 近似。
    # 对正常天的逐帧 NLL，通过 score_day 获取每天的 mean NLL。
    day_S_bounds = compute_day_boundaries_from_ts(ts_all)
    normal_nlls = []
    for day in range(14):
        s, e = day_S_bounds[day], day_S_bounds[day + 1] if day + 1 < len(day_S_bounds) else len(S_all)
        if e > s:
            r = detector.score_day(S_all[s:e], ts_all[s:e])
            normal_nlls.append(r.mean_nll)
    normal_mean = np.mean(normal_nlls)
    normal_p95 = np.percentile(normal_nlls, 95)

    # 异常类型：fall 帧 + 陌生被试 walk 帧
    anomaly_configs = [
        ("fall", "跌倒"),
        ("walk_stranger", "陌生被试 walk"),
    ]

    print(f"\n  {'异常类型':>16s}  {'正常NLL':>10s}  {'异常NLL':>10s}  {'P95比值':>8s}  {'AUC':>6s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")

    for anom_key, anom_label in anomaly_configs:
        if ckpt.is_done(anom_key):
            print(f"  {anom_label:>16s}  [已跳过]")
            continue

        # 生成异常帧
        if anom_key == "fall":
            # 从 replay 获取 fall 帧
            anom_frames = replay.generate_sequence(
                "fall", 5.0, subject_ids=[subject])
        else:
            # 陌生被试 walk 帧
            stranger = (subject + 3) % 7
            anom_frames = replay.generate_sequence(
                "walk", 5.0, subject_ids=[stranger])

        if not anom_frames:
            continue

        anom_pts = [f.astype(np.float32) for f in anom_frames]
        base_ts = 1753891200000 + 200 * 86400 * 1000
        anom_ts = [int(base_ts + i * 100) for i in range(len(anom_pts))]
        S_anom, ts_anom = extract_S_sequence(
            encoder, anom_pts, torch.device(args.device), stride=4,
            timestamps_ms=anom_ts)

        if len(S_anom) < 5:
            continue

        r_anom = detector.score_day(S_anom, ts_anom)
        anom_mean = r_anom.mean_nll

        # AUC：正常天 NLL vs 异常天 NLL（单天级比较）
        labels = [0] * len(normal_nlls) + [1]
        scores = normal_nlls + [anom_mean]
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.0

        ratio = anom_mean / normal_p95 if normal_p95 > 0 else 0
        print(f"  {anom_label:>16s}  {normal_mean:10.2f}  {anom_mean:10.2f}  {ratio:8.2f}x  {auc:6.4f}")

        entry = {
            "anomaly_type": anom_label,
            "normal_nll_mean": normal_mean,
            "anomaly_nll_mean": anom_mean,
            "normal_nll_p95": normal_p95,
            "anomaly_nll_p95": anom_mean,
            "auc": auc,
        }
        results.append(entry)
        ckpt.save_one(entry, anom_key)

    report = report_exp3_anomaly(results)
    write_report("exp3_anomaly_action.md", report)
    print(f"\n{report}")
    return results

    print("=" * 60)
    print("Experiment 3: 退化模式库")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)
    ckpt = Checkpoint("3")
    results = ckpt.get_results()

    for pattern_name, pattern_fn in PATTERNS.items():
        label = PATTERN_LABELS.get(pattern_name, pattern_name)
        print(f"\n  模式: {label} ({pattern_name})")

        for seed_idx in range(args.seeds):
            seed = 42 + seed_idx
            if ckpt.is_done(pattern_name, seed):
                print(f"    seed={seed} [已跳过]")
                continue
            rng = np.random.RandomState(seed)

            point_frames, timestamps = [], []
            day_boundaries = [0]
            baseline_days, drift_days = 7, 7
            total_days = baseline_days + drift_days

            for day in range(total_days):
                dl = 0.0 if day < baseline_days else (day - baseline_days + 1) / drift_days

                if dl > 0 and pattern_name != "gait_instability":
                    # 用 pattern 函数修改 NORMAL_PATTERN
                    modified = pattern_fn(NORMAL_PATTERN, dl)
                    # 用手动方式生成（绕过 apply_drift）
                    _generate_day_with_pattern(point_frames, timestamps, day,
                                               subjects, replay, args.day_seconds,
                                               modified, rng,
                                               subject_filter=args.subject)
                elif dl > 0 and pattern_name == "gait_instability":
                    # 步态不稳：正常 pattern + 点云噪声
                    from ai.drift.degraders import PointCloudDegrader
                    degrader = PointCloudDegrader(noise=dl)
                    _generate_day_with_degrader(point_frames, timestamps, day,
                                                subjects, replay, args.day_seconds,
                                                NORMAL_PATTERN, rng, degrader,
                                                subject_filter=args.subject)
                else:
                    _generate_day_with_pattern(point_frames, timestamps, day,
                                               subjects, replay, args.day_seconds,
                                               NORMAL_PATTERN, rng,
                                               subject_filter=args.subject)

                day_boundaries.append(len(point_frames))

            if len(point_frames) < WINDOW * 2:
                continue

            S_all, ts_all = extract_S_sequence(encoder, point_frames,
                                                   torch.device(args.device), stride=4,
                                                   timestamps_ms=timestamps)
            day_S_bounds = compute_day_boundaries_from_ts(ts_all)

            if len(S_all) < 300:
                continue

            # Fit baseline + score drift days
            bl_end = day_S_bounds[baseline_days]
            lb = LatentBaseline()
            lb.fit(S_all[:bl_end], ts_all[:bl_end])
            if not lb.is_ready:
                continue

            detector = DistributionDriftDetector(lb)

            # Baseline drift scores
            bl_scores = []
            for d in range(baseline_days):
                s, e = day_S_bounds[d], day_S_bounds[d + 1]
                if e <= s:
                    continue
                bl_scores.append(detector.score_day(S_all[s:e], ts_all[s:e]).overall_score)
            threshold = np.percentile(bl_scores, 95) if bl_scores else 0.3
            threshold = max(threshold, 0.15)

            # Drift days
            drift_scores = []
            alerted = False
            first_alert = None
            for d in range(baseline_days, total_days):
                s = day_S_bounds[d] if d < len(day_S_bounds) else day_S_bounds[-1]
                e = day_S_bounds[d + 1] if d + 1 < len(day_S_bounds) else len(S_all)
                if e <= s:
                    continue
                score = detector.score_day(S_all[s:e], ts_all[s:e]).overall_score
                drift_scores.append(score)
                if score > threshold:
                    alerted = True
                    if first_alert is None:
                        first_alert = d + 1

            # AUC: baseline days vs drift days (by drift_level)
            try:
                from sklearn.metrics import roc_auc_score
                labels = np.array([0]*len(bl_scores) + [1]*len(drift_scores))
                scores_arr = np.array(bl_scores + drift_scores)
                drift_auc = roc_auc_score(labels, scores_arr)
            except Exception:
                drift_auc = 0.0

            lead_time = total_days - first_alert if first_alert else None
            entry = {
                "pattern": pattern_name,
                "seed": seed,
                "drift_auc": drift_auc,
                "lead_time_days": lead_time,
                "detection_rate": 1.0 if alerted else 0.0,
                "threshold": threshold,
            }
            results.append(entry)
            ckpt.save_one(entry, pattern_name, seed)
            print(f"    seed={seed} AUC={drift_auc:.3f} lead={lead_time}")

    report = report_exp3_patterns(results)
    write_report("exp3_patterns.md", report)
    print(report)
    return results


# ═══════════════════════════════════════════════════════
# Experiment 4: Context 时段敏感性
# ═══════════════════════════════════════════════════════

def exp4_context_sensitivity(args):
    """Context 时段敏感性：同 S_t 错配 context GMM → NLL 升高。"""
    print("=" * 60)
    print("Experiment 4: Context 时段敏感性 (Context Sensitivity)")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    subject = args.subject if args.subject is not None else subjects[0]
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)

    ckpt = Checkpoint("4")
    results = ckpt.get_results()

    # 基线：14 天正常数据
    print("  构建基线...", end="", flush=True)
    point_frames, timestamps = [], []
    for day in range(14):
        generate_day(point_frames, timestamps,
                     day_index=day, subjects=subjects,
                     replay=replay, seconds_per_day=args.day_seconds,
                     drift_level=0.0, rng=np.random.RandomState(42),
                     subject_filter=subject)
    S_all, ts_all = extract_S_sequence(encoder, point_frames,
                                        torch.device(args.device), stride=4,
                                        timestamps_ms=timestamps)
    lb = LatentBaseline()
    lb.fit(S_all, ts_all)
    print(f" {len(S_all)} S_t")

    if not lb.is_ready:
        print("  [ERROR] LatentBaseline 未就绪")
        return []

    detector = DistributionDriftDetector(lb)

    # 生成 1 天测试数据
    pf_test, ts_test = [], []
    generate_day(pf_test, ts_test, day_index=100, subjects=subjects,
                 replay=replay, seconds_per_day=args.day_seconds,
                 drift_level=0.0, rng=np.random.RandomState(99),
                 subject_filter=subject)
    S_test, ts_test_arr = extract_S_sequence(
        encoder, pf_test, torch.device(args.device), stride=4,
        timestamps_ms=ts_test)

    # 正确 context 评分
    r_correct = detector.score_day(S_test, ts_test_arr)

    # 错位 context：偏移 timestamp ±6h, ±12h
    shifts = [
        ("0h (正确)", 0),
        ("+6h", 6 * 3600 * 1000),
        ("-6h", -6 * 3600 * 1000),
        ("+12h", 12 * 3600 * 1000),
    ]

    print(f"\n  {'偏移':>12s}  {'mean NLL':>10s}  {'drift score':>12s}  {'Δ NLL':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*10}")

    for shift_label, shift_ms in shifts:
        if ckpt.is_done(shift_label):
            print(f"  {shift_label:>12s}  [已跳过]")
            continue

        if shift_ms == 0:
            r = r_correct
        else:
            ts_shifted = ts_test_arr.copy() + shift_ms
            r = detector.score_day(S_test, ts_shifted)

        delta = r.mean_nll - r_correct.mean_nll
        delta_pct = (delta / r_correct.mean_nll * 100) if r_correct.mean_nll > 0 else 0
        print(f"  {shift_label:>12s}  {r.mean_nll:10.2f}  {r.overall_score:12.4f}  {delta:+10.2f} ({delta_pct:+.1f}%)")

        entry = {
            "shift": shift_label,
            "correct_nll": r_correct.mean_nll if shift_ms != 0 else r.mean_nll,
            "shifted_nll": r.mean_nll,
            "delta_pct": delta_pct,
        }
        results.append(entry)
        ckpt.save_one(entry, shift_label)

    report = report_exp4_context(results)
    write_report("exp4_context_sensitivity.md", report)
    print(f"\n{report}")
    return results


# ═══════════════════════════════════════════════════════
# Experiment 5: 消融实验
# ═══════════════════════════════════════════════════════

def exp5_ablation(args):
    """消融实验：关 PCA/K/context，测步态劣化 AUC 变化。"""
    from ai.drift.ablation import ABLATIONS
    from ai.drift.degraders import PointCloudDegrader

    print("=" * 60)
    print("Experiment 5: 消融实验 (Ablation)")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    subject = args.subject if args.subject is not None else subjects[0]
    encoder = load_encoder(torch.device(args.device))
    replay = ReplayGenerator(DATASET_PATH)
    ckpt = Checkpoint("5")
    results = ckpt.get_results()

    # 信号：步态劣化 σ=3cm（Exp2 中已验证可检出）
    GAIT_SIGMA = 0.03
    n_baseline_days = 7
    n_test_days = 3

    for abl in ABLATIONS:
        if ckpt.is_done(abl.name):
            print(f"  {abl.label:>25s}  [已跳过]")
            continue

        all_aucs = []
        for seed_idx in range(args.seeds):
            seed = 42 + seed_idx
            rng = np.random.RandomState(seed)

            # 独立 replay 实例，避免帧采样偏移干扰
            replay_bl = ReplayGenerator(DATASET_PATH)
            replay_test = ReplayGenerator(DATASET_PATH)
            pf_bl, ts_bl = [], []
            pf_test, ts_test = [], []
            degrader = PointCloudDegrader(noise=GAIT_SIGMA)

            for day in range(n_baseline_days):
                _generate_day_with_pattern(pf_bl, ts_bl, day,
                                           subjects, replay_bl, args.day_seconds,
                                           NORMAL_PATTERN, rng, drift_level=0.0,
                                           subject_filter=subject)
            for day in range(n_test_days):
                _generate_day_with_degrader(pf_test, ts_test, day,
                                            subjects, replay_test, args.day_seconds,
                                            NORMAL_PATTERN, rng, degrader,
                                            drift_level=0.0,
                                            subject_filter=subject)

            if len(pf_bl) < WINDOW * 2 or len(pf_test) < WINDOW * 2:
                continue

            S_bl, ts_bl_arr = extract_S_sequence(encoder, pf_bl,
                                                  torch.device(args.device), stride=4,
                                                  timestamps_ms=ts_bl)
            S_test, ts_test_arr = extract_S_sequence(encoder, pf_test,
                                                      torch.device(args.device), stride=4,
                                                      timestamps_ms=ts_test)
            if len(S_bl) < 200 or len(S_test) < 50:
                continue

            pca_dim = abl.pca_dim if abl.pca_dim else 12
            if abl.name == "no_pca":
                pca_dim = min(256, S_bl.shape[1])

            lb = LatentBaseline(
                pca_dim=pca_dim,
                n_components=abl.n_components if abl.n_components else 3,
                use_contexts=abl.use_contexts,
            )
            lb.fit(S_bl, ts_bl_arr)
            if not lb.is_ready:
                continue

            detector = DistributionDriftDetector(lb, use_nll=abl.use_nll)

            day_S_bounds = compute_day_boundaries_from_ts(ts_bl_arr)
            bl_scores = []
            for d in range(n_baseline_days):
                s, e = day_S_bounds[d], day_S_bounds[d + 1] if d + 1 < len(day_S_bounds) else len(S_bl)
                if e > s:
                    bl_scores.append(detector.score_day(S_bl[s:e], ts_bl_arr[s:e]).overall_score)

            # 测试数据较短，作为整体评分
            r_test = detector.score_day(S_test, ts_test_arr)
            drift_scores = [r_test.overall_score]

            try:
                from sklearn.metrics import roc_auc_score
                labels = [0] * len(bl_scores) + [1] * len(drift_scores)
                auc = roc_auc_score(labels, bl_scores + drift_scores)
            except Exception:
                auc = 0.0
            all_aucs.append(auc)

        if all_aucs:
            avg_auc = np.mean(all_aucs)
            entry = {"name": abl.name, "label": abl.label, "drift_auc": avg_auc}
            results.append(entry)
            ckpt.save_one(entry, abl.name)
            print(f"  {abl.label:>25s}  AUC={avg_auc:.4f}")
        else:
            print(f"  {abl.label:>25s}  [NO DATA]")

    # 计算 Δ AUC
    bl_auc = next((r["drift_auc"] for r in results if r["name"] == "baseline"), None)
    for r in results:
        r["delta_auc"] = r.get("drift_auc", 0) - bl_auc if bl_auc else 0
        r["lead_time_days"] = None  # for report compat

    report = report_exp5_ablation(results)
    write_report("exp5_ablation.md", report)
    print(f"\n{report}")
    return results


# ═══════════════════════════════════════════════════════
# Experiment 6: 传感器鲁棒性
# ═══════════════════════════════════════════════════════

def exp6_robustness(args):
    """传感器鲁棒性：点云退化叠加步态劣化 → AUC 衰减曲线。"""
    from ai.drift.degraders import DEGRADATION_LEVELS, PointCloudDegrader

    print("=" * 60)
    print("Experiment 6: 传感器鲁棒性 (Sensor Robustness)")
    print("=" * 60)

    subjects = args.subjects or get_subjects()
    subject = args.subject if args.subject is not None else subjects[0]
    encoder = load_encoder(torch.device(args.device))
    ckpt = Checkpoint("6")
    results = ckpt.get_results()

    GAIT_SIGMA = 0.08  # 步态劣化信号（Exp2 中已验证在阈值附近）
    n_baseline_days = 7
    n_test_days = 2

    for deg_type, levels in DEGRADATION_LEVELS.items():
        print(f"\n  退化类型: {deg_type}")
        print(f"    {'等级':>20s}  {'neg NLL':>10s}  {'pos NLL':>10s}  {'Δ':>8s}  {'AUC':>8s}")
        print(f"    {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

        for level_name, sensor_degrader in levels:
            if ckpt.is_done(deg_type, level_name):
                print(f"    {level_name:>20s}  [已跳过]")
                continue

            neg_scores = []  # 传感器退化 only（不应触发）
            pos_scores = []  # 传感器退化 + 步态劣化（应触发）

            for seed_idx in range(args.seeds):
                seed = 42 + seed_idx
                rng = np.random.RandomState(seed)

                # 基线：正常数据
                replay_bl = ReplayGenerator(DATASET_PATH)
                pf_bl, ts_bl = [], []
                for day in range(n_baseline_days):
                    _generate_day_with_pattern(pf_bl, ts_bl, day,
                                               subjects, replay_bl, args.day_seconds,
                                               NORMAL_PATTERN, rng, drift_level=0.0,
                                               subject_filter=subject)

                S_bl, ts_bl_arr = extract_S_sequence(encoder, pf_bl,
                                                      torch.device(args.device), stride=4,
                                                      timestamps_ms=ts_bl)
                lb = LatentBaseline()
                lb.fit(S_bl, ts_bl_arr)
                if not lb.is_ready:
                    continue
                detector = DistributionDriftDetector(lb)

                # 生成原始点云（负正共用同一 replay，避免帧采样差异）
                replay_shared = ReplayGenerator(DATASET_PATH)
                pf_raw, ts_raw = [], []
                for day in range(n_test_days):
                    _generate_day_with_pattern(pf_raw, ts_raw, day,
                                               subjects, replay_shared, args.day_seconds,
                                               NORMAL_PATTERN, rng, drift_level=0.0,
                                               subject_filter=subject)

                # 负样本：传感器退化 only
                gait_degrader = PointCloudDegrader(noise=GAIT_SIGMA)
                pf_neg = []
                for pts in pf_raw:
                    degraded = sensor_degrader.apply(pts.astype(np.float32))
                    if degraded is not None:
                        pf_neg.append(degraded)
                S_neg, ts_neg_arr = extract_S_sequence(
                    encoder, pf_neg, torch.device(args.device), stride=4,
                    timestamps_ms=ts_raw)
                r_neg = detector.score_day(S_neg, ts_neg_arr)
                neg_scores.append(r_neg.overall_score)

                # 正样本：传感器退化 + 步态劣化
                pf_pos = []
                for pts in pf_raw:
                    degraded = sensor_degrader.apply(pts.astype(np.float32))
                    if degraded is not None:
                        gait_noisy = gait_degrader.apply(degraded)
                        if gait_noisy is not None:
                            pf_pos.append(gait_noisy)
                        else:
                            pf_pos.append(degraded)
                S_pos, ts_pos_arr = extract_S_sequence(
                    encoder, pf_pos, torch.device(args.device), stride=4,
                    timestamps_ms=ts_raw)
                r_pos = detector.score_day(S_pos, ts_pos_arr)
                pos_scores.append(r_pos.overall_score)

            # AUC：负 vs 正，衡量传感器退化下步态劣化的可检测性
            if neg_scores and pos_scores:
                try:
                    from sklearn.metrics import roc_auc_score
                    labels = [0]*len(neg_scores) + [1]*len(pos_scores)
                    auc = roc_auc_score(labels, neg_scores + pos_scores)
                except Exception:
                    auc = 0.5
            else:
                auc = 0.5

            neg_mean = np.mean(neg_scores) if neg_scores else 0
            pos_mean = np.mean(pos_scores) if pos_scores else 0
            delta = pos_mean - neg_mean
            print(f"    {level_name:>20s}  {neg_mean:10.4f}  {pos_mean:10.4f}  {delta:+8.4f}  {auc:8.4f}")

            entry = {"degradation": deg_type, "level": level_name,
                     "neg_nll": neg_mean, "pos_nll": pos_mean, "auc": auc}
            results.append(entry)
            ckpt.save_one(entry, deg_type, level_name)

    report = report_exp6_robustness(results)
    write_report("exp6_robustness.md", report)
    print(f"\n{report}")
    return results




# ── 辅助生成函数 ──────────────────────────────────────

def _generate_day_with_pattern(point_frames_out, timestamps_out,
                                day_index, subjects, replay, seconds_per_day,
                                pattern, rng, drift_level=None,
                                subject_filter=None):
    """用自定义 pattern（非 NORMAL_PATTERN）生成一天数据。"""
    base_day_ts = 1753891200000 + day_index * 86400 * 1000
    time_scale = seconds_per_day / 24.0
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

        subject_id = subject_filter if subject_filter is not None else rng.choice(subjects)
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


def _generate_day_with_degrader(point_frames_out, timestamps_out,
                                 day_index, subjects, replay, seconds_per_day,
                                 pattern, rng, degrader, drift_level=None,
                                 subject_filter=None):
    """用点云退化器生成一天数据。"""
    base_day_ts = 1753891200000 + day_index * 86400 * 1000
    time_scale = seconds_per_day / 24.0
    sim_time = 0.0
    pattern_idx = 0

    # 应用 drift
    if drift_level and drift_level > 0:
        pattern = apply_drift(pattern, drift_level)

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

        subject_id = subject_filter if subject_filter is not None else rng.choice(subjects)
        if action == "lie":
            frames = replay.generate_sequence("squat", dur_s, subject_ids=[subject_id])
        else:
            frames = replay.generate_sequence(action, dur_s, subject_ids=[subject_id])

        if not frames:
            pattern_idx += 1
            continue

        n_frames = len(frames)
        for i, pts in enumerate(frames):
            degraded = degrader.apply(pts.astype(np.float32))
            if degraded is not None:
                point_frames_out.append(degraded)
                frac = i / max(n_frames, 1)
                frame_hour = start_h + frac * dur_h
                frame_hour = frame_hour % 24
                timestamps_out.append(int(base_day_ts + frame_hour * 3600 * 1000))

        sim_time += dur_s
        pattern_idx += 1


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

EXPERIMENTS = {
    "1": ("exp1_cross_person", exp1_cross_person),
    "2": ("exp2_gait_degradation", exp2_gait_degradation),
    "3": ("exp3_anomaly_action", exp3_anomaly_action),
    "4": ("exp4_context_sensitivity", exp4_context_sensitivity),
    "5": ("exp5_ablation", exp5_ablation),
    "6": ("exp6_robustness", exp6_robustness),
}


def main():
    parser = argparse.ArgumentParser(description="事前健康管理 — 仿真实验")
    parser.add_argument("--exp", default="2",
                        help="实验编号 (1-6) 或 'all'")
    parser.add_argument("--day-seconds", type=float, default=60.0,
                        help="每天仿真秒数")
    parser.add_argument("--seeds", type=int, default=3,
                        help="随机种子数")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="被试列表")
    parser.add_argument("--subject", type=int, default=None,
                        help="固定监控被试 ID（Exp2-6），避免 GMM 过拟合到 seed 指纹")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（减半天数）")

    args = parser.parse_args()

    to_run = []
    if args.exp == "all":
        to_run = list(EXPERIMENTS.keys())
    else:
        for e in args.exp.split(","):
            e = e.strip()
            if e in EXPERIMENTS:
                to_run.append(e)
            else:
                print(f"未知实验: {e}. 可用: {list(EXPERIMENTS.keys())}")

    for exp_id in to_run:
        name, fn = EXPERIMENTS[exp_id]
        print(f"\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")
        t0 = time.time()
        try:
            fn(args)
        except NotImplementedError:
            print(f"  [SKIP] {name} 尚未实现")
        print(f"  耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
