#!/usr/bin/env python3
"""
LatentBaseline 拟合 + 异常检测评估。

1. 用仿真器生成正常生活场景 → Encoder 提取 S_t
2. 拟合 LatentBaseline
3. 在异常场景上评估：正常帧 vs 异常帧的 latent density score

用法:
  python -m ai.baseline.fit_and_eval --hours 2
"""

import argparse
import os
import sys
import time
import numpy as np
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.generators import ReplayGenerator
from simulator.calibrate import ACTION_NAMES
from simulator.scenario import ScenarioEngine
from ai.encoder.encoder_model import EncoderModel
from ai.baseline.latent_baseline import LatentBaseline

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "public", "processed", "3dpchm_frames.npz",
)
ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "encoder_best.pt",
)
SCENARIOS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "simulator", "scenarios",
)

FPS = 10
WINDOW = 32
MAX_POINTS = 64

# 正常日常活动（不含 fall）
NORMAL_ACTIONS = ["stand", "sit", "walk", "squat", "lean_left", "lean_right"]

# 活动模板：模拟一天的自然节奏
DAILY_PATTERN = [
    # 凌晨 — 睡觉
    ("lie", 0, 6),
    # 早晨 — 起床活动
    ("stand", 6, 6.5),
    ("walk", 6.5, 6.7),
    ("sit", 6.7, 7.2),
    ("stand", 7.2, 7.3),
    ("walk", 7.3, 7.5),
    # 上午 — 居家活动
    ("sit", 7.5, 9),
    ("stand", 9, 9.1),
    ("walk", 9.1, 9.3),
    ("squat", 9.3, 9.35),
    ("stand", 9.35, 9.5),
    ("sit", 9.5, 11.5),
    # 中午
    ("stand", 11.5, 11.6),
    ("walk", 11.6, 11.8),
    ("sit", 11.8, 13),
    # 下午
    ("stand", 13, 13.1),
    ("walk", 13.1, 13.3),
    ("sit", 13.3, 15),
    ("stand", 15, 15.1),
    ("walk", 15.1, 15.3),
    ("lean_left", 15.3, 15.33),
    ("lean_right", 15.33, 15.36),
    ("sit", 15.36, 17.5),
    # 傍晚
    ("stand", 17.5, 17.6),
    ("walk", 17.6, 17.8),
    ("sit", 17.8, 18.5),
    ("stand", 18.5, 18.6),
    ("walk", 18.6, 18.7),
    ("sit", 18.7, 20),
    # 晚上
    ("stand", 20, 20.1),
    ("walk", 20.1, 20.2),
    ("sit", 20.2, 22),
    ("stand", 22, 22.1),
    ("walk", 22.1, 22.2),
    ("lie", 22.2, 24),
]


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


def generate_daily_life(hours: float, subjects: list[int],
                        replay: ReplayGenerator) -> tuple[list[np.ndarray], list[int]]:
    """生成一天的正常生活：返回 (点云帧列表, 时间戳列表)。"""
    total_s = hours * 3600
    rng = np.random.RandomState(42)
    point_frames: list[np.ndarray] = []
    timestamps: list[int] = []

    # 将小时模板转为实际活动序列
    sim_time = 0.0
    pattern_idx = 0

    while sim_time < total_s:
        # 循环使用 DAILY_PATTERN
        if pattern_idx >= len(DAILY_PATTERN):
            pattern_idx = 0

        action, start_h, end_h = DAILY_PATTERN[pattern_idx]
        dur_h = end_h - start_h
        # 压缩时间：1 模拟小时 = 30 实际秒
        dur_s = dur_h * 30
        dur_s = min(dur_s, total_s - sim_time)
        if dur_s < 0.5:
            pattern_idx += 1
            continue

        subject_id = rng.choice(subjects)

        # lie 用 squat 的 GMM 参数近似（3DPCHM 无 lying）
        if action == "lie":
            frames = replay.generate_sequence("squat", dur_s, subject_ids=[subject_id])
        else:
            frames = replay.generate_sequence(action, dur_s, subject_ids=[subject_id])

        if not frames:
            pattern_idx += 1
            continue

        n_frames = len(frames)
        for i, pts in enumerate(frames):
            point_frames.append(pts.astype(np.float32))
            # 时间戳映射到模式指定的时段：用 start_h 作为当天的小时
            frame_hour = start_h + (i / n_frames) * dur_h
            frame_hour = frame_hour % 24
            # 基础时间戳：2026-07-31 00:00 UTC+8
            base_day_ts = 1753891200000  # 2026-07-31 00:00 Beijing
            timestamps.append(int(base_day_ts + frame_hour * 3600 * 1000))

        sim_time += dur_s
        pattern_idx += 1

    return point_frames, timestamps


def extract_S_sequence_fast(encoder: EncoderModel, point_frames: list[np.ndarray],
                            device: torch.device, stride: int = 8) -> np.ndarray:
    """
    高效逐帧提取 S_t。使用 backbone+buffer+TCN，比 encoder.forward() 快 100x。

    1. 每帧 → backbone → 1024-d feature
    2. Buffer 32 backbone features
    3. TCN(buffer) → 256-d S_t（取最后一帧）

    stride: S_t 采样间隔（默认每 8 帧取一个 S_t）
    返回: (M, 256) where M ≈ len(point_frames) / stride
    """
    N = len(point_frames)
    if N < WINDOW:
        return np.array([])

    rng = np.random.RandomState(42)
    S_list = []
    bb_buffer = []  # backbone feature buffer

    for i in range(N):
        pts = point_frames[i]
        # Pad to 64
        pts = pad_to_64(pts, rng)
        pts_t = torch.from_numpy(pts.astype(np.float32)).unsqueeze(0).to(device)  # (1, 64, 5)
        xyz = pts_t[:, :, :3]
        feats = pts_t[:, :, 3:]

        with torch.no_grad():
            bb = encoder.backbone(xyz, feats)  # (1, 1024)

        bb_buffer.append(bb.squeeze(0))  # (1024,)
        if len(bb_buffer) > WINDOW:
            bb_buffer = bb_buffer[-WINDOW:]

        if len(bb_buffer) < WINDOW:
            continue

        # 每 stride 帧提取一个 S_t
        if i % stride != 0:
            continue

        seq = torch.stack(list(bb_buffer)).unsqueeze(0)  # (1, 32, 1024)
        with torch.no_grad():
            S_seq = encoder.tcn(seq)  # (1, 32, 256)
        S_list.append(S_seq[0, -1].cpu().numpy())  # (256,)

    return np.array(S_list, dtype=np.float32)


def evaluate_on_scenario(lb: LatentBaseline, yaml_path: str,
                         dataset_path: str, device: torch.device) -> dict:
    """在单个异常场景上评估 LatentBaseline。使用高效 backbone+buffer 方式。"""
    from simulator.eval import load_world_model

    encoder, _, ok = load_world_model()
    if not ok:
        return {"error": "model_unavailable"}

    engine = ScenarioEngine(yaml_path, speed=20, dataset_path=dataset_path)
    frames = engine.run()

    scores = []
    gt_is_anomaly = []
    rng = np.random.RandomState(42)
    bb_buffer = []  # backbone feature buffer

    for fg in frames:
        pts = fg.raw_points if fg.raw_points is not None else fg.points
        if pts is None or len(pts) < 4:
            # 用零向量占位
            dummy = np.zeros((MAX_POINTS, 5), dtype=np.float32)
            pts_t = torch.from_numpy(dummy).unsqueeze(0).to(device)
            xyz, feats = pts_t[:, :, :3], pts_t[:, :, 3:]
            with torch.no_grad():
                bb = encoder.backbone(xyz, feats)
        else:
            pts = pad_to_64(pts, rng)
            pts_t = torch.from_numpy(pts.astype(np.float32)).unsqueeze(0).to(device)
            xyz, feats = pts_t[:, :, :3], pts_t[:, :, 3:]
            with torch.no_grad():
                bb = encoder.backbone(xyz, feats)

        bb_buffer.append(bb.squeeze(0))
        if len(bb_buffer) > WINDOW:
            bb_buffer = bb_buffer[-WINDOW:]

        if len(bb_buffer) < WINDOW:
            continue

        # TCN → S_t
        seq = torch.stack(list(bb_buffer)).unsqueeze(0)  # (1, WINDOW, 1024)
        with torch.no_grad():
            S_seq = encoder.tcn(seq)  # (1, WINDOW, 256)

        S_t = S_seq[0, -1].cpu().numpy()  # (256,)
        score = lb.score(S_t, fg.ts)
        scores.append(score)
        gt_is_anomaly.append(fg.gt is not None and fg.gt.anomaly_type is not None)

    scores = np.array(scores)
    gt_is_anomaly = np.array(gt_is_anomaly)

    normal_scores = scores[~gt_is_anomaly] if (~gt_is_anomaly).any() else np.array([])
    anomaly_scores = scores[gt_is_anomaly] if gt_is_anomaly.any() else np.array([])

    from sklearn.metrics import roc_auc_score as _roc_auc
    auc = _roc_auc(gt_is_anomaly.astype(int), scores) if gt_is_anomaly.any() and (~gt_is_anomaly).any() else 0.0

    return {
        "scenario": engine._scenario_name,
        "n_frames": len(scores),
        "n_normal": int((~gt_is_anomaly).sum()),
        "n_anomaly": int(gt_is_anomaly.sum()),
        "normal_score_mean": float(normal_scores.mean()) if len(normal_scores) > 0 else 0,
        "anomaly_score_mean": float(anomaly_scores.mean()) if len(anomaly_scores) > 0 else 0,
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=2.0,
                        help="正常生活仿真时长（小时）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--output", default=None,
                        help="LatentBaseline 保存路径")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 加载 Encoder ──────────────────────────────────
    print(f"加载 Encoder: {ENCODER_PATH}")
    encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
    ckpt = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()

    # ── 加载 ReplayGenerator ─────────────────────────
    replay = ReplayGenerator(DATASET_PATH)

    # ── 1. 生成正常生活数据 ───────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 1: 生成 {args.hours}h 正常生活仿真数据")
    print(f"{'='*60}")
    t0 = time.time()

    point_frames, timestamps = generate_daily_life(args.hours, args.subjects, replay)
    print(f"  点云帧: {len(point_frames):,}  ({time.time() - t0:.0f}s)")

    # ── 2. 提取 S_t ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2: Encoder 提取 S_t")
    print(f"{'='*60}")
    t1 = time.time()

    stride = 2
    S_all = extract_S_sequence_fast(encoder, point_frames, device, stride=stride)
    # 时间戳对齐: S_all[i] 对应 point_frames[(i*stride) + WINDOW - 1]
    ts_base = np.array(timestamps[WINDOW - 1:], dtype=np.int64)
    ts_all = ts_base[::stride]  # stride 对齐
    ts_all = ts_all[:len(S_all)]

    print(f"  S_t: {S_all.shape}  ({time.time() - t1:.0f}s)")

    if len(S_all) < 500:
        print("ERROR: S_t 太少，请增加 --hours")
        sys.exit(1)

    # ── 3. 拟合 LatentBaseline ────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 3: 拟合 LatentBaseline (PCA→32d, GMM×contexts)")
    print(f"{'='*60}")
    t2 = time.time()

    lb = LatentBaseline()
    lb.fit(S_all, ts_all)

    n_contexts = len(lb._gmms)
    print(f"  Contexts: {n_contexts}")
    for ctx in sorted(lb._gmms.keys()):
        n_samples = len(lb._scores.get(ctx, []))
        print(f"    {ctx}: {n_samples:,} samples")

    if not lb.is_ready:
        print("ERROR: LatentBaseline 未就绪（数据不足）")
        sys.exit(1)

    print(f"  Ready: ✅  ({time.time() - t2:.0f}s)")

    # ── 4. 保存 ───────────────────────────────────────
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "checkpoints", "latent_baseline.pkl",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lb.save(output_path)
    print(f"\n  保存到: {output_path}")

    # ── 5. 在异常场景上评估 ───────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: 异常场景评估")
    print(f"{'='*60}")

    scenario_files = sorted(
        [f for f in os.listdir(SCENARIOS_DIR) if f.endswith(".yaml")]
    )

    for sf in scenario_files:
        yaml_path = os.path.join(SCENARIOS_DIR, sf)
        result = evaluate_on_scenario(lb, yaml_path, DATASET_PATH, device)
        if "error" in result:
            print(f"  {sf}: {result['error']}")
            continue
        print(f"\n  [{sf}]")
        print(f"    帧数: {result['n_frames']} (正常={result['n_normal']}, 异常={result['n_anomaly']})")
        print(f"    正常帧平均分: {result['normal_score_mean']:.4f}")
        print(f"    异常帧平均分: {result['anomaly_score_mean']:.4f}")
        ratio = result['anomaly_score_mean'] / (result['normal_score_mean'] + 1e-10)
        print(f"    倍率: {ratio:.2f}x")
        print(f"    AUC:  {result['auc']:.4f}  {'✅' if result['auc'] > 0.7 else '⚠️' if result['auc'] > 0.55 else '❌'}")


if __name__ == "__main__":
    main()
