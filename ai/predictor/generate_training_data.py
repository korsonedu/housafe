#!/usr/bin/env python3
"""
生成 Predictor 训练数据 — 连续生活场景中的 S_t 序列。

从 3DPCHM replay 帧拼接跨动作连续序列 → Encoder 提取 S_t → 存 .npz。

核心改变：训练数据包含动作转移（stand→walk→sit），
而不是孤立动作片段。让 Predictor 学到真正的时序动力学。

用法:
  python -m ai.predictor.generate_training_data
  python -m ai.predictor.generate_training_data --hours 4 --output ai/data/training/predictor_continuous.npz
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
from ai.encoder.encoder_model import EncoderModel

# ── 配置 ──────────────────────────────────────────────

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "public", "processed", "3dpchm_frames.npz",
)
ENCODER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "encoder_best.pt",
)

FPS = 10
WINDOW = 32          # Encoder 输入窗口
PRED_WINDOW = 31     # Predictor 输入帧数（用前 31 帧预测第 32 帧）
MAX_POINTS = 64       # 每帧采样点数

# 日常生活动作（排除 fall 和无关动作）
DAILY_ACTIONS = ["stand", "sit", "walk", "squat", "lean_left", "lean_right"]

# 自然活动序列模板：持续秒数范围
ACTIVITY_PATTERNS = [
    # (action, min_s, max_s) — 模拟自然生活节奏
    ("stand", 3, 8),     # 短暂站立
    ("walk", 5, 15),     # 走动
    ("sit", 10, 60),     # 坐着（看电视/吃饭）
    ("stand", 2, 5),
    ("walk", 3, 10),
    ("squat", 2, 5),     # 蹲下（捡东西）
    ("stand", 1, 3),
    ("walk", 8, 20),     # 长距离走动
    ("sit", 30, 120),    # 长时间坐着
    ("stand", 2, 5),
    ("walk", 5, 10),
    ("lean_left", 1, 3),  # 微动作
    ("stand", 1, 3),
    ("lean_right", 1, 3),
    ("walk", 3, 8),
    ("sit", 5, 20),
    ("stand", 3, 8),
    ("walk", 5, 15),
    ("squat", 2, 5),
    ("stand", 2, 5),
]


def generate_activity_sequence(total_hours: float, rng: np.random.RandomState) -> list[tuple[str, float]]:
    """生成自然的日常活动序列"""
    total_s = total_hours * 3600
    seq = []
    elapsed = 0.0
    while elapsed < total_s:
        action, min_s, max_s = ACTIVITY_PATTERNS[
            rng.randint(0, len(ACTIVITY_PATTERNS))
        ]
        dur = rng.uniform(min_s, max_s)
        dur = min(dur, total_s - elapsed)
        if dur < 0.5:
            break
        seq.append((action, dur))
        elapsed += dur
    return seq


def pad_to_64(pts: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """将点云 pad 到 64 点"""
    n = len(pts)
    if n == 0:
        return np.zeros((MAX_POINTS, 5), dtype=np.float32)
    if n >= MAX_POINTS:
        return pts[rng.choice(n, MAX_POINTS, replace=False)]
    extra = MAX_POINTS - n
    idx = rng.randint(0, n, size=extra)
    jitter = rng.randn(extra, 5).astype(np.float32) * 0.01
    return np.vstack([pts, pts[idx] + jitter])


def extract_training_samples(
    encoder: EncoderModel,
    point_frames: list[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    stride: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    批量提取 Predictor 训练样本。

    每个样本来自一次独立的 encoder.forward(32帧):
      - 输入: S[0:31] (31 frame latent states from same encoder call)
      - 目标: S[31]   (the 32nd latent state from same encoder call)

    这保证训练与推理的一致性: Predictor 始终接收同一 encoder 调用产出的 S 序列。
    """
    N = len(point_frames)
    if N < WINDOW:
        return None

    rng = np.random.RandomState(42)
    X_chunks, y_chunks = [], []

    # 创建窗口起始索引列表（按 stride 跳过）
    window_starts = list(range(0, N - WINDOW + 1, stride))
    total_windows = len(window_starts)

    for batch_start in range(0, total_windows, batch_size):
        batch_indices = window_starts[batch_start:batch_start + batch_size]
        if not batch_indices:
            break

        # 构建 batch: (B, WINDOW, 64, 5)
        batch_windows = []
        for i in batch_indices:
            window_frames = point_frames[i:i + WINDOW]
            batch_windows.append(
                np.stack([pad_to_64(f, rng) for f in window_frames])
            )

        batch_t = torch.from_numpy(
            np.stack(batch_windows).astype(np.float32)
        ).to(device)  # (B, WINDOW, 64, 5)

        with torch.no_grad():
            out = encoder(batch_t)
            S = out["S"]  # (B, WINDOW, 256)

        # 每个窗口: 前31帧→输入, 第32帧→目标
        X_chunks.append(S[:, :PRED_WINDOW].cpu().numpy())   # (B, 31, 256)
        y_chunks.append(S[:, PRED_WINDOW].cpu().numpy())    # (B, 256)

        if (batch_start // batch_size) % 20 == 0:
            print(f"  Encoder batch {batch_start // batch_size + 1}: "
                  f"{batch_start + len(batch_indices)}/{total_windows} windows")

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="生成 Predictor 训练数据")
    parser.add_argument("--hours", type=float, default=2.0,
                        help="仿真时长（小时），默认 2")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "data", "training", "predictor_continuous.npz"),
                        help="输出 .npz 路径")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                        help="训练被试（默认 0-5，留 6 做 eval）")
    parser.add_argument("--dataset", default=DATASET_PATH)
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── 加载 Encoder ──────────────────────────────────
    print(f"加载 Encoder: {ENCODER_PATH}")
    encoder = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)
    ckpt = torch.load(ENCODER_PATH, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    encoder.eval()
    print(f"  Params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M")

    # ── 加载 ReplayGenerator ─────────────────────────
    print(f"加载数据集: {args.dataset}")
    replay = ReplayGenerator(args.dataset)
    rng = np.random.RandomState(42)

    # ── 生成数据 ─────────────────────────────────────
    activity_seq = generate_activity_sequence(args.hours, rng)
    total_s = sum(d for _, d in activity_seq)
    print(f"\n生成 {args.hours}h 连续生活场景 ({total_s:.0f}s, {len(activity_seq)} 个活动段)")

    all_S = []        # per-frame S_t
    all_actions = []  # action label per frame (for debugging)
    all_subjects_per_frame = []

    t0 = time.time()
    point_buf: list[np.ndarray] = []  # body-centered raw point clouds
    frame_actions: list[int] = []

    for seg_i, (action_name, duration_s) in enumerate(activity_seq):
        # 选一个被试（同一场景内尽量用同一被试，保证时序一致性）
        subject_id = rng.choice(args.subjects)

        # 用 ReplayGenerator 获取该动作的连续帧
        frames = replay.generate_sequence(action_name, duration_s, subject_ids=[subject_id])
        if not frames:
            continue

        action_id = ACTION_NAMES.index(action_name) if action_name in ACTION_NAMES else 0

        for pts in frames:
            point_buf.append(pts.astype(np.float32))
            frame_actions.append(action_id)

        if seg_i % 20 == 0:
            elapsed = time.time() - t0
            sim_s = sum(d for _, d in activity_seq[:seg_i + 1])
            print(f"  [{elapsed:.0f}s] {sim_s:.0f}s/{total_s:.0f}s  "
                  f"帧数={len(point_buf)}  当前: {action_name}({duration_s:.0f}s)")

    print(f"\n点云帧总数: {len(point_buf)}  ({time.time() - t0:.0f}s)")

    # ── Encoder 批量提取训练样本 ────────────────────
    print(f"\nEncoder 批量提取训练样本 (window={WINDOW}, stride=32)...")
    t1 = time.time()

    result = extract_training_samples(encoder, point_buf, device, batch_size=64, stride=32)
    if result is None:
        print("ERROR: 点云帧太少，请增加 --hours")
        sys.exit(1)

    X, y = result
    print(f"  X{X.shape} y{y.shape}  ({time.time() - t1:.0f}s)")

    # ── 划分 train/val ───────────────────────────────
    n_val = min(int(len(X) * 0.1), 5000)
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]

    # ── 保存 ─────────────────────────────────────────
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        args.output,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        latent_dim=256,
        window=PRED_WINDOW,
        total_hours=args.hours,
        n_point_frames=len(point_buf),
        n_latent_frames=len(X),
        encoder_epoch=ckpt.get("epoch", -1),
        encoder_acc=float(ckpt.get("linear_acc", 0)),
    )
    print(f"\n✅ 保存到: {args.output}")
    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val:   {len(X_val):,} samples")
    print(f"   Latent dim: 256, Predictor window: {PRED_WINDOW}")


if __name__ == "__main__":
    main()
