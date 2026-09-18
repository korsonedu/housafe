#!/usr/bin/env python3
"""
Predictor 训练 v2 — 连续生活场景中的 256-d latent state 序列。

关键改进:
  1. 输入: 256-d S_t (TCN latent state)，不是 1024-d backbone output
  2. 训练数据: 连续场景中的跨动作序列，不是孤立动作片段
  3. 模型: ai.predictor.model.TinyPredictor (in_dim=256)

用法:
  python -m ai.predictor.train_predictor_v2 --data ai/data/training/predictor_continuous.npz
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.predictor.model import TinyPredictor

BATCH = 128
EPOCHS = 150
LR = 1e-3


def parse_args():
    p = argparse.ArgumentParser(description="Predictor 训练 v2 — 连续场景")
    p.add_argument("--data", required=True, help="generate_training_data.py 输出的 .npz")
    p.add_argument("--output", default=None, help="模型输出路径")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--lr", type=float, default=LR)
    return p.parse_args()


def compute_errors(model, loader, device, max_samples=None):
    """计算所有样本的 cosine error"""
    model.eval()
    errs = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            err = 1.0 - nn.functional.cosine_similarity(pred, y_batch, dim=-1)
            errs.append(err.cpu().numpy())
            if max_samples and sum(len(e) for e in errs) >= max_samples:
                break
    return np.concatenate(errs)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── 加载数据 ─────────────────────────────────────
    print(f"加载训练数据: {args.data}")
    d = np.load(args.data)
    X_train = d["X_train"].astype(np.float32)  # (N, 31, 256)
    y_train = d["y_train"].astype(np.float32)  # (N, 256)
    X_val = d["X_val"].astype(np.float32)
    y_val = d["y_val"].astype(np.float32)

    print(f"  Train: {X_train.shape[0]:,} samples")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Input shape: {X_train.shape[1:]} (T={X_train.shape[1]}, D={X_train.shape[2]})")
    print(f"  Target shape: {y_train.shape[1:]}")

    # ── 数据加载器 ─────────────────────────────────
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    # ── Persistence baseline ─────────────────────────
    print("\n=== Persistence Baseline ===")
    # Baseline: 用 S[t] 直接当作 S[t+1] 的预测
    bl_errors = []
    for i in range(min(2000, len(X_val))):
        # S[-1] is the last known state, compare with y
        last_S = X_val[i, -1]                       # (256,)
        target_S = y_val[i]                          # (256,)
        err = 1.0 - nn.functional.cosine_similarity(
            torch.from_numpy(last_S), torch.from_numpy(target_S), dim=0
        ).item()
        bl_errors.append(err)
    bl_errors = np.array(bl_errors)
    print(f"  Error (S[t] → S[t+1]):  {bl_errors.mean():.6f} ± {bl_errors.std():.6f}")

    # ── 模型 ────────────────────────────────────────
    print(f"\n=== 模型 ===")
    model = TinyPredictor(in_dim=256, hid=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  TinyPredictor(in_dim=256, hid=128): {n_params:,} params")
    print(f"  Input: (B, T, 256) → Output: (B, 256)")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # ── 训练 ────────────────────────────────────────
    print(f"\n=== Training ({args.epochs} epochs) ===")
    t0 = time.time()
    best_val_err = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = (1.0 - nn.functional.cosine_similarity(pred, y_batch, dim=-1)).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        sched.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        if (epoch + 1) % 10 == 0:
            val_err = compute_errors(model, val_loader, device, max_samples=2000)
            val_mean = val_err.mean()
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train loss: {avg_loss:.6f} | "
                  f"Val error: {val_mean:.6f} ± {val_err.std():.6f} | "
                  f"LR: {sched.get_last_lr()[0]:.2e}")

            if val_mean < best_val_err:
                best_val_err = val_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    print(f"\n  训练完成 ({train_time:.0f}s, {train_time/60:.1f}min)")
    print(f"  Best val error: {best_val_err:.6f}")

    # ── 评估 ────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    val_errors = compute_errors(model, val_loader, device)
    print(f"\n=== Results ===")
    print(f"  Val error:     {val_errors.mean():.6f} ± {val_errors.std():.6f}")
    print(f"  Persistence:   {bl_errors.mean():.6f} ± {bl_errors.std():.6f}")
    imp = (bl_errors.mean() - val_errors.mean()) / (bl_errors.mean() + 1e-10)
    print(f"  Improvement:   {imp*100:.1f}%  {'✅' if imp > 0.3 else '⚠️'}")

    # ── 保存 ────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "checkpoints", "predictor_best.pt",
        )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 备份旧模型
    if os.path.exists(output_path):
        backup = output_path + ".bak"
        print(f"\n  备份旧模型: {backup}")
        os.rename(output_path, backup)

    torch.save(model.state_dict(), output_path)
    print(f"  模型保存到: {output_path}")

    # ── 简单异常检测测试 ────────────────────────────
    print(f"\n=== 异常检测能力评估 ===")
    # 用 val error 的分布：error > mean + 2*std 为异常
    threshold = val_errors.mean() + 2 * val_errors.std()
    anomaly_rate = (val_errors > threshold).mean()
    print(f"  Threshold (μ+2σ): {threshold:.4f}")
    print(f"  Val 中 > threshold: {anomaly_rate:.1%}")
    print(f"  P95: {np.percentile(val_errors, 95):.4f}")
    print(f"  P99: {np.percentile(val_errors, 99):.4f}")


if __name__ == "__main__":
    main()
