#!/usr/bin/env python3
"""
Encoder 训练脚本 — PointNet++ + TCN 联合训练。

用法:
  python ai/encoder/train_encoder.py \
    --data ai/data/public/processed/3dpchm_frames.npz \
    --epochs 300 --batch 32 --lr 1e-4 \
    --out ai/checkpoints

远端 GPU (AutoDL):
  pip install torch numpy scipy scikit-learn tqdm
  python ai/encoder/train_encoder.py --data ./3dpchm_frames.npz --device cuda
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# 确保 ai/ 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.encoder.encoder_model import EncoderModel
from ai.encoder.losses import EncoderLoss
from ai.data.public.dataset import (load_frames_fast, build_sequence_samples,
                                      FastRadarDataset, FastSequenceDataset)

# ── 参数 ────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Encoder (PointNet++ + TCN)")
    p.add_argument("--data", required=True, help="Path to .npz file")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux-weight", type=float, default=0.05,
                   help="Auxiliary classifier loss weight (0 = pure self-supervised)")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--max-points", type=int, default=64)
    p.add_argument("--heldout", type=int, default=6, help="Subject 0-indexed to hold out")
    p.add_argument("--out", default="ai/checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="", help="Resume from checkpoint")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


# ── 工具 ────────────────────────────────────────────────

def linear_eval(model: EncoderModel, train_loader: DataLoader,
                test_loader: DataLoader, device: torch.device) -> float:
    """训练监控：冻结 Encoder → 逻辑回归 → 分类准确率（仅供观察，非产品指标）"""
    model.eval()

    def extract_features(loader):
        feats, labs = [], []
        with torch.no_grad():
            for seq, labels in loader:
                seq = seq.to(device)
                out = model(seq)
                feats.append(out["S_mean"].cpu().numpy())
                labs.append(labels.numpy())
        return np.vstack(feats), np.concatenate(labs)

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return float(acc)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, loss: float, acc: float, path: str):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "linear_acc": acc,
    }, path)


# ── 主训练 ──────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 数据 ──
    print(f"\nLoading data from {args.data}...")
    pts, subjects, actions, frame_ids = load_frames_fast(args.data)
    print(f"  Loaded {len(pts):,} frames in memory")

    t0 = time.time()
    # 训练时序窗口
    train_samples = build_sequence_samples(
        pts, subjects, actions, frame_ids,
        heldout_subjects=[args.heldout], exclude_fall=True,
        window_size=args.window, stride=8,
    )
    # 验证时序窗口 — stride=window, 不重叠，独立评估
    val_samples = build_sequence_samples(
        pts, subjects, actions, frame_ids,
        heldout_subjects=[args.heldout], exclude_fall=True,
        window_size=args.window, stride=args.window,
    )
    print(f"  Built {len(train_samples):,} train / {len(val_samples):,} val windows in {time.time()-t0:.1f}s")

    train_ds = FastSequenceDataset(pts, train_samples, max_points=args.max_points)
    val_ds = FastSequenceDataset(pts, val_samples, max_points=args.max_points)
    # 分类评估 Dataset (逐帧) — 训练用排除 fall，验证用全部
    cls_train = FastRadarDataset(pts, subjects, actions, frame_ids,
                                  split="train", heldout_subjects=[args.heldout],
                                  exclude_fall=True, max_points=args.max_points)
    cls_val = FastRadarDataset(pts, subjects, actions, frame_ids,
                                split="val", heldout_subjects=[args.heldout],
                                exclude_fall=False, max_points=args.max_points)

    print(f"Train seqs: {len(train_ds):,} | Val seqs: {len(val_ds):,}")
    print(f"Cls train frames: {len(cls_train):,} | Cls val frames: {len(cls_val):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=2, pin_memory=True)
    # 线性评估用的 loader（时序窗口）
    lin_train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                   num_workers=2, pin_memory=True)
    lin_val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=2, pin_memory=True)

    # ── 模型 ──
    model = EncoderModel(pointnet_out=1024, tcn_hidden=512,
                          latent_dim=256, n_actions=12)
    model = model.to(device)

    criterion = EncoderLoss(temperature=0.1, aux_weight=args.aux_weight,
                             smooth_weight=0.1, var_weight=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    os.makedirs(args.out, exist_ok=True)

    if args.eval_only:
        acc = linear_eval(model, lin_train_loader, lin_val_loader, device)
        print(f"Linear eval accuracy: {acc:.4f}")
        return

    # ── 训练循环 ──
    best_acc = 0.0
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {"total": 0.0, "contrastive": 0.0, "aux": 0.0,
                        "smooth": 0.0, "var": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for seq, labels in pbar:
            seq = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = model(seq)
            losses = criterion(out["z"], out["S"], out["logits"], labels)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.3f}",
                "contrast": f"{losses['contrastive'].item():.3f}",
                "aux": f"{losses['aux'].item():.3f}",
            })

        scheduler.step()

        # 平均损失
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        avg_loss = epoch_losses["total"]

        # ── 每 10 epoch 评估 ──
        if (epoch + 1) % 10 == 0 or epoch < 10:
            model.eval()
            acc = linear_eval(model, lin_train_loader, lin_val_loader, device)
            elapsed = time.time() - t0
            print(f"  [Eval] Epoch {epoch+1} | "
                  f"Loss: {avg_loss:.4f} | Linear Acc: {acc:.4f} | "
                  f"Time: {elapsed/60:.1f}m | LR: {scheduler.get_last_lr()[0]:.2e}")

            # 保存最佳
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, optimizer, epoch, avg_loss, acc,
                                os.path.join(args.out, "encoder_best.pt"))
                print(f"  → Saved best (acc={acc:.4f})")

            # 定期保存
            save_checkpoint(model, optimizer, epoch, avg_loss, acc,
                            os.path.join(args.out, f"encoder_epoch{epoch+1:03d}.pt"))

            model.train()

        # Early stop: 损失 30 epoch 不降
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 30:
            print(f"\nEarly stop at epoch {epoch+1} (no improvement for 30 epochs)")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Training done in {elapsed/3600:.1f}h")
    print(f"Best linear accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {args.out}")


if __name__ == "__main__":
    main()
