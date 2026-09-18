#!/usr/bin/env python3
"""纯自监督 Encoder 训练 — 单文件，零依赖，直接上传 GPU 运行。

用法:
  scp train_noaux.py 3dpchm_frames.npz gpu-server:~/
  ssh gpu-server
  pip install torch numpy scipy scikit-learn tqdm
  python train_noaux.py

输出: encoder_noaux.pt (纯自监督, aux_weight=0)
"""

import os, sys, argparse, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════
# 1. PointNet++ Backbone
# ═══════════════════════════════════════════════════════════

def square_distance(src, dst):
    B, N, _ = src.shape; _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, npoint):
    device = xyz.device; B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device; B, N, _ = xyz.shape; _, S, _ = new_xyz.shape
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = sqrdists.sort(dim=-1)
    group_idx = group_idx[:, :, :nsample]
    dist_sorted = sqrdists.gather(dim=-1, index=group_idx)
    mask = dist_sorted > radius ** 2
    group_first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)
    group_idx[mask] = group_first[mask]
    return group_idx

def index_points(points, idx):
    device = points.device; B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.mlp_convs = nn.ModuleList(); self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        B, N, _ = xyz.shape; S = min(self.npoint, N)
        fps_idx = farthest_point_sample(xyz, S)
        new_xyz = index_points(xyz, fps_idx.unsqueeze(-1).repeat(1, 1, 1)).squeeze(2)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm
        new_points = new_points.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=-1)[0].permute(0, 2, 1)
        return new_xyz, new_points

class PointNet2Backbone(nn.Module):
    def __init__(self, in_channels=5, out_channels=1024):
        super().__init__()
        self.sa1 = SetAbstraction(16, 0.4, 16, 3+(in_channels-3), [64,64,128])
        self.sa2 = SetAbstraction(8, 0.8, 16, 3+128, [128,128,256])
        self.sa3 = SetAbstraction(1, 2.0, 8, 3+256, [256,512,out_channels])

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        feat1 = features if features is not None else None
        xyz1, feat1 = self.sa1(xyz, feat1)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)
        return feat3.squeeze(1)

# ═══════════════════════════════════════════════════════════
# 2. TCN
# ═══════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return F.relu(self.bn(self.conv(x)))

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([CausalConv1d(hidden_dim, hidden_dim, 3, dilation=d) for d in [1,2,4,8]])
        self.output_proj = nn.Sequential(nn.Conv1d(hidden_dim, output_dim, 1), nn.BatchNorm1d(output_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual
        x = self.output_proj(x)
        return x.permute(0, 2, 1)

# ═══════════════════════════════════════════════════════════
# 3. Encoder Model
# ═══════════════════════════════════════════════════════════

class EncoderModel(nn.Module):
    def __init__(self, pointnet_out=1024, tcn_hidden=512, latent_dim=256, n_actions=12):
        super().__init__()
        self.backbone = PointNet2Backbone(in_channels=5, out_channels=pointnet_out)
        self.tcn = TemporalConvNet(input_dim=pointnet_out, hidden_dim=tcn_hidden, output_dim=latent_dim)
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, n_actions))
        self.projection = nn.Sequential(nn.Linear(latent_dim, 128))

    def forward(self, seq):
        B, T, N, D = seq.shape
        # 合并 batch 和 time 维度，一次性处理所有帧
        seq_flat = seq.reshape(B * T, N, D)
        xyz = seq_flat[:, :, :3]
        feats = seq_flat[:, :, 3:]
        pf_flat = self.backbone(xyz, feats)  # (B*T, 1024)
        pf_stack = pf_flat.reshape(B, T, -1)  # (B, T, 1024)
        S = self.tcn(pf_stack)
        S_mean = S.mean(dim=1)
        logits = self.classifier(S_mean)
        z = self.projection(S)
        return {"S": S, "S_mean": S_mean, "logits": logits, "z": z}

# ═══════════════════════════════════════════════════════════
# 4. Loss
# ═══════════════════════════════════════════════════════════

class EncoderLoss(nn.Module):
    def __init__(self, temperature=0.1, aux_weight=0.0, smooth_weight=0.1, var_weight=0.01):
        super().__init__()
        self.temperature = temperature
        self.aux_weight = aux_weight
        self.smooth_weight = smooth_weight
        self.var_weight = var_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z, S, logits, labels):
        B, T, _ = z.shape
        z_flat = F.normalize(z.reshape(B * T, -1), dim=-1)
        sim = torch.matmul(z_flat, z_flat.T) / self.temperature
        pos_mask = torch.zeros(B * T, B * T, device=z.device)
        for b in range(B):
            s, e = b * T, (b + 1) * T
            pos_mask[s:e, s:e] = 1.0
        pos_mask.fill_diagonal_(0.0)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.exp(sim.diagonal())
        all_sum = all_sum.clamp(min=1e-10)
        L_contrastive = -torch.log((pos_sum / all_sum).clamp(min=1e-10)).mean()
        L_aux = self.ce(logits, labels)
        L_smooth = F.mse_loss(S[:, 1:], S[:, :-1])
        var = S.var(dim=(0, 1)).mean()
        L_var = F.mse_loss(var, torch.ones_like(var))
        total = L_contrastive + self.aux_weight * L_aux + self.smooth_weight * L_smooth + self.var_weight * L_var
        return {"total": total, "contrastive": L_contrastive, "aux": L_aux, "smooth": L_smooth, "var": L_var}

# ═══════════════════════════════════════════════════════════
# 5. Dataset Loading (minimal)
# ═══════════════════════════════════════════════════════════

def load_frames_fast(path):
    data = np.load(path, allow_pickle=True)
    pts = data["points"]
    subjects = data["subject_ids"].astype(np.int32)
    actions = data["action_labels"].astype(np.int32)
    frame_ids = data["frame_indices"].astype(np.int32)
    return pts, subjects, actions, frame_ids

def build_sequence_samples(pts_list, subjects, actions, frame_ids, heldout_subjects=None, exclude_fall=True, window_size=32, stride=8):
    heldout = set(heldout_subjects or [])
    is_heldout = np.isin(subjects, list(heldout))
    is_fall = (actions == 2)
    mask = ~is_heldout
    if exclude_fall: mask = mask & (~is_fall)
    valid_idx = np.where(mask)[0]
    subj_groups = defaultdict(list)
    for idx in valid_idx: subj_groups[int(subjects[idx])].append(idx)
    samples = []
    for sid, idx_list in subj_groups.items():
        idx_arr = np.array(idx_list)
        order = np.argsort(frame_ids[idx_arr])
        idx_arr = idx_arr[order]
        n = len(idx_arr)
        for start in range(0, n - window_size + 1, stride):
            window_indices = idx_arr[start:start + window_size].tolist()
            window_actions = actions[window_indices]
            majority_label = int(np.bincount(window_actions).argmax())
            samples.append((window_indices, majority_label))
    return samples

class FastSequenceDataset(Dataset):
    def __init__(self, pts, samples, max_points=64):
        self.pts = pts; self.samples = samples; self.max_points = max_points
        self._rng = np.random.RandomState(42)  # 复用，避免每次 __getitem__ 创建
        self._zero = np.zeros((max_points, 5), dtype=np.float32)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        indices, label = self.samples[idx]
        frames = np.empty((len(indices), self.max_points, 5), dtype=np.float32)
        for fi, i in enumerate(indices):
            p = self.pts[i]
            n = len(p)
            if n == 0:
                frames[fi] = self._zero
            elif n >= self.max_points:
                frames[fi] = p[self._rng.choice(n, self.max_points, replace=False)]
            else:
                extra = self.max_points - n
                idx_extra = self._rng.randint(0, n, size=extra)
                jitter = self._rng.randn(extra, 5).astype(np.float32) * 0.01
                frames[fi, :n] = p
                frames[fi, n:] = p[idx_extra] + jitter
        return torch.from_numpy(frames).float(), label


# ═══════════════════════════════════════════════════════════
# 6. Training
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="3dpchm_frames.npz")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="encoder_noaux.pt")
    return p.parse_args()

def linear_eval(model, train_loader, test_loader, device):
    model.eval()
    def extract(loader):
        feats, labs = [], []
        with torch.no_grad():
            for seq, labels in loader:
                out = model(seq.to(device))
                feats.append(out["S_mean"].cpu().numpy())
                labs.append(labels.numpy())
        return np.vstack(feats), np.concatenate(labs)
    X_train, y_train = extract(train_loader)
    X_test, y_test = extract(test_loader)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.data}...")
    pts, subjects, actions, frame_ids = load_frames_fast(args.data)
    print(f"  {len(pts):,} frames, {len(set(subjects))} subjects")

    train_samples = build_sequence_samples(pts, subjects, actions, frame_ids,
                                           heldout_subjects=[6], exclude_fall=True,
                                           window_size=32, stride=8)
    val_samples = build_sequence_samples(pts, subjects, actions, frame_ids,
                                         heldout_subjects=[6], exclude_fall=True,
                                         window_size=32, stride=32)
    print(f"  {len(train_samples):,} train / {len(val_samples):,} val windows")

    train_ds = FastSequenceDataset(pts, train_samples, max_points=64)
    val_ds = FastSequenceDataset(pts, val_samples, max_points=64)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    lin_train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    lin_val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256, n_actions=12)
    model = model.to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = EncoderLoss(temperature=0.1, aux_weight=0.0, smooth_weight=0.1, var_weight=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for seq, labels in pbar:
            seq = seq.to(device); labels = labels.to(device)
            out = model(seq)
            losses = criterion(out["z"], out["S"], out["logits"], labels)
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += losses["total"].item(); n_batches += 1
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0; val_n = 0
            with torch.no_grad():
                for seq, labels in val_loader:
                    seq = seq.to(device); labels = labels.to(device)
                    out = model(seq)
                    losses = criterion(out["z"], out["S"], out["logits"], labels)
                    val_loss += losses["total"].item(); val_n += 1
            val_loss /= max(val_n, 1)
            try:
                acc = linear_eval(model, lin_train_loader, lin_val_loader, device)
            except:
                acc = 0.0
            print(f"  Epoch {epoch+1}: train={avg_loss:.4f} val={val_loss:.4f} lin_acc={acc:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                            "loss": val_loss, "linear_acc": acc}, args.out)
                print(f"  → Saved {args.out}")

        if epoch > 50 and avg_loss < 0.001:
            print(f"  Converged at epoch {epoch+1}"); break

    print(f"Done. {time.time()-t0:.0f}s. Best → {args.out}")

if __name__ == "__main__":
    main()
