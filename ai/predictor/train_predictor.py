#!/usr/bin/env python3
"""
快速 Predictor 训练 + H3/H4 评估 — 30 分钟出结果。

改进:
  - 输入投影 1024→128，小型 GRU（不是 44M Transformer）
  - 线性 baseline 直接算
  - tqdm 进度条，每 epoch 可见
"""

import sys, os, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BATCH, EPOCHS, LR, WINDOW, HELDOUT = 128, 100, 1e-3, 32, 6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--latent", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class FastPredDataset(Dataset):
    def __init__(self, S, subjects, actions, frame_ids,
                 heldout=None, exclude_fall=True, window=32):
        heldout = heldout or []
        mask = ~np.isin(subjects, heldout)
        if exclude_fall:
            mask = mask & (actions != 2)
        valid = np.where(mask)[0]
        # group by subject, sort by frame, build windows
        samples = []
        for sid in np.unique(subjects[valid]):
            idx = valid[subjects[valid] == sid]
            idx = idx[np.argsort(frame_ids[idx])].tolist()
            for t in range(window, len(idx)):
                samples.append((idx[t-window:t], idx[t]))
        self.samples = samples
        self.S = S

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        h, t = self.samples[i]
        return (torch.from_numpy(self.S[h].astype(np.float32)),
                torch.from_numpy(self.S[t].astype(np.float32)))


class TinyPredictor(nn.Module):
    """小预测器: input_proj(1024→128) + 2-layer GRU → output_proj(128→1024)"""
    def __init__(self, in_dim=1024, hid=128):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hid)
        self.gru = nn.GRU(hid, hid, 2, batch_first=True)
        self.proj_out = nn.Linear(hid, in_dim)

    def forward(self, x):
        x = self.proj_in(x)
        _, h = self.gru(x)
        return self.proj_out(h[-1])


def compute_errors(model, ds, device, n=None):
    model.eval()
    errs = []
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    with torch.no_grad():
        for h, t in loader:
            h, t = h.to(device), t.to(device)
            p = model(h)
            errs.append((1 - nn.functional.cosine_similarity(p, t)).cpu().numpy())
            if n and sum(len(e) for e in errs) >= n:
                break
    return np.concatenate(errs)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load
    print("Loading...")
    d = np.load(args.latent, allow_pickle=True)
    S, subj, act, fid = d["S"].astype(np.float32), d["subjects"], d["actions"], d["frame_ids"]

    # Datasets
    t0 = time.time()
    train_ds = FastPredDataset(S, subj, act, fid, [HELDOUT], exclude_fall=True, window=WINDOW)
    normal_ds = FastPredDataset(S, subj, act, fid, [HELDOUT], exclude_fall=True, window=WINDOW)
    normal_ds.samples = [(h, t) for h, t in normal_ds.samples if act[t] != 2][:3000]
    fall_ds = FastPredDataset(S, subj, act, fid, [HELDOUT], exclude_fall=False, window=WINDOW)
    fall_ds.samples = [(h, t) for h, t in fall_ds.samples if act[t] == 2]
    print(f"Train: {len(train_ds):,} | Norm: {len(normal_ds):,} | Fall: {len(fall_ds):,} "
          f"({time.time()-t0:.0f}s)")

    # H3 baseline: cosine(S_t, S_{t+1})
    print("\n=== H3 Baseline ===")
    bl = []
    for h, t in FastPredDataset(S, subj, act, fid, [HELDOUT], exclude_fall=True, window=2).samples[:5000]:
        bl.append(1 - nn.functional.cosine_similarity(
            torch.from_numpy(S[h[-1]]), torch.from_numpy(S[t]), dim=0).item())
    bl = np.array(bl)
    print(f"Persistence baseline: {bl.mean():.6f} ± {bl.std():.6f}")

    # Train
    print(f"\n=== Training ({EPOCHS} epochs) ===")
    model = TinyPredictor(S.shape[1], 128).to(device)
    opt = optim.AdamW(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for h, t in loader:
            h, t = h.to(device), t.to(device)
            loss = (1 - nn.functional.cosine_similarity(model(h), t)).mean()
            opt.zero_grad(); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total += loss.item()
        sched.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            tr_err = compute_errors(model, train_ds, device, 3000).mean()
            print(f"  Epoch {epoch+1:3d} | Loss: {total/len(loader):.6f} | TrainErr: {tr_err:.6f} | LR: {sched.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), "/tmp/predictor_best.pt")

    # H3
    print("\n=== H3: Prediction Accuracy ===")
    pred_err = compute_errors(model, normal_ds, device, 3000)
    imp = (bl.mean() - pred_err.mean()) / (bl.mean() + 1e-10)
    print(f"  Predictor: {pred_err.mean():.6f} ± {pred_err.std():.6f}")
    print(f"  Baseline:  {bl.mean():.6f} ± {bl.std():.6f}")
    print(f"  Improve:   {imp*100:.1f}%  {'✅' if imp > 0.3 else '❌'}")

    # H4
    print("\n=== H4: Anomaly Detection ===")
    n_err = compute_errors(model, normal_ds, device, 3000)
    f_err = compute_errors(model, fall_ds, device, 3000)
    ratio = f_err.mean() / (n_err.mean() + 1e-10)
    t_s, p_v = stats.ttest_ind(f_err, n_err)
    auc = roc_auc_score(
        np.concatenate([np.zeros(len(n_err)), np.ones(len(f_err))]),
        np.concatenate([n_err, f_err]))
    print(f"  Normal:  {n_err.mean():.6f} ± {n_err.std():.6f}")
    print(f"  Fall:    {f_err.mean():.6f} ± {f_err.std():.6f}")
    print(f"  Ratio:   {ratio:.2f}x  {'✅' if ratio > 2 else '❌'}")
    print(f"  AUC:     {auc:.4f}     {'✅' if auc > 0.9 else '❌'}")
    print(f"  t-test:  t={t_s:.2f}, p={p_v:.2e}  {'✅' if p_v < 0.01 else '❌'}")
    print(f"\n=== 结论 ===")
    if imp > 0.3 and ratio > 2 and auc > 0.9 and p_v < 0.01:
        print("✅ 世界模型路线成立")
    else:
        print("❌ 部分假设未通过，见上")


if __name__ == "__main__":
    main()
