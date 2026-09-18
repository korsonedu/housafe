"""7-fold CV with noaux encoder — batched version."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ai.encoder.pointnet_backbone import PointNet2Backbone
from ai.encoder.tcn import TemporalConvNet

WINDOW, STRIDE, MAX_PTS = 32, 8, 64

class SEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PointNet2Backbone(in_channels=5, out_channels=1024)
        self.tcn = TemporalConvNet(input_dim=1024, hidden_dim=512, output_dim=256)

    def forward_backbone_batch(self, frames):
        """frames: (N, 64, 5) → (N, 1024)"""
        xyz = frames[:, :, :3]; feats = frames[:, :, 3:]
        return self.backbone(xyz, feats)

    def forward_tcn_batch(self, windows):
        """windows: (M, 32, 1024) → (M, 32, 256)"""
        return self.tcn(windows)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ckpt = torch.load(os.path.join(_ROOT, 'ai', 'encoder', 'encoder_noaux.pt'),
                  map_location='cpu', weights_only=True)
model = SEncoder()
sd = {k: v for k, v in ckpt['model_state_dict'].items() if 'backbone' in k or 'tcn' in k}
model.load_state_dict(sd, strict=False)
model.eval()
dev = torch.device('cpu')

def pad_frames(frames):
    """批量 pad → (N, 64, 5)"""
    N = len(frames)
    padded = np.zeros((N, MAX_PTS, 5), dtype=np.float32)
    rng = np.random.RandomState(42)
    for i in range(N):
        p = frames[i]; n = len(p)
        if n == 0: continue
        elif n >= MAX_PTS: padded[i] = p[rng.choice(n, MAX_PTS, replace=False)]
        else:
            padded[i, :n] = p
            extra = MAX_PTS - n; idx = rng.randint(0, n, size=extra)
            padded[i, n:] = p[idx] + rng.randn(extra, 5).astype(np.float32) * 0.01
    return padded

def extract_S_batched(enc, frames):
    """批量提取 S_t: backbone batch → window batch → TCN"""
    N = len(frames)
    if N < WINDOW: return np.array([], dtype=np.float32)

    padded = pad_frames(frames)  # (N, 64, 5)

    # 1. Backbone batch
    bb_batch = 500
    bb_all = []
    for s in range(0, N, bb_batch):
        e = min(s + bb_batch, N)
        batch = torch.from_numpy(padded[s:e]).to(dev)
        with torch.no_grad(): bb_all.append(enc.forward_backbone_batch(batch).cpu().numpy())
    bb = np.concatenate(bb_all)  # (N, 1024)

    # 2. Window batch → TCN
    win_batch = 200
    windows, wb_idx = [], []
    S_list = []
    for center in range(WINDOW // 2, N - WINDOW // 2, STRIDE):
        start = center - WINDOW // 2
        windows.append(bb[start:start + WINDOW])
        if len(windows) >= win_batch:
            batch = torch.from_numpy(np.stack(windows)).to(dev)
            with torch.no_grad(): S_batch = enc.forward_tcn_batch(batch).cpu().numpy()
            S_list.append(S_batch.reshape(-1, 256))
            windows = []
    if windows:
        batch = torch.from_numpy(np.stack(windows)).to(dev)
        with torch.no_grad(): S_batch = enc.forward_tcn_batch(batch).cpu().numpy()
        S_list.append(S_batch.reshape(-1, 256))

    return np.concatenate(S_list) if S_list else np.array([], dtype=np.float32)


from ai.baseline.latent_baseline import LatentBaseline
from simulator.generators import ACTION_NAMES

DATASET_PATH = os.path.join(_ROOT, 'ai', 'data', 'public', 'processed', '3dpchm_frames.npz')
data = np.load(DATASET_PATH, allow_pickle=True)
pts = data['points']; actions = data['action_labels'].astype(np.int32)
subjects = data['subject_ids'].astype(np.int32)
fall_id = list(ACTION_NAMES).index('fall')

results = []
for ts_id in range(7):
    t0 = time.time()
    print(f"S{ts_id}: building index...", end=" ", flush=True)
    train_idx = []
    for sid in [s for s in range(7) if s != ts_id]:
        for aid in set(actions):
            if aid == fall_id: continue
            idx = np.where((subjects == sid) & (actions == aid))[0]
            if len(idx) > 1000: idx = np.random.RandomState(42).choice(idx, 1000, replace=False)
            train_idx.extend(idx)
    train_idx = np.array(train_idx)
    tr_frames = [pts[i].astype(np.float32) for i in train_idx]
    print(f"{len(tr_frames)} frames, extracting...", end=" ", flush=True)
    S_tr = extract_S_batched(model, tr_frames)
    if len(S_tr) < 500:
        print(f"train={len(S_tr)} (skip)")
        continue
    tr_ts = np.array([1753891200000 + int(i/len(S_tr)*14*86400*1000) for i in range(len(S_tr))], dtype=np.int64)

    print(f"{len(S_tr)} S_t, fitting GMM...", end=" ", flush=True)
    lb = LatentBaseline(); lb.fit(S_tr, tr_ts)
    if not lb.is_ready:
        print(f"fail ({len(lb._gmms)} ctx)")
        continue
    print(f"{len(lb._gmms)} ctx, extracting test...", end=" ", flush=True)

    tn_idx = np.where((subjects == ts_id) & (actions != fall_id))[0]
    if len(tn_idx) > 5000: tn_idx = np.random.RandomState(42).choice(tn_idx, 5000, replace=False)
    tf_idx = np.where((subjects == ts_id) & (actions == fall_id))[0]

    tn_frames = [pts[i].astype(np.float32) for i in tn_idx]
    tf_frames = [pts[i].astype(np.float32) for i in tf_idx]
    S_n = extract_S_batched(model, tn_frames)
    S_f = extract_S_batched(model, tf_frames)
    if len(S_n) < 5 or len(S_f) < 3:
        print(f"test too small (n={len(S_n)}, f={len(S_f)})")
        continue
    tn_ts = np.array([1753891200000+100*86400000+int(i/len(S_n)*86400) for i in range(len(S_n))], dtype=np.int64)
    tf_ts = np.array([1753891200000+200*86400000+int(i/len(S_f)*86400) for i in range(len(S_f))], dtype=np.int64)

    print(f"n={len(S_n)} f={len(S_f)}, scoring...", end=" ", flush=True)
    ns = [lb.score(S_n[i], tn_ts[i]) for i in range(len(S_n))]
    fs = [lb.score(S_f[i], tf_ts[i]) for i in range(len(S_f))]
    auc = roc_auc_score([0]*len(ns)+[1]*len(fs), ns+fs)
    results.append(auc)
    print(f"AUC={auc:.4f} ({time.time()-t0:.0f}s)")

print(f"\n=== Noaux 7-Fold CV ===")
print(f"Mean AUC = {np.mean(results):.4f} +/- {np.std(results):.4f}")
print(f"Per-subject: {[round(r,4) for r in results]}")
print(f"Subjects: {len(results)}/7")
