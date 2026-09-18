"""NLinear 预测基线 vs 密度方法 — 逐 fold 处理，避免 OOM."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from ai.drift.common import load_encoder, extract_S_sequence
from ai.baseline.latent_baseline import LatentBaseline
from simulator.generators import ACTION_NAMES

DATASET = 'ai/data/public/processed/3dpchm_frames.npz'
WINDOW, DEVICE = 32, torch.device('cpu')

class NLinear(nn.Module):
    def __init__(self, seq_len=32, feat_dim=256):
        super().__init__()
        self.linear = nn.Linear(seq_len * feat_dim, feat_dim)
    def forward(self, x):
        B, T, D = x.shape
        x_last = x[:, -1, :]
        x_norm = x - x_last.unsqueeze(1)
        return self.linear(x_norm.reshape(B, -1)) + x_last

def cosine_dist(pred, target):
    return (1.0 - nn.functional.cosine_similarity(pred, target, dim=-1)).cpu().numpy()

encoder = load_encoder(DEVICE)
data = np.load(DATASET, allow_pickle=True)
pts = data['points']; actions = data['action_labels'].astype(np.int32)
subjects = data['subject_ids'].astype(np.int32); frame_ids = data['frame_indices'].astype(np.int32)
fall_id = list(ACTION_NAMES).index('fall')

nlin_aucs, const_aucs, dens_aucs = [], [], []

for ts_id in range(7):
    t0 = time.time()
    train_s = [s for s in range(7) if s != ts_id]
    print(f"S{ts_id}: ", end="", flush=True)

    # ── Extract S_t per fold ──
    tr_mask = np.isin(subjects, train_s) & (actions != fall_id)
    tr_idx = np.where(tr_mask)[0]
    # Subsample to keep memory manageable: max 1500 frames per subject per action
    idx_list = []
    for sid in train_s:
        for aid in set(actions):
            if aid == fall_id: continue
            idx = np.where((subjects == sid) & (actions == aid))[0]
            if len(idx) > 1500: idx = np.random.RandomState(42).choice(idx, 1500, replace=False)
            idx_list.extend(idx)
    tr_idx = np.array(idx_list)
    tr_frames = [pts[i].astype(np.float32) for i in tr_idx]
    tr_ts = [1753891200000 + int(subjects[i])*86400000 + int(frame_ids[i])*100 for i in tr_idx]
    print(f"extract({len(tr_frames)}fr)...", end="", flush=True)
    S_tr_full, ts_tr_full = extract_S_sequence(encoder, tr_frames, DEVICE, stride=8, timestamps_ms=tr_ts)

    # Get aligned subject/action arrays for S_t
    S_tr_subj = np.array([subjects[tr_idx[i]] for i in range(0, len(tr_idx), 8)])[:len(S_tr_full)]
    S_tr_act = np.array([actions[tr_idx[i]] for i in range(0, len(tr_idx), 8)])[:len(S_tr_full)]
    S_tr_fid = np.array([frame_ids[tr_idx[i]] for i in range(0, len(tr_idx), 8)])[:len(S_tr_full)]

    # ── GMM ──
    lb = LatentBaseline(); lb.fit(S_tr_full, ts_tr_full[:len(S_tr_full)])

    tn_idx = np.where((subjects == ts_id) & (actions != fall_id))[0]
    if len(tn_idx) > 5000: tn_idx = np.random.RandomState(42).choice(tn_idx, 5000, replace=False)
    tf_idx = np.where((subjects == ts_id) & (actions == fall_id))[0]

    tn_frames = [pts[i].astype(np.float32) for i in tn_idx]; tn_ts_list = [1753891200000+100*86400000+int(frame_ids[i])*100 for i in tn_idx]
    tf_frames = [pts[i].astype(np.float32) for i in tf_idx]; tf_ts_list = [1753891200000+200*86400000+int(frame_ids[i])*100 for i in tf_idx]
    S_n, ts_n = extract_S_sequence(encoder, tn_frames, DEVICE, stride=8, timestamps_ms=tn_ts_list)
    S_f, ts_f = extract_S_sequence(encoder, tf_frames, DEVICE, stride=8, timestamps_ms=tf_ts_list)

    ns = [lb.score(S_n[i], ts_n[i]) for i in range(len(S_n))]
    fs = [lb.score(S_f[i], ts_f[i]) for i in range(len(S_f))]
    auc_d = roc_auc_score([0]*len(ns)+[1]*len(fs), ns+fs)
    dens_aucs.append(auc_d)
    print(f"GMM={auc_d:.4f} ", end="", flush=True)

    # ── Constant Predictor ──
    errs_n, errs_f = [], []
    for i in range(1, len(S_n)):
        errs_n.append(float(cosine_dist(torch.tensor(S_n[i-1]).unsqueeze(0), torch.tensor(S_n[i]).unsqueeze(0))[0]))
    for i in range(1, len(S_f)):
        errs_f.append(float(cosine_dist(torch.tensor(S_f[i-1]).unsqueeze(0), torch.tensor(S_f[i]).unsqueeze(0))[0]))
    auc_c = roc_auc_score([0]*len(errs_n)+[1]*len(errs_f), errs_n+errs_f) if errs_n and errs_f else 0
    const_aucs.append(auc_c)
    print(f"Const={auc_c:.4f} ", end="", flush=True)

    # ── NLinear ──
    ds = DataLoader(list(zip(
        [torch.from_numpy(S_tr_full[i:i+WINDOW].astype(np.float32)) for i in range(len(S_tr_full)-WINDOW)],
        [torch.from_numpy(S_tr_full[i+WINDOW].astype(np.float32)) for i in range(len(S_tr_full)-WINDOW)]
    )), batch_size=256, shuffle=True)

    model = NLinear(seq_len=WINDOW, feat_dim=256)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(20):
        for x, y in ds:
            opt.zero_grad(); loss = nn.functional.mse_loss(model(x), y); loss.backward(); opt.step()
    model.eval()

    errs_n2, errs_f2 = [], []
    with torch.no_grad():
        for i in range(WINDOW, len(S_n)):
            x = torch.from_numpy(S_n[i-WINDOW:i].astype(np.float32)).unsqueeze(0)
            errs_n2.append(float(cosine_dist(model(x), torch.tensor(S_n[i]).unsqueeze(0))[0]))
        for i in range(WINDOW, len(S_f)):
            x = torch.from_numpy(S_f[i-WINDOW:i].astype(np.float32)).unsqueeze(0)
            errs_f2.append(float(cosine_dist(model(x), torch.tensor(S_f[i]).unsqueeze(0))[0]))
    auc_n = roc_auc_score([0]*len(errs_n2)+[1]*len(errs_f2), errs_n2+errs_f2) if errs_n2 and errs_f2 else 0
    nlin_aucs.append(auc_n)
    print(f"NLinear={auc_n:.4f} ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== Results ===")
print(f"Density (GMM):     {np.mean(dens_aucs):.4f} +/- {np.std(dens_aucs):.4f}")
print(f"Constant Pred:     {np.mean(const_aucs):.4f} +/- {np.std(const_aucs):.4f}")
print(f"NLinear Pred:      {np.mean(nlin_aucs):.4f} +/- {np.std(nlin_aucs):.4f}")
for i in range(7):
    print(f"  S{i}: GMM={dens_aucs[i]:.4f} Const={const_aucs[i]:.4f} NLinear={nlin_aucs[i]:.4f}")
