"""实验 A：监督方法过拟合验证。

训练 12 类逻辑回归（subjects 0-5 的 S_t），
对比训练集 vs held-out subject 6 的 fall 检测 AUC。
训练集 AUC >> 测试集 AUC → 过拟合确认。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ai.drift.common import load_encoder, extract_S_sequence
from simulator.generators import ACTION_NAMES

DATASET = 'ai/data/public/processed/3dpchm_frames.npz'
DEVICE = torch.device('cpu')

encoder = load_encoder(DEVICE)
data = np.load(DATASET, allow_pickle=True)
pts = data['points']; actions = data['action_labels'].astype(np.int32)
subjects = data['subject_ids'].astype(np.int32); frame_ids = data['frame_indices'].astype(np.int32)
fall_id = list(ACTION_NAMES).index('fall')

# ── 提取训练 subjects (0-5) 的 S_t ──
print("Extracting S_t for train subjects 0-5...", flush=True)
train_idx = []
for sid in range(6):
    for aid in set(actions):
        idx = np.where((subjects == sid) & (actions == aid))[0]
        if len(idx) > 1500:
            idx = np.random.RandomState(42).choice(idx, 1500, replace=False)
        train_idx.extend(idx)
train_idx = np.array(train_idx)
tr_frames = [pts[i].astype(np.float32) for i in train_idx]
tr_ts = [1753891200000 + int(subjects[i])*86400000 + int(frame_ids[i])*100 for i in train_idx]
S_tr, ts_tr = extract_S_sequence(encoder, tr_frames, DEVICE, stride=8, timestamps_ms=tr_ts)
S_tr_act = np.array([actions[train_idx[i]] for i in range(0, len(train_idx), 8)])[:len(S_tr)]
print(f"  Train: {len(S_tr)} S_t", flush=True)

# ── 提取 held-out subject 6 的 S_t ──
print("Extracting S_t for held-out subject 6...", flush=True)
te_idx = np.where(subjects == 6)[0]
te_frames = [pts[i].astype(np.float32) for i in te_idx]
te_ts = [1753891200000 + int(subjects[i])*86400000 + int(frame_ids[i])*100 for i in te_idx]
S_te, ts_te = extract_S_sequence(encoder, te_frames, DEVICE, stride=8, timestamps_ms=te_ts)
S_te_act = np.array([actions[te_idx[i]] for i in range(0, len(te_idx), 8)])[:len(S_te)]
print(f"  Test: {len(S_te)} S_t", flush=True)

# ── 训练 12 类逻辑回归 ──
print("Training logistic regression...", flush=True)
clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(S_tr, S_tr_act)

# ── 训练集 AUC：anomaly score = 1 - P(normal classes) ──
def fall_auc(S, acts):
    probs = clf.predict_proba(S)  # (N, 12)
    normal_mask = np.ones(12, dtype=bool); normal_mask[fall_id] = False
    normal_prob = probs[:, normal_mask].sum(axis=1)
    scores = 1.0 - normal_prob
    labels = (acts == fall_id).astype(int)
    if len(np.unique(labels)) < 2:
        return None
    return roc_auc_score(labels, scores)

tr_auc = fall_auc(S_tr, S_tr_act)
te_auc = fall_auc(S_te, S_te_act)

print(f"\n=== 结果 ===")
print(f"训练集 AUC (subjects 0-5): {tr_auc:.4f}" if tr_auc else "训练集 AUC: 无法计算")
print(f"测试集 AUC (subject 6):    {te_auc:.4f}" if te_auc else "测试集 AUC: 无法计算")
if tr_auc and te_auc:
    print(f"过拟合差距: {tr_auc - te_auc:+.4f}")
    print(f"结论: {'✅ 过拟合确认' if tr_auc - te_auc > 0.15 else '⚠️ 差距不显著'}")
