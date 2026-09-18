"""NLinear predictor baseline — 比 GRU 更简的线性预测器。

NLinear (Zeng et al. 2023): 输入序列减去最后一帧，过线性层，再加回最后一帧。
这是一类在长时预测中经常击败 Transformer 的极简基线。
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class NLinear(nn.Module):
    """NLinear: x_norm = x - x_last, out = W * x_norm + x_last"""
    def __init__(self, seq_len=32, feat_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.linear = nn.Linear(seq_len * feat_dim, feat_dim)

    def forward(self, x):
        # x: (B, seq_len, feat_dim) — 历史 S_t 序列
        B, T, D = x.shape
        x_last = x[:, -1, :]  # (B, D) — 最后一帧
        x_norm = x - x_last.unsqueeze(1)  # 减去最后一帧
        x_flat = x_norm.reshape(B, -1)    # (B, T*D)
        out = self.linear(x_flat) + x_last  # (B, D)
        return out


def eval_nlinear_predictor(encoder, data_loader_fn, device='cpu'):
    """
    训练 NLinear 预测器，返回逐帧预测误差作为异常分数。

    encoder: 已加载的 EncoderModel，用于提取 S_t
    data_loader_fn: callable(subjects) → [(S_seq, labels), ...]
    """
    pass  # 需要接入数据集加载流程


def cosine_distance(pred, target):
    """余弦距离 ∈ [0, 2]"""
    pred_n = torch.nn.functional.normalize(pred, dim=-1)
    target_n = torch.nn.functional.normalize(target, dim=-1)
    return 1.0 - (pred_n * target_n).sum(dim=-1)


class ConstantPredictor:
    """基线：预测下一帧 = 当前帧"""
    def predict(self, S_history):
        return S_history[:, -1, :]  # S_{t+1} ≈ S_t


class LinearPredictor:
    """基线：线性外推 S_{t+1} = S_t + (S_t - S_{t-1})"""
    def predict(self, S_history):
        return S_history[:, -1, :] + (S_history[:, -1, :] - S_history[:, -2, :])
