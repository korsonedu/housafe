"""
Predictor 模型 — 从隐状态历史预测下一帧。

提供两个版本：
  - CausalStatePredictor: 因果 Transformer（44M，Phase B0 探索版）
  - TinyPredictor: 轻量 GRU（1.3M，训练验证版，H3/H4 通过）
"""

import torch
import torch.nn as nn


class CausalStatePredictor(nn.Module):
    """因果 Transformer — ~8M params（未使用，保留作为对比基线）"""

    def __init__(self, latent_dim=256, num_layers=4, num_heads=8,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 32, latent_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(32, 32) * float('-inf'), diagonal=1))
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, S_history: torch.Tensor) -> torch.Tensor:
        B, T, _ = S_history.shape
        x = self.input_proj(S_history) + self.pos_embed[:, :T, :]
        x = self.transformer(x, x, tgt_mask=self.causal_mask[:T, :T])
        return self.output_proj(x[:, -1, :])


class TinyPredictor(nn.Module):
    """轻量 GRU 预测器 — 1.3M params，H3 改善 47%，H4 AUC 0.94"""

    def __init__(self, in_dim=256, hid=128):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hid)
        self.gru = nn.GRU(hid, hid, 2, batch_first=True)
        self.proj_out = nn.Linear(hid, in_dim)

    def forward(self, x):
        x = self.proj_in(x)
        _, h = self.gru(x)
        return self.proj_out(h[-1])
