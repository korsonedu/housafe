"""
Encoder 损失函数 — 多任务联合训练。

L_total = L_contrastive + 0.05*L_aux + L_var + 0.1*L_smooth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLoss(nn.Module):
    """Encoder 联合损失"""

    def __init__(self, temperature: float = 0.1, aux_weight: float = 0.05,
                 smooth_weight: float = 0.1, var_weight: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.aux_weight = aux_weight
        self.smooth_weight = smooth_weight
        self.var_weight = var_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z: torch.Tensor, S: torch.Tensor,
                logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        z:      (B, T, 128)  对比投影
        S:      (B, T, 256)  隐状态
        logits: (B, n_actions)  分类预测
        labels: (B,)  动作标签（窗口级，已在 Dataset 中取众数）
        """
        B, T, _ = z.shape

        # ── 1. 对比损失 (InfoNCE) ──
        # 同一序列内相邻帧是 positive pair
        # 不同序列的帧是 negative pairs
        z_flat = z.reshape(B * T, -1)  # (B*T, 128)
        z_flat = F.normalize(z_flat, dim=-1)

        # 相似度矩阵
        sim = torch.matmul(z_flat, z_flat.T) / self.temperature  # (B*T, B*T)

        # 构造正样本对: 每个帧的正样本是同一序列的相邻帧
        # (同一 batch item 的 T 帧之间互为 positive)
        pos_mask = torch.zeros(B * T, B * T, device=z.device)
        for b in range(B):
            start, end = b * T, (b + 1) * T
            pos_mask[start:end, start:end] = 1.0
        # 排除自身
        pos_mask.fill_diagonal_(0.0)

        # InfoNCE
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1) - torch.exp(sim.diagonal())  # 排除自身
        all_sum = all_sum.clamp(min=1e-10)
        pos_ratio = pos_sum / all_sum
        L_contrastive = -torch.log(pos_ratio.clamp(min=1e-10)).mean()

        # ── 2. 辅助分类损失 ──
        L_aux = self.ce(logits, labels)

        # ── 3. 连续性正则 ──
        # 相邻帧的隐状态变化应该平滑
        L_smooth = F.mse_loss(S[:, 1:], S[:, :-1])

        # ── 4. Variance 正则 ──
        # 防止隐状态方差爆炸/坍缩
        var = S.var(dim=(0, 1)).mean()
        L_var = F.mse_loss(var, torch.ones_like(var))  # 目标方差 ≈ 1

        # ── 总损失 ──
        total = (L_contrastive
                 + self.aux_weight * L_aux
                 + self.smooth_weight * L_smooth
                 + self.var_weight * L_var)

        return {
            "total": total,
            "contrastive": L_contrastive,
            "aux": L_aux,
            "smooth": L_smooth,
            "var": L_var,
        }
