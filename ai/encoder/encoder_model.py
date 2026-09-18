"""
完整 Encoder: PointNet++ Backbone + Temporal ConvNet + 投影头

输入: (B, T, N, 5)  时序点云序列
输出: (B, T, 256)    时序隐状态 S_t

训练: 对比学习 (InfoNCE) + 辅助分类 + 连续性正则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.encoder.pointnet_backbone import PointNet2Backbone
from ai.encoder.tcn import TemporalConvNet


class EncoderModel(nn.Module):
    """世界模型 Encoder: 点云序列 → 隐状态序列"""

    def __init__(self,
                 pointnet_out: int = 1024,
                 tcn_hidden: int = 512,
                 latent_dim: int = 256,
                 n_actions: int = 12):
        super().__init__()
        self.latent_dim = latent_dim

        # 逐帧点云 → 全局特征
        self.backbone = PointNet2Backbone(in_channels=5, out_channels=pointnet_out)

        # 时序聚合
        self.tcn = TemporalConvNet(input_dim=pointnet_out, hidden_dim=tcn_hidden,
                                    output_dim=latent_dim)

        # 对比学习投影头 (SimCLR style)
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128),
        )

        # 辅助分类头（仅训练用）
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_actions),
        )

    def forward_backbone(self, points: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        单帧点云 → 全局特征。用于逐帧编码。

        points: (B, N, 5)  完整点云
        xyz:    (B, N, 3)  坐标部分
        Returns: (B, pointnet_out)
        """
        features = points[:, :, 3:]  # velocity + intensity
        return self.backbone(xyz, features)

    def forward_sequence(self, seq_points: torch.Tensor,
                         seq_xyz: torch.Tensor) -> torch.Tensor:
        """
        点云序列 → 隐状态序列。用于训练和推理。

        seq_points: (B, T, N, 5)
        seq_xyz:    (B, T, N, 3)
        Returns: (B, T, latent_dim)
        """
        B, T, N, _ = seq_points.shape

        # 逐帧编码
        pf = seq_points.view(B * T, N, 5)
        xf = seq_xyz.view(B * T, N, 3)
        frame_feats = self.forward_backbone(pf, xf)  # (B*T, 1024)
        frame_feats = frame_feats.view(B, T, -1)      # (B, T, 1024)

        # 时序聚合
        return self.tcn(frame_feats)  # (B, T, latent_dim)

    def forward(self, seq_points: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        seq_points: (B, T, N, 5)
        Returns: {S, z, logits}
          S: (B, T, latent_dim)  隐状态
          z: (B, T, 128)         对比投影
          logits: (B, T, n_actions)  分类预测（均值池化）
          S_mean: (B, latent_dim)  序列均值（用于分类）
        """
        B, T, N, _ = seq_points.shape
        xyz = seq_points[:, :, :, :3]

        S = self.forward_sequence(seq_points, xyz)  # (B, T, latent_dim)

        # 对比投影
        z = self.projection(S)  # (B, T, 128)

        # 辅助分类（序列均值 → 动作）
        S_mean = S.mean(dim=1)  # (B, latent_dim)
        logits = self.classifier(S_mean)  # (B, n_actions)

        return {
            "S": S,
            "z": z,
            "logits": logits,
            "S_mean": S_mean,
        }


if __name__ == "__main__":
    model = EncoderModel(pointnet_out=1024, tcn_hidden=512, latent_dim=256)

    # 模拟 32 帧序列，每帧 32 点，5 通道
    seq = torch.randn(4, 32, 32, 5)
    out = model(seq)

    print(f"S:      {out['S'].shape}")       # (4, 32, 256)
    print(f"z:      {out['z'].shape}")       # (4, 32, 128)
    print(f"logits: {out['logits'].shape}")  # (4, 12)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
