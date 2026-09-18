"""
PointNet++ Backbone — 稀疏毫米波点云 → 逐帧全局特征。

参考: Qi et al. "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
修改: 输入 5 通道 (x, y, z, velocity, intensity)，适配 10-50 点稀疏点云。

架构:
  SA1: (N, 5) → (N/4, 128)    [分组半径 0.4m, 采样 16 点]
  SA2: (N/4, 128+3) → (N/16, 256)  [分组半径 0.8m, 采样 32 点]
  SA3: (N/16, 256+3) → (1, 1024)   [全局分组，max pooling]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点之间的欧氏距离平方。 (B, N, 3) × (B, M, 3) → (B, N, M)"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样 (FPS)。 (B, N, 3) → (B, npoint) indices"""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor,
                     new_xyz: torch.Tensor) -> torch.Tensor:
    """球查询。 (B, N, 3) + (B, S, 3) → (B, S, nsample) indices"""
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    # 按距离排序，取最近的 nsample 个
    _, group_idx = sqrdists.sort(dim=-1)        # (B, S, N)
    group_idx = group_idx[:, :, :nsample]       # (B, S, nsample)，nsample > N 时取全部

    # 标记超出半径的点
    dist_sorted = sqrdists.gather(dim=-1, index=group_idx)  # (B, S, nsample)
    mask = dist_sorted > radius ** 2

    # 超出半径的用第一个点（最近的点）填充
    group_first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)  # (B, S, nsample)
    group_idx[mask] = group_first[mask]

    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引收集点。 (B, N, C) + (B, S, K) → (B, S, K, C)"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device) \
        .view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


class SetAbstraction(nn.Module):
    """Set Abstraction 层: 采样 + 分组 + PointNet"""

    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None = None):
        """
        xyz: (B, N, 3) 坐标
        points: (B, N, C) 特征（可以为 None，表示只用坐标）
        Returns: new_xyz (B, npoint, 3), new_points (B, npoint, out_channel)
        """
        B, N, _ = xyz.shape
        S = min(self.npoint, N)

        # FPS 采样
        fps_idx = farthest_point_sample(xyz, S)  # (B, S)
        new_xyz = index_points(xyz, fps_idx.unsqueeze(-1).repeat(1, 1, 1)).squeeze(2)  # (B, S, 3)

        # 球查询分组
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (B, S, nsample)
        grouped_xyz = index_points(xyz, idx)  # (B, S, nsample, 3)

        # 相对坐标
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, 3)

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, S, nsample, C)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, S, nsample, 3+C)
        else:
            new_points = grouped_xyz_norm

        # PointNet MLP
        new_points = new_points.permute(0, 3, 1, 2)  # (B, C+3, S, nsample)
        for i, (conv, bn) in enumerate(zip(self.mlp_convs, self.mlp_bns)):
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, dim=-1)[0]  # (B, out_ch, S)
        new_points = new_points.permute(0, 2, 1)  # (B, S, out_ch)

        return new_xyz, new_points


class PointNet2Backbone(nn.Module):
    """PointNet++ Backbone — 3 层 Set Abstraction → 全局 1024-d 特征"""

    def __init__(self, in_channels: int = 5, out_channels: int = 1024):
        super().__init__()
        self.in_channels = in_channels

        # SA1: 局部精细特征. in_channel = 3(xyz_norm) + (5-3)(velocity,intensity)
        self.sa1 = SetAbstraction(
            npoint=16, radius=0.4, nsample=16,
            in_channel=3 + (in_channels - 3),
            mlp=[64, 64, 128],
        )
        # SA2: 中尺度. in_channel = 3(xyz_norm) + 128(sa1_out)
        self.sa2 = SetAbstraction(
            npoint=8, radius=0.8, nsample=16,  # ≤ SA1 输出的 16 个点
            in_channel=3 + 128,
            mlp=[128, 128, 256],
        )
        # SA3: 全局. in_channel = 3(xyz_norm) + 256(sa2_out)
        self.sa3 = SetAbstraction(
            npoint=1, radius=2.0, nsample=8,   # ≤ SA2 输出的 8 个点
            in_channel=3 + 256,
            mlp=[256, 512, out_channels],
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None = None) -> torch.Tensor:
        """
        xyz: (B, N, 3)  点云坐标
        features: (B, N, D)  附加特征 (velocity, intensity 等)，None 则只用 xyz
        Returns: (B, out_channels) 全局特征
        """
        B, N, _ = xyz.shape

        if features is None:
            feat1 = None
        else:
            feat1 = features  # (B, N, D)

        xyz1, feat1 = self.sa1(xyz, feat1)  # → (B, 16, 3), (B, 16, 128)
        xyz2, feat2 = self.sa2(xyz1, feat1)  # → (B, 8, 3), (B, 8, 256)
        _, feat3 = self.sa3(xyz2, feat2)     # → (B, 1, 3), (B, 1, 1024)

        return feat3.squeeze(1)  # (B, 1024)


# 快速测试
if __name__ == "__main__":
    model = PointNet2Backbone(in_channels=5, out_channels=1024)
    xyz = torch.randn(4, 32, 3)       # batch=4, 32 points, xyz
    feat = torch.randn(4, 32, 2)       # velocity + intensity
    out = model(xyz, feat)
    print(f"Output shape: {out.shape}")  # Expected: (4, 1024)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
