"""
Temporal ConvNet — 因果时序卷积，编码帧间动态。

架构: 4 层因果空洞卷积，感受野 1+2+4+8=15 帧 (~1s)
输入: (B, T, 1024)  PointNet++ per-frame features
输出: (B, T, 256)   temporal-aware features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果卷积: 只看过去，不偷看未来"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))  # 只在左侧 pad（因果）
        return F.relu(self.bn(self.conv(x)))


class TemporalConvNet(nn.Module):
    """4 层因果空洞卷积 + 输出投影"""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512,
                 output_dim: int = 256):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        dilations = [1, 2, 4, 8]
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(CausalConv1d(hidden_dim, hidden_dim, 3, dilation=d))

        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, 1),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, input_dim)
        Returns: (B, T, output_dim)
        """
        x = x.permute(0, 2, 1)  # → (B, C, T)
        x = self.input_proj(x)

        for layer in self.layers:
            residual = x
            x = layer(x)
            # 残差连接（对齐维度）
            if x.shape == residual.shape:
                x = x + residual

        x = self.output_proj(x)
        return x.permute(0, 2, 1)  # → (B, T, output_dim)


if __name__ == "__main__":
    model = TemporalConvNet(input_dim=1024, hidden_dim=512, output_dim=256)
    x = torch.randn(4, 32, 1024)  # batch=4, 32 frames, 1024-d per frame
    out = model(x)
    print(f"Output shape: {out.shape}")  # Expected: (4, 32, 256)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
