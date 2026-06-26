import torch
import torch.nn as nn


class SCA(nn.Module):
    """Simplified Channel Attention (NAFNet). No sigmoid."""

    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv2(self.conv1(self.pool(x)))


class NAFBlock(nn.Module):
    """SimpleGate + DWConv3×3 + SCA channel attention."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.expand = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.spatial = nn.Conv2d(dim * 2, dim * 2, 3, padding=1,
                                 groups=dim * 2, bias=False)
        self.sca = SCA(dim)
        self.post = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.expand(x)
        x = self.spatial(x)
        g1, g2 = x.chunk(2, dim=1)
        x = g1 * g2
        x = self.sca(x)
        x = self.post(x)
        return x + shortcut
