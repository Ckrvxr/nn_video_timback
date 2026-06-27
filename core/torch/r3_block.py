import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Channel RMSNorm — native NCHW, no permute needed."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class LayerScale(nn.Module):
    """Per-channel learnable scaling and bias before residual add."""

    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim, 1, 1), init))
        self.beta = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma + self.beta


class SCA(nn.Module):
    """Simplified Channel Attention (NAFNet). No sigmoid."""

    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class R3Block(nn.Module):
    """RMSNorm → SimpleGate → DWConv3×3 → SCA → LayerScale(γx+β) + shortcut."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.expand = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.spatial = nn.Conv2d(dim * 2, dim * 2, 3, padding=1,
                                 groups=dim * 2, bias=False)
        self.sca = SCA(dim)
        self.post = nn.Conv2d(dim, dim, 1, bias=False)
        self.ls = LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.spatial(x)
        g1, g2 = x.chunk(2, dim=1)
        x = g1 * g2
        x = self.sca(x)
        x = self.post(x)
        x = self.ls(x)
        return x + shortcut
