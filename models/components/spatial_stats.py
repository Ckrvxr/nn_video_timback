import torch
import torch.nn as nn


class SpatialStats(nn.Module):
    """Lightweight conv tower extracting multi-scale spatial features for MoE routing."""
    def __init__(self, in_ch: int = 1, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)
