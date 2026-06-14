import torch
import torch.nn as nn


class HyperBlock(nn.Module):
    def __init__(self, channels: int, latent: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, channels * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.net(x)
        scale, shift = p.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1) + 1.0
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift
