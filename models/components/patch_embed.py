import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 16, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
