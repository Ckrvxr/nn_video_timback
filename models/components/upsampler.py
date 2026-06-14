import torch
import torch.nn as nn


class UpsampleHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.scale = scale
        if scale == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels * scale * scale, 3, padding=1, bias=False)
            self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.scale > 1:
            x = self.shuffle(x)
        return x


class MultiScaleUpsampler(nn.Module):
    def __init__(self, out_channels: int, feat_channels: int, scales: list[int]):
        super().__init__()
        self.heads = nn.ModuleDict({
            str(s): UpsampleHead(feat_channels, out_channels, s) for s in scales
        })

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        return self.heads[str(int(scale))](x)
