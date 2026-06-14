import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.normalized_shape = (channels,)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * dw_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels, bias=False)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1, bias=False),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1, bias=False)

        self.norm2 = LayerNorm2d(channels)
        ffn_channels = channels * ffn_expand
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_channels, 1, bias=False),
            SimpleGate(),
            nn.Conv2d(ffn_channels // 2, channels, 1, bias=False),
        )
        self.drop_out = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.chunk(2, dim=1)[0] * x.chunk(2, dim=1)[1]
        x = x * self.sca(x)
        x = self.conv3(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_out(x)
        x = x + shortcut

        return x
