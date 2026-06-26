import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2


class SS2D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4):
        super().__init__()
        self.scans = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, headdim=d_state)
            for _ in range(4)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        L = H * W
        outs = []
        for i, scan_fn in enumerate(self.scans):
            if i == 0:
                seq = x.reshape(B, D, L).transpose(1, 2)
                out = scan_fn(seq).transpose(1, 2).reshape(B, D, H, W)
            elif i == 1:
                seq = x.flip(-1).flip(-2).reshape(B, D, L).transpose(1, 2)
                out = scan_fn(seq).transpose(1, 2).reshape(B, D, H, W)
                out = out.flip(-1).flip(-2)
            elif i == 2:
                seq = x.transpose(-1, -2).reshape(B, D, L).transpose(1, 2)
                out = scan_fn(seq).transpose(1, 2).reshape(B, D, W, H)
                out = out.transpose(-1, -2)
            else:
                seq = x.transpose(-1, -2).flip(-1).reshape(B, D, L).transpose(1, 2)
                out = scan_fn(seq).transpose(1, 2).reshape(B, D, W, H)
                out = out.flip(-1).transpose(-1, -2)
            outs.append(out)
        return torch.cat(outs, dim=1)


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels, 1, 1), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class VMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4):
        super().__init__()
        C = d_model
        self.norm = nn.LayerNorm(C, eps=1e-5)
        self.expand = nn.Conv2d(C, C * 2, 1, bias=False)
        self.prefilter = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.ss2d = SS2D(C, d_state, d_conv)
        self.cross_merge = nn.Conv2d(C * 4, C, 1, bias=False)
        self.blender = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.project = nn.Conv2d(C, C, 1, bias=False)
        self.layerscale = LayerScale(C, init=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        B, C, H, W = x.shape

        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.expand(x)
        gated = F.silu(x)
        g1, gate = gated.chunk(2, dim=1)

        g1 = self.prefilter(g1)
        g1 = self.ss2d(g1)
        g1 = self.cross_merge(g1)
        g1 = self.blender(g1)
        g1 = g1 * gate
        g1 = self.project(g1)
        g1 = self.layerscale(g1)

        return g1 + shortcut
