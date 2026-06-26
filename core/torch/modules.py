import torch
import torch.nn as nn
import torch.nn.functional as F


def pixel_unshuffle(x: torch.Tensor, r: int) -> torch.Tensor:
    BC, H, W = x.shape[1], x.shape[2], x.shape[3]
    h2, w2 = H // r, W // r
    x = x.reshape(-1, BC, h2, r, w2, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.reshape(-1, BC * r * r, h2, w2)


def pixel_shuffle(x: torch.Tensor, r: int) -> torch.Tensor:
    B, C, H, W = x.shape
    c2 = C // (r * r)
    x = x.reshape(B, c2, r, r, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.reshape(B, c2, H * r, W * r)


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.ca = nn.Conv2d(3, 3, 1, bias=True)
        self.conv = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        delta_scale = torch.tensor([0.1, 0.1, 0.1])
        self.delta_scale = nn.Parameter(delta_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=(2, 3), keepdim=True)
        ca = torch.sigmoid(self.ca(gap))
        x = x * ca
        x = torch.tanh(self.conv(x))
        return x * self.delta_scale.view(1, -1, 1, 1)


def haar_dwt(x: torch.Tensor):
    """2D Haar DWT on NCHW input [B, C, H, W]. Returns LL, LH, HL, HH."""
    H2, W2 = x.shape[2] // 2, x.shape[3] // 2
    tl = x[:, :, 0:H2 * 2:2, 0:W2 * 2:2]
    tr = x[:, :, 0:H2 * 2:2, 1:W2 * 2:2]
    bl = x[:, :, 1:H2 * 2:2, 0:W2 * 2:2]
    br = x[:, :, 1:H2 * 2:2, 1:W2 * 2:2]
    LL = (tl + tr + bl + br) / 4
    LH = (tl - tr + bl - br) / 4
    HL = (tl + tr - bl - br) / 4
    HH = (tl - tr - bl + br) / 4
    return LL, LH, HL, HH


def haar_iwt(LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
    """2D Haar IWT. Returns reconstructed [B, C, H*2, W*2]."""
    tl = LL + LH + HL + HH
    tr = LL - LH + HL - HH
    bl = LL + LH - HL - HH
    br = LL - LH - HL + HH
    B, C, H2, W2 = LL.shape
    out = torch.empty(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = tl
    out[:, :, 0::2, 1::2] = tr
    out[:, :, 1::2, 0::2] = bl
    out[:, :, 1::2, 1::2] = br
    return out


class DetailPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(6, 6, 5, padding=2, groups=6, bias=False)
        self.c1 = nn.Conv2d(6, 8, 1, bias=False)
        self.c2 = nn.Conv2d(8, 3, 1, bias=False)
        scale = torch.tensor([0.2, 0.04, 0.04])
        self.scale = nn.Parameter(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.c1(h)
        detail = self.c2(h)
        return detail * self.scale.view(1, -1, 1, 1)
