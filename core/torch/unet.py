import torch
import torch.nn.functional as F
import torch.nn as nn

from core.torch.r3_block import R3Block


class R3UNet(nn.Module):
    """2-level U-Net with R3Block encoder/decoder + skip connections.
       Internal padding handles arbitrary input sizes."""

    def __init__(self, dim: int = 32, n1: int = 2, n2: int = 2,
                 n3: int = 4, nmid: int = 4):
        super().__init__()
        # encoder
        self.enc1 = nn.Sequential(*[R3Block(dim) for _ in range(n1)])
        self.down1 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 4, dim * 2, 1, bias=False),
        )
        self.enc2 = nn.Sequential(*[R3Block(dim * 2) for _ in range(n2)])
        self.down2 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 8, dim * 4, 1, bias=False),
        )
        # middle
        self.mid = nn.Sequential(*[R3Block(dim * 4) for _ in range(nmid)])

        # decoder (N denotes target level, matching encN)
        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, bias=False),
        )
        self.merge2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = nn.Sequential(*[R3Block(dim * 2) for _ in range(n3)])

        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
        )
        self.merge1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = nn.Sequential(*[R3Block(dim) for _ in range(n2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')

        s1 = self.enc1(x)
        h = self.down1(s1)
        s2 = self.enc2(h)
        h = self.down2(s2)

        h = self.mid(h)

        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.merge2(h)
        h = self.dec2(h)

        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        h = self.merge1(h)
        h = self.dec1(h)

        return h[:, :, :H, :W]
