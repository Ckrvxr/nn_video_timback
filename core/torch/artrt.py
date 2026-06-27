import torch.nn as nn

from core.torch.modules import pixel_unshuffle, pixel_shuffle
from core.torch.r3_block import R3Block
from core.torch.unet import R3UNet


class ArtRT(nn.Module):
    def __init__(self, dim: int = 64, n1: int = 8, n2: int = 8,
                 n3: int = 8, nmid: int = 8, np: int = 1, ns: int = 1):
        super().__init__()
        self.pre_body = nn.Sequential(*[R3Block(192) for _ in range(np)])
        self.head = nn.Conv2d(192, dim, 1, bias=False)
        self.body = R3UNet(dim=dim, n1=n1, n2=n2, n3=n3, nmid=nmid)
        self.tail = nn.Conv2d(dim, 192, 1, bias=False)
        self.post_body = nn.Sequential(*[R3Block(192) for _ in range(ns)])

    def forward(self, x):
        h = pixel_unshuffle(x, 8)
        h = self.pre_body(h)
        h = self.head(h)
        h = self.body(h)
        h = self.tail(h)
        h = self.post_body(h)
        h = pixel_shuffle(h, 8)
        return h
