import torch.nn as nn

from core.torch.modules import pixel_unshuffle, pixel_shuffle
from core.torch.unet import NAFUNet


class ArtRT(nn.Module):
    def __init__(self, dim: int = 48, n1: int = 2, n2: int = 6,
                 n3: int = 8, nmid: int = 12):
        super().__init__()
        self.head = nn.Conv2d(192, dim, 1, bias=False)
        self.body = NAFUNet(dim=dim, n1=n1, n2=n2, n3=n3, nmid=nmid)
        self.tail = nn.Conv2d(dim, 192, 1, bias=False)

    def forward(self, x):
        h = pixel_unshuffle(x, 8)
        h = self.head(h)
        h = self.body(h)
        h = self.tail(h)
        h = pixel_shuffle(h, 8)
        return h
