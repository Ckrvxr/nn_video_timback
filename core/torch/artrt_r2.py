import torch
import torch.nn as nn

from core.torch.modules import pixel_unshuffle, pixel_shuffle, haar_dwt, haar_iwt
from core.torch.ssm_block import VMambaBlock


class ArtRT(nn.Module):
    def __init__(self, d_model: int = 48, d_state: int = 48,
                 expand: int = 4, chunk_size: int = 512):
        super().__init__()
        self.body = VMambaBlock(d_model, d_state=d_state, expand=expand, chunk_size=chunk_size)
        self.refine = nn.Conv2d(12, 3, 5, padding=2, padding_mode='reflect', bias=False)

    def forward(self, x):
        LL, LH, HL, HH = haar_dwt(x)
        h = pixel_unshuffle(LL, 4)
        h = self.body(h)
        h = pixel_shuffle(h, 4)
        h = self.refine(torch.cat([h, LH, HL, HH], dim=1))
        delta = haar_iwt(h, LH, HL, HH)
        return x + delta
