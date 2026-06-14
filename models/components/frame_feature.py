import torch.nn as nn


class FrameFeature(nn.Module):
    def __init__(self, in_ch: int = 48, out_ch: int = 32):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.pw(self.silu(self.dw(x))))
