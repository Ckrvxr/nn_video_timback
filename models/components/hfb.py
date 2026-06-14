import torch.nn as nn


class HFB(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return x + self.pw(self.silu(self.dw(x)))
