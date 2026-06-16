import torch
import torch.nn as nn


class DSBlock(nn.Module):
    def __init__(self, nf: int, dilation: int, gamma_beta: bool = False):
        super().__init__()
        self.gamma_beta = gamma_beta
        self.depthwise = nn.Conv2d(nf, nf, 3, 1, dilation, dilation=dilation, groups=nf, bias=False)
        self.pointwise = nn.Conv2d(nf, nf, 1, bias=False)
        self.act = nn.PReLU(nf)

    def forward(self, x, gamma=None, beta=None):
        x = self.pointwise(self.depthwise(x))
        if self.gamma_beta and gamma is not None:
            x = x * gamma[:, :8].view(-1, 8, 1, 1) + beta[:, :8].view(-1, 8, 1, 1)
        return self.act(x)


class ChannelNet(nn.Module):
    def __init__(self, nf: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nf, 3, 1, 1, dilation=1)
        self.act1 = nn.PReLU(nf)

        self.ds2 = DSBlock(nf, 2, gamma_beta=True)
        self.ds3 = DSBlock(nf, 4)
        self.ds4 = DSBlock(nf, 8)
        self.ds5 = DSBlock(nf, 16)
        self.ds6 = DSBlock(nf, 32)
        self.ds7 = DSBlock(nf, 1)

        self.out = nn.Conv2d(nf, 1, 1)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.ds2(x, gamma, beta)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.ds5(x)
        x = self.ds6(x)
        x = self.ds7(x)
        x = self.out(x)
        return 0.1 * torch.tanh(x)
