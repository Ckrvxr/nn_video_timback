import torch
import torch.nn as nn


class DilatedHDCStream(nn.Module):
    def __init__(self, nf: int = 4, dilations: list[int] | None = None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)

        convs = []
        for i, d in enumerate(dilations):
            cin = 4 if i == 0 else nf
            convs.append(nn.Conv2d(cin, nf, 3, 1, d, dilation=d))
            convs.append(nn.PReLU(nf))
        convs.append(nn.Conv2d(nf, 4, 1))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unshuffle(x)
        for mod in self.convs:
            x = mod(x)
        x = self.shuffle(x)
        return 0.1 * torch.tanh(x)
