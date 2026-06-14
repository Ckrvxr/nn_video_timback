import torch.nn as nn


class SCA(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // 8, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // 8, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))
