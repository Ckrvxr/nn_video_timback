import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // 2, channels // 2, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.conv3 = nn.Conv2d(channels // 2, channels, 1, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x + identity
