import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        channels = 64
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels * 4, channels * 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
