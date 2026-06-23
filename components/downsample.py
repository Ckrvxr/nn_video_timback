import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleChain(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 4, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(32, out_ch, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if H != W:
            side = min(H, W)
            crop_h = (H - side) // 2
            crop_w = (W - side) // 2
            x = x[:, :, crop_h:crop_h + side, crop_w:crop_w + side]

        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')
        elif x.shape[-1] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        x = self.relu1(self.conv1(x))
        c2 = self.relu2(self.conv2(x))
        c3 = self.relu3(self.conv3(c2))
        c4 = self.relu4(self.conv4(c3))
        c5 = self.relu5(self.conv5(c4))
        return c2, c3, c4, c5
