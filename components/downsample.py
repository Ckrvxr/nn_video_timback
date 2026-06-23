import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleChain(nn.Module):
    def __init__(self, in_ch: int = 1, num_features: int = 64):
        super().__init__()
        self.entry = nn.Conv2d(in_ch, 8, kernel_size=9, stride=2, padding=4, padding_mode='reflect')
        self.relu0 = nn.ReLU(inplace=True)

        self.down2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.skip2 = nn.Conv2d(8, 16, kernel_size=1, stride=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.down3 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.skip3 = nn.Conv2d(16, 24, kernel_size=1, stride=2)
        self.relu3 = nn.ReLU(inplace=True)

        self.down4 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.skip4 = nn.Conv2d(24, 48, kernel_size=1, stride=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.down5 = nn.Conv2d(48, num_features, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.skip5 = nn.Conv2d(48, num_features, kernel_size=1, stride=2)
        self.relu5 = nn.ReLU(inplace=True)

        self.grid_proj = nn.Linear(num_features * 16, num_features)

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

        x = self.relu0(self.entry(x))
        c2 = self.relu2(self.down2(x) + self.skip2(x))
        c3 = self.relu3(self.down3(c2) + self.skip3(c2))
        c4 = self.relu4(self.down4(c3) + self.skip4(c3))
        c5 = self.relu5(self.down5(c4) + self.skip5(c4))

        z_c2 = c2.mean(dim=(2, 3))
        z_c3 = c3.mean(dim=(2, 3))
        z_c4 = c4.mean(dim=(2, 3))
        z_c5 = self.grid_proj(F.adaptive_avg_pool2d(c5, 4).flatten(1))

        return z_c2, z_c3, z_c4, z_c5
