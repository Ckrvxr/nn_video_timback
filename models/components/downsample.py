import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleChain(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 64):
        super().__init__()
        ch1 = max(4, out_ch // 4)
        ch2 = max(8, out_ch // 2)
        ch3 = max(12, (out_ch * 3) // 4)
        self.conv1 = nn.Conv2d(in_ch, ch1, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(ch3, out_ch, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Center crop to square if not already
        if H != W:
            side = min(H, W)
            crop_h = (H - side) // 2
            crop_w = (W - side) // 2
            x = x[:, :, crop_h:crop_h + side, crop_w:crop_w + side]
        
        # Resize to 256x256 to keep memory usage low and output resolution fixed to 16x16
        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')
        elif x.shape[-1] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        return x
