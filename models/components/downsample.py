import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleChain(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Center crop to square if not already
        if H != W:
            side = min(H, W)
            crop_h = (H - side) // 2
            crop_w = (W - side) // 2
            x = x[:, :, crop_h:crop_h + side, crop_w:crop_w + side]
        
        # Resize to 256x256 to keep memory usage low and output resolution fixed to 64x64
        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')
        elif x.shape[-1] < 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
