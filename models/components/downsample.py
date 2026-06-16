import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleChain(nn.Module):
    def __init__(self, target_size: int = 256):
        super().__init__()
        self.target_size = target_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Center crop to square if not already
        if H != W:
            side = min(H, W)
            crop_h = (H - side) // 2
            crop_w = (W - side) // 2
            x = x[:, :, crop_h:crop_h + side, crop_w:crop_w + side]
        
        # Direct downsampling is much more memory efficient than upsampling then pooling
        if x.shape[-1] > 256:
            x = F.interpolate(x, size=(256, 256), mode='area')
        return x
