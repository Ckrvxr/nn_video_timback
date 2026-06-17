import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorTightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_chroma = pred[:, 1:]
        target_chroma = target[:, 1:]

        kx = self.sobel_x.to(dtype=pred_chroma.dtype, device=pred_chroma.device).repeat(2, 1, 1, 1)
        ky = self.sobel_y.to(dtype=pred_chroma.dtype, device=pred_chroma.device).repeat(2, 1, 1, 1)
        grad_pred = F.conv2d(pred_chroma, kx, padding=1, groups=2).abs() + F.conv2d(pred_chroma, ky, padding=1, groups=2).abs()
        grad_target = F.conv2d(target_chroma, kx, padding=1, groups=2).abs() + F.conv2d(target_chroma, ky, padding=1, groups=2).abs()

        return F.l1_loss(grad_pred, grad_target)
