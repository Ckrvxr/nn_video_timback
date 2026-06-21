import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        kx = self.sobel_x.to(dtype=pred.dtype, device=pred.device).repeat(C, 1, 1, 1)
        ky = self.sobel_y.to(dtype=pred.dtype, device=pred.device).repeat(C, 1, 1, 1)
        grad_pred = F.conv2d(pred, kx, padding=1, groups=C).abs() + F.conv2d(pred, ky, padding=1, groups=C).abs()
        grad_target = F.conv2d(target, kx, padding=1, groups=C).abs() + F.conv2d(target, ky, padding=1, groups=C).abs()
        return F.l1_loss(grad_pred, grad_target)
