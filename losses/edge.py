import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k_x = self.kernel_x.to(dtype=pred.dtype, device=pred.device)
        k_y = self.kernel_y.to(dtype=pred.dtype, device=pred.device)
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        p_x = F.conv2d(pred_gray, k_x, padding=1)
        p_y = F.conv2d(pred_gray, k_y, padding=1)
        t_x = F.conv2d(target_gray, k_x, padding=1)
        t_y = F.conv2d(target_gray, k_y, padding=1)
        return F.l1_loss(p_x, t_x) + F.l1_loss(p_y, t_y)


class TotalVariationLoss(nn.Module):
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.diff(dim=2).abs().mean() + pred.diff(dim=3).abs().mean()
