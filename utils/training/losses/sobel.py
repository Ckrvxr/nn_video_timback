import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    """Edge-aware L1 loss on Sobel gradient magnitude.

    Computes horizontal/vertical Sobel gradients of both pred and target,
    then L1 on the gradient magnitude maps.  Helps preserve edges.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kernel_x', sobel_x.reshape(1, 1, 3, 3))
        self.register_buffer('kernel_y', sobel_y.reshape(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        px = F.conv2d(pred.reshape(B * C, 1, H, W), self.kernel_x.to(pred), padding=1)
        py = F.conv2d(pred.reshape(B * C, 1, H, W), self.kernel_y.to(pred), padding=1)
        tx = F.conv2d(target.reshape(B * C, 1, H, W), self.kernel_x.to(pred), padding=1)
        ty = F.conv2d(target.reshape(B * C, 1, H, W), self.kernel_y.to(pred), padding=1)

        pmag = torch.sqrt(px * px + py * py + 1e-8).reshape(B, C, H, W)
        tmag = torch.sqrt(tx * tx + ty * ty + 1e-8).reshape(B, C, H, W)

        return F.l1_loss(pmag, tmag)
