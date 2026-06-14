import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorHistogramLoss(nn.Module):
    def __init__(self, bins: int = 32):
        super().__init__()
        self.bins = bins

    def _soft_histogram(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > 32 or x.shape[-2] > 32:
            x = F.adaptive_avg_pool2d(x, (32, 32))

        centers = torch.linspace(-1, 1, self.bins + 1, device=x.device, dtype=x.dtype)
        centers = ((centers[:-1] + centers[1:]) / 2).view(1, 1, 1, 1, self.bins)

        sigma = 2.0 / (self.bins - 1)
        x = x.unsqueeze(-1)
        weights = torch.exp(-0.5 * ((x - centers) / sigma) ** 2)

        hist = weights.sum(dim=[2, 3])
        return hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._soft_histogram(pred), self._soft_histogram(target))
