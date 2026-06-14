import torch
import torch.nn as nn

from losses.charbonnier import CharbonnierLoss
from losses.ms_ssim import MSSSIMLoss
from losses.temporal import TemporalConsistencyLoss
from losses.edge import SobelEdgeLoss, TotalVariationLoss
from losses.histogram import ColorHistogramLoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_ssim = config.get('ms_ssim', 0.1)
        self.w_temp = config.get('temporal_consistency', 0.0)
        self.w_sobel = config.get('sobel_gradient', 0.0)
        self.w_tv = config.get('total_variation', 0.0)
        self.w_hist = config.get('gaussian_soft_histogram', 0.0)

        self.char = CharbonnierLoss()
        self.ms_ssim = MSSSIMLoss()
        self.temporal = TemporalConsistencyLoss(self.w_temp) if self.w_temp > 0 else None
        self.sobel = SobelEdgeLoss() if self.w_sobel > 0 else None
        self.tv = TotalVariationLoss() if self.w_tv > 0 else None
        self.histogram = ColorHistogramLoss() if self.w_hist > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_next: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = self.char(pred, target) * self.w_char
        losses['ssim'] = self.ms_ssim(pred, target) * self.w_ssim
        if self.sobel is not None:
            losses['sobel_gradient'] = self.sobel(pred, target) * self.w_sobel
        if self.tv is not None:
            losses['total_variation'] = self.tv(pred) * self.w_tv
        if self.temporal is not None and pred_next is not None:
            losses['temporal_consistency'] = self.temporal(pred, pred_next) * self.w_temp
        if self.histogram is not None:
            losses['gaussian_soft_histogram'] = self.histogram(pred, target) * self.w_hist
        losses['total'] = sum(losses.values())
        return losses
