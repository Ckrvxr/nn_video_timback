import torch
import torch.nn as nn

from losses.charbonnier import CharbonnierLoss
from losses.ms_ssim import MSSSIMLoss
from losses.temporal import TemporalConsistencyLoss
from losses.laplacian import LaplacianPyramidLoss
from losses.color_tight import ColorTightLoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_laplacian = config.get('laplacian', 0.0)
        self.w_color_tight = config.get('color_tight', 0.0)
        self.w_temp = config.get('temporal_consistency', 0.0)
        self.w_msssim = config.get('ms_ssim', 0.0)

        self.char = CharbonnierLoss()
        self.laplacian = LaplacianPyramidLoss() if self.w_laplacian > 0 else None
        self.color_tight = ColorTightLoss() if self.w_color_tight > 0 else None
        self.temporal = TemporalConsistencyLoss(self.w_temp) if self.w_temp > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_prev: torch.Tensor = None,
        target_prev: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = self.char(pred, target) * self.w_char
        if self.laplacian is not None:
            losses['laplacian'] = self.laplacian(pred, target) * self.w_laplacian
        if self.color_tight is not None:
            losses['color_tight'] = self.color_tight(pred, target) * self.w_color_tight
        if self.temporal is not None and pred_prev is not None and target_prev is not None:
            losses['temporal_consistency'] = self.temporal(pred, pred_prev, target, target_prev) * self.w_temp
        losses['total'] = sum(losses.values())
        return losses
