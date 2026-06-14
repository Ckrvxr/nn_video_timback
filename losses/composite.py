import torch
import torch.nn as nn

from losses.charbonnier import CharbonnierLoss
from losses.ms_ssim import MSSSIMLoss
from losses.perceptual import PerceptualLoss
from losses.temporal import TemporalConsistencyLoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_ssim = config.get('ms_ssim', 0.1)
        self.w_perc = config.get('perceptual', 0.05)
        self.w_temp = config.get('temporal', 0.1)

        self.char = CharbonnierLoss()
        self.ms_ssim = MSSSIMLoss()
        self.perceptual = PerceptualLoss() if self.w_perc > 0 else None
        self.temporal = TemporalConsistencyLoss(self.w_temp) if self.w_temp > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_prev: torch.Tensor = None,
        pred_next: torch.Tensor = None,
        flow: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        losses['char'] = self.char(pred, target) * self.w_char
        losses['ssim'] = self.ms_ssim(pred, target) * self.w_ssim

        if self.perceptual is not None:
            losses['perc'] = self.perceptual(pred, target) * self.w_perc

        if self.temporal is not None and all(x is not None for x in [pred_prev, pred_next, flow]):
            losses['temp'] = self.temporal(pred_prev, pred_next, flow)

        losses['total'] = sum(losses.values())
        return losses
