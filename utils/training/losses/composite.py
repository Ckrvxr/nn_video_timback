import torch
import torch.nn as nn

from utils.training.losses.charbonnier import CharbonnierLoss
from utils.training.losses.temporal import TemporalConsistencyLoss
from utils.training.losses.sobel import SobelLoss
from utils.training.losses.fft import FFTLoss
from utils.training.losses.wavelet import WaveletLoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_wavelet = config.get('wavelet', 0.0)
        self.w_sobel = config.get('sobel', 0.0)
        self.w_temp = config.get('temporal_consistency', 0.0)
        self.w_fft = config.get('fft', 0.0)

        self.char = CharbonnierLoss()
        self.wavelet = WaveletLoss() if self.w_wavelet > 0 else None
        self.sobel = SobelLoss() if self.w_sobel > 0 else None
        self.temporal = TemporalConsistencyLoss(self.w_temp) if self.w_temp > 0 else None
        self.fft = FFTLoss() if self.w_fft > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_prev: torch.Tensor = None,
        target_prev: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = self.char(pred, target) * self.w_char
        if self.wavelet is not None:
            losses['wavelet'] = self.wavelet(pred, target) * self.w_wavelet
        if self.sobel is not None:
            losses['sobel'] = self.sobel(pred, target) * self.w_sobel
        if self.temporal is not None and pred_prev is not None and target_prev is not None:
            losses['temporal_consistency'] = self.temporal(pred, pred_prev, target, target_prev) * self.w_temp
        if self.fft is not None:
            losses['fft'] = self.fft(pred, target) * self.w_fft
        losses['total'] = sum(losses.values())
        return losses
