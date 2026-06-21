import torch
import torch.nn as nn

from models.components.color_space import ictcp_to_rgb
from utils.training.losses.charbonnier import CharbonnierLoss
from utils.training.losses.temporal import TemporalConsistencyLoss
from utils.training.losses.sobel import SobelLoss
from utils.training.losses.fft import FFTLoss
from utils.training.losses.wavelet import WaveletLoss
from utils.training.losses.ms_ssim import MSSSIMLoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict, device: torch.device = None):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_wavelet = config.get('wavelet', 0.0)
        self.w_sobel = config.get('sobel', 0.0)
        self.w_temp = config.get('temporal_consistency', 0.0)
        self.w_fft = config.get('fft', 0.0)
        self.w_rgb = config.get('rgb', 0.0)
        self.w_ms_ssim = config.get('ms_ssim', 0.0)

        self.char = CharbonnierLoss()
        self.wavelet = WaveletLoss()
        self.sobel = SobelLoss()
        self.temporal = TemporalConsistencyLoss(self.w_temp)
        self.fft = FFTLoss()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MSSSIMLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_prev: torch.Tensor = None,
        target_prev: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = (self.char(pred, target) * self.w_char) if self.w_char > 0 else torch.tensor(0.0, device=pred.device)
        if self.w_wavelet > 0:
            losses['wavelet'] = self.wavelet(pred, target) * self.w_wavelet
        else:
            losses['wavelet'] = torch.tensor(0.0, device=pred.device)
        if self.w_sobel > 0:
            losses['sobel'] = self.sobel(pred, target) * self.w_sobel
        else:
            losses['sobel'] = torch.tensor(0.0, device=pred.device)
        if self.w_temp > 0 and pred_prev is not None and target_prev is not None:
            losses['temporal_consistency'] = self.temporal(pred, pred_prev, target, target_prev) * self.w_temp
        elif self.w_temp > 0:
            losses['temporal_consistency'] = torch.tensor(0.0, device=pred.device)
        if self.w_fft > 0:
            losses['fft'] = self.fft(pred, target) * self.w_fft
        else:
            losses['fft'] = torch.tensor(0.0, device=pred.device)
        if self.w_rgb > 0:
            pred_rgb = ictcp_to_rgb(pred)
            target_rgb = ictcp_to_rgb(target)
            losses['rgb'] = self.l1(pred_rgb, target_rgb) * self.w_rgb
        else:
            losses['rgb'] = torch.tensor(0.0, device=pred.device)
        if self.w_ms_ssim > 0:
            losses['ms_ssim'] = self.ms_ssim(pred, target) * self.w_ms_ssim
        else:
            losses['ms_ssim'] = torch.tensor(0.0, device=pred.device)
        losses['total'] = sum(losses.values())
        return losses

    def update_weights(self, weights: dict):
        for k, v in weights.items():
            if k == 'charbonnier':
                self.w_char = v
            elif k == 'wavelet':
                self.w_wavelet = v
            elif k == 'sobel':
                self.w_sobel = v
            elif k == 'temporal_consistency':
                self.w_temp = v
            elif k == 'fft':
                self.w_fft = v
            elif k == 'rgb':
                self.w_rgb = v
            elif k == 'ms_ssim':
                self.w_ms_ssim = v
