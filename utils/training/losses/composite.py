import torch
import torch.nn as nn

from utils.color_space import ictcp_to_rgb
from utils.memory import gpu_low
from utils.training.losses.charbonnier import CharbonnierLoss
from utils.training.losses.temporal import TemporalConsistencyLoss
from utils.training.losses.fft import FFTLoss
from utils.training.losses.ms_ssim import MSSSIMLoss
from utils.training.losses.gmsd import GMSDLoss
from utils.training.losses.haarpsi import HaarPSILoss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_temp = config.get('temporal_consistency', 0.0)
        self.w_fft = config.get('fft', 0.0)
        self.w_rgb = config.get('rgb', 0.0)
        self.w_ms_ssim = config.get('ms_ssim', 0.0)
        self.w_gmsd = config.get('gmsd', 0.0)
        self.w_haarpsi = config.get('haarpsi', 0.0)

        self.char = CharbonnierLoss()
        self.temporal = TemporalConsistencyLoss(self.w_temp)
        self.fft = FFTLoss()
        self.l1 = nn.L1Loss()
        self.ms_ssim = MSSSIMLoss()
        self.gmsd = GMSDLoss()
        self.haarpsi = HaarPSILoss()

    def _compute_conditional_loss(self, weight: float, loss_fn, *args, 
                                   condition: bool = True, **kwargs) -> torch.Tensor:
        """Helper to compute loss only if weight > 0 and condition is met."""
        if weight > 0 and condition:
            return loss_fn(*args, **kwargs) * weight
        return torch.tensor(0.0, device=args[0].device)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_prev: torch.Tensor = None,
        target_prev: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}
        losses['char'] = self._compute_conditional_loss(self.w_char, self.char, pred, target)
        
        has_temporal = pred_prev is not None and target_prev is not None
        losses['temporal_consistency'] = self._compute_conditional_loss(
            self.w_temp, self.temporal, pred, pred_prev, target, target_prev, condition=has_temporal
        )
        
        _vram_low = gpu_low(0.15, pred.device)
        losses['fft'] = self._compute_conditional_loss(self.w_fft, self.fft, pred, target, condition=not _vram_low)
        
        if self.w_rgb > 0:
            pred_rgb = ictcp_to_rgb(pred)
            target_rgb = ictcp_to_rgb(target)
            losses['rgb'] = self._compute_conditional_loss(self.w_rgb, self.l1, pred_rgb, target_rgb)
        else:
            losses['rgb'] = torch.tensor(0.0, device=pred.device)
        
        losses['ms_ssim'] = self._compute_conditional_loss(self.w_ms_ssim, self.ms_ssim, pred, target, condition=not _vram_low)
        losses['gmsd'] = self._compute_conditional_loss(self.w_gmsd, self.gmsd, pred, target, condition=not _vram_low)
        losses['haarpsi'] = self._compute_conditional_loss(self.w_haarpsi, self.haarpsi, pred, target, condition=not _vram_low)
        
        losses['total'] = sum(losses.values())
        return losses

    def update_weights(self, weights: dict):
        weight_map = {
            'charbonnier': 'w_char',
            'temporal_consistency': 'w_temp',
            'fft': 'w_fft',
            'rgb': 'w_rgb',
            'ms_ssim': 'w_ms_ssim',
            'gmsd': 'w_gmsd',
            'haarpsi': 'w_haarpsi',
        }
        for k, v in weights.items():
            if k in weight_map:
                setattr(self, weight_map[k], v)
