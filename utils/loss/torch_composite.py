import torch
import torch.nn as nn

from utils.loss.torch_charbonnier import charbonnier_loss
from utils.loss.torch_fft import fft_loss
from utils.loss.torch_fft_binary import fbin_loss


class CompositeLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.w_char = config.get('charbonnier', 1.0)
        self.w_fft = config.get('fft', 0.0)
        self.w_fbin = config.get('fbin', 0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {}
        losses['char'] = charbonnier_loss(pred, target) * self.w_char
        if self.w_fft > 0:
            losses['fft'] = fft_loss(pred, target) * self.w_fft
        else:
            losses['fft'] = torch.tensor(0.0, device=pred.device)
        if self.w_fbin > 0:
            losses['fbin'] = fbin_loss(pred, target) * self.w_fbin
        else:
            losses['fbin'] = torch.tensor(0.0, device=pred.device)
        losses['total'] = sum(losses.values())
        return losses

    def update_weights(self, weights: dict):
        weight_map = {
            'charbonnier': 'w_char',
            'fft': 'w_fft',
            'fbin': 'w_fbin',
        }
        for k, v in weights.items():
            attr = weight_map.get(k)
            if attr:
                setattr(self, attr, v)
