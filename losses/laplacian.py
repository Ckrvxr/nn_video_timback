import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramidLoss(nn.Module):
    def __init__(self, n_levels: int = 5, weights: list[float] | None = None):
        super().__init__()
        self.n_levels = n_levels
        if weights is None:
            weights = [1.0 / (2 ** i) for i in range(n_levels)]
        self.register_buffer('_weights', torch.tensor(weights))

    def _gauss_decimate(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2, 2)

    def _gauss_expand(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        cur_pred, cur_target = pred, target
        for i in range(self.n_levels):
            pred_dec = self._gauss_decimate(cur_pred)
            target_dec = self._gauss_decimate(cur_target)
            pred_detail = cur_pred - self._gauss_expand(pred_dec, cur_pred.shape[-2:])
            target_detail = cur_target - self._gauss_expand(target_dec, cur_target.shape[-2:])
            total = total + F.l1_loss(pred_detail, target_detail) * self._weights[i]
            cur_pred, cur_target = pred_dec, target_dec
        return total / self._weights.sum()
