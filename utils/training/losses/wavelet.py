import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletLoss(nn.Module):
    def __init__(self, n_levels: int = 4, weights: list[float] | None = None):
        super().__init__()
        self.n_levels = n_levels
        if weights is None:
            weights = [1.0 / (2 ** i) for i in range(n_levels)]
        self.register_buffer('_weights', torch.tensor(weights))

    def _haar_dwt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        H2, W2 = H // 2 * 2, W // 2 * 2
        x = x[:, :, :H2, :W2]
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 2
        lh = (a - b + c - d) / 2
        hl = (a + b - c - d) / 2
        hh = (a - b - c + d) / 2
        return ll, lh, hl, hh

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        cur_pred, cur_target = pred, target
        max_levels = min(self.n_levels,
                         int(torch.log2(torch.tensor(pred.shape[-2], dtype=torch.float32))),
                         int(torch.log2(torch.tensor(pred.shape[-1], dtype=torch.float32))))
        for i in range(max_levels):
            ll_p, lh_p, hl_p, hh_p = self._haar_dwt(cur_pred)
            ll_t, lh_t, hl_t, hh_t = self._haar_dwt(cur_target)
            loss_hf = (F.l1_loss(lh_p, lh_t) + F.l1_loss(hl_p, hl_t) + F.l1_loss(hh_p, hh_t)) / 3.0
            total = total + loss_hf * self._weights[i]
            cur_pred, cur_target = ll_p, ll_t
        return total / self._weights[:max_levels].sum()
