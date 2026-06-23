import torch
import torch.nn as nn


class HaarPSILoss(nn.Module):
    """Haar Wavelet-Based Perceptual Similarity Index.

    Uses 2D Haar decomposition at 3 scales. Compares horizontal and vertical
    detail coefficients via a weighted similarity metric with logistic
    pooling based on coefficient magnitude.
    """
    def __init__(self, n_scales: int = 3, C: float = 0.001, alpha: float = 4.2):
        super().__init__()
        self.n_scales = n_scales
        self.C = C
        self.alpha = alpha

    def _haar_decomp(self, x: torch.Tensor):
        B, C, H, W = x.shape
        H2, W2 = H // 2, W // 2
        top_left = x[:, :, 0:H2*2:2, 0:W2*2:2]
        top_right = x[:, :, 0:H2*2:2, 1:W2*2:2]
        bottom_left = x[:, :, 1:H2*2:2, 0:W2*2:2]
        bottom_right = x[:, :, 1:H2*2:2, 1:W2*2:2]

        LL = (top_left + top_right + bottom_left + bottom_right) / 4
        LH = (top_left - top_right + bottom_left - bottom_right) / 4
        HL = (top_left + top_right - bottom_left - bottom_right) / 4
        HH = (top_left - top_right - bottom_left + bottom_right) / 4
        return LL, LH, HL, HH

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_lum = pred[:, 0:1]
        target_lum = target[:, 0:1]
        device = pred.device

        total_weight = 0.0
        total_score = 0.0

        cur_pred, cur_target = pred_lum, target_lum
        max_scales = min(self.n_scales,
                         int(torch.log2(torch.tensor(pred.shape[-2], dtype=torch.float32))),
                         int(torch.log2(torch.tensor(pred.shape[-1], dtype=torch.float32))))

        for scale_idx in range(max_scales):
            cur_pred, LH_pred, HL_pred, _ = self._haar_decomp(cur_pred)
            cur_target, LH_target, HL_target, _ = self._haar_decomp(cur_target)

            for coeff_pred, coeff_target in [(LH_pred, LH_target), (HL_pred, HL_target)]:
                abs_pred = coeff_pred.abs()
                abs_target = coeff_target.abs()

                sim = (2 * abs_pred * abs_target + self.C) / (abs_pred ** 2 + abs_target ** 2 + self.C)
                max_abs = torch.maximum(abs_pred, abs_target)
                weight = torch.sigmoid(self.alpha * max_abs)

                B = sim.shape[0]
                sim_flat = sim.view(B, -1)
                weight_flat = weight.view(B, -1)
                total_score = total_score + (sim_flat * weight_flat).sum(dim=1)
                total_weight = total_weight + weight_flat.sum(dim=1)

        haarpsi = total_score / (total_weight + 1e-8)
        loss = (1.0 - haarpsi).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss
