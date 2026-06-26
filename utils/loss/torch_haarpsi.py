import torch
import torch.nn.functional as F


def haar_decomp(x: torch.Tensor):
    """2D Haar wavelet decomposition on NCHW input [B, C, H, W]. Returns LL, LH, HL, HH."""
    H2, W2 = x.shape[2] // 2, x.shape[3] // 2
    tl = x[:, :, 0:H2 * 2:2, 0:W2 * 2:2]
    tr = x[:, :, 0:H2 * 2:2, 1:W2 * 2:2]
    bl = x[:, :, 1:H2 * 2:2, 0:W2 * 2:2]
    br = x[:, :, 1:H2 * 2:2, 1:W2 * 2:2]
    LL = (tl + tr + bl + br) / 4
    LH = (tl - tr + bl - br) / 4
    HL = (tl + tr - bl - br) / 4
    HH = (tl - tr - bl + br) / 4
    return LL, LH, HL, HH


def _luminance(x: torch.Tensor) -> torch.Tensor:
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def haarpsi_loss(pred: torch.Tensor, target: torch.Tensor,
                 n_scales: int = 3, C: float = 0.001, alpha: float = 4.2) -> torch.Tensor:
    pred_lum = _luminance(pred)
    target_lum = _luminance(target)

    total_weight = 0.0
    total_score = 0.0
    cur_pred, cur_target = pred_lum, target_lum

    for _ in range(n_scales):
        cur_pred, LH_pred, HL_pred, _ = haar_decomp(cur_pred)
        cur_target, LH_target, HL_target, _ = haar_decomp(cur_target)

        for coeff_pred, coeff_target in [(LH_pred, HL_target), (HL_pred, LH_target)]:
            abs_pred = coeff_pred.abs()
            abs_target = coeff_target.abs()

            sim = (2 * abs_pred * abs_target + C) / (abs_pred ** 2 + abs_target ** 2 + C)
            max_abs = torch.maximum(abs_pred, abs_target)
            weight = torch.sigmoid(alpha * max_abs)

            B = sim.shape[0]
            sim_flat = sim.reshape(B, -1)
            weight_flat = weight.reshape(B, -1)
            total_score = total_score + (sim_flat * weight_flat).sum(dim=1)
            total_weight = total_weight + weight_flat.sum(dim=1)

    haarpsi = total_score / (total_weight + 1e-8)
    loss = 1.0 - haarpsi.mean()
    return torch.where(torch.isfinite(loss), loss, torch.tensor(0.0, device=loss.device))
