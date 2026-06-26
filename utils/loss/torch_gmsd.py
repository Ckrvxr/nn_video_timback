import torch
import torch.nn.functional as F


def gmsd_loss(pred: torch.Tensor, target: torch.Tensor,
              C: float = 0.0026, alpha: float = 1.0) -> torch.Tensor:
    """Gradient Magnitude Similarity Deviation loss.

    Args:
        pred, target: [B, C, H, W] NCHW tensors in [0, 1] range.
        C: stabilizer constant (default from paper for [0,1] data).
        alpha: power scaling (default 1.0).

    Returns:
        Scalar loss (higher = more different).
    """
    # Prewitt 3x3 kernels
    kx = torch.tensor([[[[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]]]], dtype=pred.dtype, device=pred.device) / 3.0
    ky = torch.tensor([[[[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]]]], dtype=pred.dtype, device=pred.device) / 3.0

    # Per-channel gradient: [B, C, H, W] → conv with [1, C, 3, 3] → [B, C, H, W]
    kx = kx.expand(pred.shape[1], -1, -1, -1)
    ky = ky.expand(pred.shape[1], -1, -1, -1)

    Gx_pred = F.conv2d(pred, kx, padding=1, groups=pred.shape[1])
    Gy_pred = F.conv2d(pred, ky, padding=1, groups=pred.shape[1])
    Gx_target = F.conv2d(target, kx, padding=1, groups=target.shape[1])
    Gy_target = F.conv2d(target, ky, padding=1, groups=target.shape[1])

    GM_pred = torch.sqrt(Gx_pred ** 2 + Gy_pred ** 2 + 1e-12)
    GM_target = torch.sqrt(Gx_target ** 2 + Gy_target ** 2 + 1e-12)

    sim = (2 * GM_pred * GM_target + C) / (GM_pred ** 2 + GM_target ** 2 + C)

    B, C, H, W = sim.shape
    sim_flat = sim.reshape(B, -1)
    mean_sim = sim_flat.mean(dim=1, keepdim=True)
    std_sim = ((sim_flat - mean_sim) ** 2).mean(dim=1).sqrt()

    gmsd = std_sim.mean()
    return gmsd * alpha
