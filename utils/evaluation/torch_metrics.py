import torch


def calculate_psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR per sample in batch. Input: NCHW [B, C, H, W]."""
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse + 1e-8)


def calculate_ssim_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0,
                         K1: float = 0.01, K2: float = 0.03) -> torch.Tensor:
    """Simplified SSIM. Input: NCHW [B, C, H, W]."""
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    mu_pred = pred.mean(dim=(2, 3), keepdim=True)
    mu_target = target.mean(dim=(2, 3), keepdim=True)

    sigma_pred = (pred ** 2).mean(dim=(2, 3), keepdim=True) - mu_pred ** 2
    sigma_target = (target ** 2).mean(dim=(2, 3), keepdim=True) - mu_target ** 2
    sigma_pt = (pred * target).mean(dim=(2, 3), keepdim=True) - mu_pred * mu_target

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pt + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    return ssim.mean(dim=(1, 2, 3))
