import torch
import torch.nn.functional as F
def _gaussian_window(size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    ker = g[:, None] * g[None, :]
    return ker.expand(channels, 1, size, size).contiguous()


_SSIM_WINDOW: torch.Tensor | None = None


def _get_ssim_window(channels: int, device: torch.device) -> torch.Tensor:
    global _SSIM_WINDOW
    if _SSIM_WINDOW is None or _SSIM_WINDOW.device != device or _SSIM_WINDOW.size(0) != channels:
        _SSIM_WINDOW = _gaussian_window(11, 1.5, channels).to(device)
    return _SSIM_WINDOW


def calculate_psnr_batch(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-sample PSNR for a batch of images in [-data_range/2, data_range/2]."""
    mse = (img1 - img2).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-8))


def calculate_ssim_batch(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-sample SSIM for a batch of images in [-data_range/2, data_range/2]."""
    img1, img2 = img1.float(), img2.float()
    # Map to [0, 1] for stability; data_range is folded into the constants.
    x = (img1 + data_range / 2.0) / data_range
    y = (img2 + data_range / 2.0) / data_range

    channels = x.size(1)
    window = _get_ssim_window(channels, x.device)
    pad = window.size(-1) // 2

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy

    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    # Per-sample mean over channels and spatial dims
    return ssim_map.mean(dim=(1, 2, 3))
