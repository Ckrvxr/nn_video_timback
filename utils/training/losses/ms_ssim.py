import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel[None, None, :, :]


class MSSSIMLoss(nn.Module):
    def __init__(self, n_channels: int = 3, window_size: int = 11, sigma: float = 1.5,
                 n_scales: int = 5, K1: float = 0.01, K2: float = 0.03):
        super().__init__()
        self.n_scales = n_scales
        self.C1 = (K1 * 1.0) ** 2
        self.C2 = (K2 * 1.0) ** 2
        kernel = _gaussian_kernel(window_size, sigma, 'cpu', torch.float32)
        self.register_buffer('_kernel', kernel)
        self._n_channels = n_channels

    def _ssim_per_scale(self, x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        k = kernel.repeat(C, 1, 1, 1)
        pad = k.shape[-1] // 2

        mu_x = F.conv2d(x, k, padding=pad, groups=C)
        mu_y = F.conv2d(y, k, padding=pad, groups=C)
        mu_xx = F.conv2d(x * x, k, padding=pad, groups=C)
        mu_yy = F.conv2d(y * y, k, padding=pad, groups=C)
        mu_xy = F.conv2d(x * y, k, padding=pad, groups=C)

        sigma_x = torch.clamp(mu_xx - mu_x ** 2, min=0)
        sigma_y = torch.clamp(mu_yy - mu_y ** 2, min=0)
        sigma_xy = mu_xy - mu_x * mu_y

        luminance = (2 * mu_x * mu_y + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)
        cs = (2 * sigma_xy + self.C2) / (sigma_x + sigma_y + self.C2)

        ssim_map = luminance * cs
        return ssim_map.mean(dim=(1, 2, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pool = F.avg_pool2d
        kernel = self._kernel.to(pred.device, pred.dtype)

        msssim = 1.0
        cur_pred, cur_target = pred, target
        max_scales = min(self.n_scales,
                         int(torch.log2(torch.tensor(pred.shape[-2], dtype=torch.float32))),
                         int(torch.log2(torch.tensor(pred.shape[-1], dtype=torch.float32))))

        for _ in range(max_scales):
            ssim = self._ssim_per_scale(cur_pred, cur_target, kernel)
            msssim = msssim * ssim
            cur_pred = pool(cur_pred, 2, stride=2)
            cur_target = pool(cur_target, 2, stride=2)

        loss = (1.0 - msssim).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss
