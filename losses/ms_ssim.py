import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) * 0.5


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def create_window(size: int, sigma: float, channel: int) -> torch.Tensor:
    ker = gaussian_kernel(size, sigma)
    ker = ker[:, None] * ker[None, :]
    ker = ker.expand(channel, 1, size, size).contiguous()
    return ker


class MSSSIMLoss(nn.Module):
    def __init__(self, size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.size = size
        self.sigma = sigma
        self.register_buffer('window', create_window(size, sigma, 3))

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        window = self.window.to(x.device, dtype=x.dtype)
        mu_x = F.conv2d(x, window, groups=3, padding=self.size // 2)
        mu_y = F.conv2d(y, window, groups=3, padding=self.size // 2)
        sigma_x = F.conv2d(x * x, window, groups=3, padding=self.size // 2) - mu_x ** 2
        sigma_y = F.conv2d(y * y, window, groups=3, padding=self.size // 2) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, window, groups=3, padding=self.size // 2) - mu_x * mu_y
        cs = (2 * sigma_xy + C2) / (sigma_x + sigma_y + C2)
        ssim = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1) * cs
        return ssim.mean(), cs.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x, y = _to_01(pred), _to_01(target)
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        cs = []
        for i in range(4):
            _, c = self._ssim(x, y)
            cs.append(c)
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
        ssim, _ = self._ssim(x, y)
        cs = torch.stack(cs, dim=0)
        return 1 - ssim * (cs ** torch.tensor(weights[:4], device=cs.device)).prod(dim=0)
