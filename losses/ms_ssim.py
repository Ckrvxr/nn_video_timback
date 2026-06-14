import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, 1) * kernel.view(1, 1, 1, size)


def ssim(
    img1: torch.Tensor, img2: torch.Tensor,
    kernel: torch.Tensor, c1: float, c2: float,
) -> torch.Tensor:
    ch = img1.shape[1]
    kernel = kernel.repeat(ch, 1, 1, 1).to(img1.device)

    mu1 = F.conv2d(img1, kernel, padding=kernel.shape[-1] // 2, groups=ch)
    mu2 = F.conv2d(img2, kernel, padding=kernel.shape[-1] // 2, groups=ch)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel.shape[-1] // 2, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel.shape[-1] // 2, groups=ch) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel.shape[-1] // 2, groups=ch) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map


class MSSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = gaussian_kernel(11, 1.5)
        self.register_buffer('kernel', kernel)
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = (0.01 * 2) ** 2
        c2 = (0.03 * 2) ** 2

        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        msssim = 1.0
        for w in self.weights:
            ssim_map = ssim(pred, target, self.kernel, c1, c2).mean(dim=1)
            msssim = msssim * (ssim_map.mean() ** w)

            if len(self.weights) > 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)

        return 1 - msssim
