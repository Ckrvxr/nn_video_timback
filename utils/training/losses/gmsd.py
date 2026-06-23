import torch
import torch.nn as nn
import torch.nn.functional as F


class GMSDLoss(nn.Module):
    """Gradient Magnitude Similarity Deviation.

    GMSD = std(GMS_map). The deviation of local quality — high deviation
    means some regions are much worse than others. Lower is better.
    Uses Scharr gradient filters for accurate edge detection.
    """
    def __init__(self, channels: int = 3, C: float = 0.0026):
        super().__init__()
        self.C = C
        kx = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32) / 16.0
        ky = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32) / 16.0
        self.register_buffer('_kx', kx.view(1, 1, 3, 3))
        self.register_buffer('_ky', ky.view(1, 1, 3, 3))
        self._channels = channels

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kx = self._kx.to(x.device, x.dtype).repeat(C, 1, 1, 1)
        ky = self._ky.to(x.device, x.dtype).repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gm_pred = self._gradient_magnitude(pred)  # [B, C, H, W]
        gm_target = self._gradient_magnitude(target)
        gm_pred = gm_pred.mean(dim=1, keepdim=True)
        gm_target = gm_target.mean(dim=1, keepdim=True)

        gms = (2 * gm_pred * gm_target + self.C) / (gm_pred ** 2 + gm_target ** 2 + self.C)
        B = gms.shape[0]
        gms_flat = gms.view(B, -1)
        loss = gms_flat.std(dim=1, unbiased=False).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss
