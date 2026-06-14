import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleWarpAlign(nn.Module):
    def __init__(self, n_features: int = 64):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features, 3, padding=1, bias=False),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, n_features, 3, padding=1, bias=False),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, 2, 3, padding=1, bias=False),
        )

    def forward(
        self, feat_prev: torch.Tensor, feat_cur: torch.Tensor, feat_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_prev = self.flow_net(torch.cat([feat_cur, feat_prev], dim=1))
        flow_next = self.flow_net(torch.cat([feat_cur, feat_next], dim=1))
        aligned_prev = self._warp(feat_prev, flow_prev)
        aligned_next = self._warp(feat_next, flow_next)
        return aligned_prev, aligned_next

    def _warp(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=feat.device, dtype=torch.float32),
            torch.arange(w, device=feat.device, dtype=torch.float32),
            indexing='ij',
        )
        grid = torch.empty(n, h, w, 2, device=feat.device, dtype=feat.dtype)
        grid[..., 0] = 2.0 * (grid_x + flow[:, 0]) / (w - 1) - 1.0
        grid[..., 1] = 2.0 * (grid_y + flow[:, 1]) / (h - 1) - 1.0
        return F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
