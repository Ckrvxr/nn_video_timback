import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _flow_warp(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij',
        )
        grid_x = grid_x.float() + flow[:, 0:1, :, :]
        grid_y = grid_y.float() + flow[:, 1:2, :, :]
        grid_x = 2 * grid_x / (w - 1) - 1
        grid_y = 2 * grid_y / (h - 1) - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).squeeze(1)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(
        self,
        pred_cur: torch.Tensor,
        pred_next: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        warp_next = self._flow_warp(pred_next, flow)
        diff = pred_cur - warp_next

        grad_x = F.conv2d(diff, self.sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_y = F.conv2d(diff, self.sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

        edge_mask = torch.exp(-10 * (grad_x ** 2 + grad_y ** 2).mean(dim=1, keepdim=True))
        loss = torch.mean(torch.abs(diff) * edge_mask.detach())
        return loss * self.weight
