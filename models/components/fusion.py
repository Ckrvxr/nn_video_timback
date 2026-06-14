import torch
import torch.nn as nn
import torch.nn.functional as F


class TSAFusion(nn.Module):
    def __init__(self, n_features: int = 64):
        super().__init__()
        self.conv1x1 = nn.Conv2d(n_features * 3, n_features, 1, bias=False)

        self.temporal_attn = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, 2, 3, padding=1),
            nn.Sigmoid(),
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(n_features, n_features // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(n_features // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_prev: torch.Tensor,
        feat_cur: torch.Tensor,
        feat_next: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.conv1x1(torch.cat([feat_prev, feat_cur, feat_next], dim=1))

        temp_weight = self.temporal_attn(fused)
        fused = fused * temp_weight[:, 0:1, :, :]
        fused = fused + feat_cur * temp_weight[:, 1:2, :, :]

        spatial_weight = self.spatial_attn(fused)
        fused = fused * spatial_weight

        return fused
