import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class FlowNet(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 7, padding=3),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 5, padding=2),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.PReLU(hidden),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlowGuidedDeformAlignment(nn.Module):
    def __init__(self, n_features: int = 64):
        super().__init__()
        self.flow_net = FlowNet(n_features * 2, n_features)
        self.offset_mask_conv = nn.Sequential(
            nn.Conv2d(n_features + 2, n_features, 3, padding=1),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, n_features, 3, padding=1),
            nn.PReLU(n_features),
            nn.Conv2d(n_features, 27, 3, padding=1),  # 18 offset + 9 mask
        )
        self.dcn_weight = nn.Parameter(torch.randn(n_features, n_features, 3, 3) * 0.1)

    def forward(
        self, feat_prev: torch.Tensor, feat_cur: torch.Tensor, feat_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_prev = self.flow_net(torch.cat([feat_cur, feat_prev], dim=1))
        flow_next = self.flow_net(torch.cat([feat_cur, feat_next], dim=1))

        aligned_prev = self._deform_align(feat_prev, flow_prev)
        aligned_next = self._deform_align(feat_next, flow_next)

        return aligned_prev, aligned_next

    def _deform_align(self, feat_src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        offset_mask = self.offset_mask_conv(torch.cat([feat_src, flow], dim=1))
        offset = offset_mask[:, :18, :, :]
        mask = torch.sigmoid(offset_mask[:, 18:, :, :])

        aligned = ops.deform_conv2d(
            feat_src,
            offset=offset,
            weight=self.dcn_weight,
            mask=mask,
            stride=1,
            padding=1,
        )
        return aligned
