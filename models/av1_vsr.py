import torch
import torch.nn as nn

from models.components.alignment import SimpleWarpAlign
from models.components.fusion import TSAFusion
from models.components.resblock import ResBlock
from models.components.upsampler import MultiScaleUpsampler


class AV1VSR(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_features: int = 64,
        n_blocks: int = 8,
        scales: list[int] = None,
    ):
        super().__init__()
        scales = scales or [1, 2, 3, 4, 5, 6]

        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, n_features, 3, padding=1, bias=False),
            nn.PReLU(n_features),
        )

        self.alignment = SimpleWarpAlign(n_features)
        self.fusion = TSAFusion(n_features)

        backbone = []
        for _ in range(n_blocks):
            backbone.append(ResBlock(n_features))
        self.backbone = nn.Sequential(*backbone)
        self.conv_last = nn.Conv2d(n_features, n_features, 3, padding=1, bias=False)

        self.upsampler = MultiScaleUpsampler(in_channels, n_features, scales)

    def forward(
        self, frame_prev: torch.Tensor, frame_cur: torch.Tensor,
        frame_next: torch.Tensor, scale: int = 4
    ) -> torch.Tensor:
        scale = int(scale)

        f_prev = self.conv_first(frame_prev)
        f_cur = self.conv_first(frame_cur)
        f_next = self.conv_first(frame_next)

        a_prev, a_next = self.alignment(f_prev, f_cur, f_next)
        fused = self.fusion(a_prev, f_cur, a_next)

        deep = self.backbone(fused)
        deep = self.conv_last(deep) + fused

        out = self.upsampler(deep, scale)
        return torch.tanh(out)
