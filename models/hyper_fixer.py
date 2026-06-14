import torch
import torch.nn as nn

from models.components.frame_feature import FrameFeature
from models.components.hfb import HFB
from models.components.sca import SCA
from models.components.hyper_block import HyperBlock


class HyperFixer(nn.Module):
    def __init__(
        self,
        n_features: int = 48,
        n_blocks: int = 2,
        latent: int = 512,
    ):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.frame_feature = FrameFeature(in_ch=48, out_ch=32)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(96, n_features, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.hyper = HyperBlock(n_features, latent)
        self.hfbs = nn.Sequential(*[HFB(n_features) for _ in range(n_blocks)])
        self.sca = SCA(n_features)
        self.global_residual = nn.Conv2d(48, n_features, 1, bias=False)
        self.recon = nn.Sequential(
            nn.Conv2d(n_features, 48, 3, padding=1, bias=False),
            nn.PixelShuffle(4),
        )

    def forward(self, f_prev: torch.Tensor, f_cur: torch.Tensor, f_next: torch.Tensor, scale: int = None) -> torch.Tensor:
        pu_prev = self.pixel_unshuffle(f_prev)
        pu_cur = self.pixel_unshuffle(f_cur)
        pu_next = self.pixel_unshuffle(f_next)

        f_prev = self.frame_feature(pu_prev)
        f_cur = self.frame_feature(pu_cur)
        f_next = self.frame_feature(pu_next)

        return self._backbone(f_prev, f_cur, f_next, pu_cur)

    def forward_cached(
        self,
        f_prev: torch.Tensor,
        f_cur: torch.Tensor,
        f_next: torch.Tensor,
        skip_raw: torch.Tensor,
    ) -> torch.Tensor:
        return self._backbone(f_prev, f_cur, f_next, skip_raw)

    def _backbone(self, f_prev, f_cur, f_next, skip_raw):
        x = self.temporal_fusion(torch.cat([f_prev, f_cur, f_next], dim=1))
        x = self.hyper(x)
        x = self.hfbs(x)
        x = self.sca(x)
        x = x + self.global_residual(skip_raw)
        return self.recon(x)


class FeatureRingCache:
    def __init__(self, max_frames: int = 3, device: torch.device = None):
        self.max_frames = max_frames
        self.features = []
        self.raw_skip = None
        self.device = device

    def push(self, feat: torch.Tensor, raw_skip: torch.Tensor):
        self.features.append(feat)
        self.raw_skip = raw_skip
        if len(self.features) > self.max_frames:
            self.features.pop(0)

    @property
    def is_warm(self) -> bool:
        return len(self.features) == self.max_frames

    def get_triple(self):
        return self.features[-3], self.features[-2], self.features[-1]
