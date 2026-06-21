import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components import (
    ParallelExperts,
    MoERouter,
    DownsampleChain,
    SequenceProcessor,
    yuv_to_ictcp,
    ictcp_to_yuv,
)


class MambaFixer(nn.Module):
    def __init__(self, num_features: int = 64, state_dimension: int = 32,
                 num_features_stream: int = 2, num_experts: int = 100,
                 n_active: int = 4,
                 dilation_rates: list[int] | None = None,
                 routing_threshold: float = 1.0):
        super().__init__()
        self.routing_threshold = routing_threshold
        self.num_features = num_features
        self.n_experts = num_experts
        self.n_active = n_active

        self.downsample = DownsampleChain(1, num_features)

        self.t_ssm = SequenceProcessor(num_features * 2, state_dimension)
        self.z_out_proj = nn.Linear(num_features * 2, num_features)

        # Fusion: c2(8) + c3(16) + c4(32) + z_out(num_features) + h_t(state_dim)
        fusion_in = 8 + 16 + 32 + num_features + state_dimension
        fusion_hidden = min(128, fusion_in)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(inplace=True),
        )

        # Router: fusion_hidden + 6 stats → hidden → n_experts
        self.router = MoERouter(fusion_hidden, num_experts)

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        self.experts = ParallelExperts(num_experts, num_features_stream, dilation_rates, in_ch=3)

        self.register_buffer('_t_state', torch.zeros(state_dimension))
        self.register_buffer('prev_z_c5', torch.zeros(num_features))
        self._has_state = False



    def reset_state(self, batch_size: int = 1, device: torch.device | None = None):
        d = self._t_state.shape[-1]
        dev = device or self._t_state.device
        self._t_state = torch.zeros(batch_size, d, device=dev, dtype=self._t_state.dtype)
        self.prev_z_c5 = torch.zeros(batch_size, self.num_features, device=dev, dtype=self._t_state.dtype)
        self._has_state = True

    def _ensure_state(self, B: int, dev: torch.device):
        if not self._has_state:
            d = self._t_state.shape[-1]
            self._t_state = torch.zeros(B, d, device=dev, dtype=self._t_state.dtype)
            self.prev_z_c5 = torch.zeros(B, self.num_features, device=dev, dtype=self._t_state.dtype)
            self._has_state = True
        elif self._t_state.shape[0] != B:
            self._t_state = self._t_state[:1].expand(B, -1).contiguous()
            self.prev_z_c5 = self.prev_z_c5[:1].expand(B, -1).contiguous()

    def forward_ssm_ictcp(self, ictcp: torch.Tensor):
        B = ictcp.shape[0]
        dev = ictcp.device
        self._ensure_state(B, dev)

        i_ch = ictcp[:, 0:1]
        c2, c3, c4, c5 = self.downsample(i_ch)

        z_c2 = c2.mean(dim=(2, 3))
        z_c3 = c3.mean(dim=(2, 3))
        z_c4 = c4.mean(dim=(2, 3))
        z_c5 = c5.mean(dim=(2, 3))

        diff = z_c5 - self.prev_z_c5.detach()
        ssm_in = torch.cat([z_c5, diff], dim=-1).unsqueeze(1)
        out_t, self._t_state = self.t_ssm(ssm_in, self._t_state)
        z_out = self.z_out_proj(out_t[:, -1, :])

        self.prev_z_c5 = z_out.detach()
        return z_c2, z_c3, z_c4, z_out, self._t_state

    def forward(self, x: torch.Tensor):
        z_c2, z_c3, z_c4, z_out, h_t = self.forward_ssm_ictcp(x)

        router_in = torch.cat([z_c2, z_c3, z_c4, z_out, h_t], dim=-1)
        router_in = self.fusion(router_in)

        idx, weights, logits = self.router(router_in, x, k=self.n_active, threshold=self.routing_threshold)

        self._last_expert_idx = idx.detach().cpu()
        self._balancing_loss = self.router.load_balancing_loss(logits)

        delta = self.experts(x, idx, weights)

        return x + delta
