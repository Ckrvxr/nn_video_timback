import torch
import torch.nn as nn

from models.components import (
    DilatedHDCStream,
    MoERouter,
    DownsampleChain,
    SequenceProcessor,
    SpatialStats,
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

        self.downsample = DownsampleChain(1, num_features)

        self.ssm_fwd = SequenceProcessor(num_features, state_dimension)
        self.ssm_bwd = SequenceProcessor(num_features, state_dimension)
        self.ssm_proj = nn.Linear(num_features * 2, num_features)
        self.t_ssm = SequenceProcessor(num_features, state_dimension)
        self.spatial_stats = SpatialStats()

        self.router = MoERouter(num_features + 16, num_experts)
        self.n_active = n_active

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 32]
        self.experts_i = nn.ModuleList(
            [DilatedHDCStream(num_features_stream, dilation_rates) for _ in range(num_experts)])
        self.experts_ct = nn.ModuleList(
            [DilatedHDCStream(num_features_stream, dilation_rates) for _ in range(num_experts)])
        self.experts_cp = nn.ModuleList(
            [DilatedHDCStream(num_features_stream, dilation_rates) for _ in range(num_experts)])

        self.register_buffer('_t_state', torch.zeros(state_dimension))
        self._has_state = False

    def reset_state(self, batch_size: int = 1, device: torch.device | None = None):
        d = self._t_state.shape[-1]
        dev = device or self._t_state.device
        self._t_state = torch.zeros(batch_size, d, device=dev, dtype=self._t_state.dtype)
        self._has_state = True

    def _ensure_state(self, B: int, dev: torch.device):
        if not self._has_state:
            d = self._t_state.shape[-1]
            self._t_state = torch.zeros(B, d, device=dev, dtype=self._t_state.dtype)
            self._has_state = True
        elif self._t_state.shape[0] != B:
            self._t_state = self._t_state[:1].expand(B, -1).contiguous()

    def forward_ssm(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_ssm_ictcp(x)

    def forward_ssm_ictcp(self, ictcp: torch.Tensor) -> torch.Tensor:
        B = ictcp.shape[0]
        dev = ictcp.device
        self._ensure_state(B, dev)

        i_ch = ictcp[:, 0:1]
        feat = self.downsample(i_ch)

        flat = feat.view(B, self.num_features, -1).transpose(1, 2)  # [B, 256, num_features]

        out_fwd, _ = self.ssm_fwd(flat, None)
        z_fwd = out_fwd.mean(dim=1)

        out_bwd, _ = self.ssm_bwd(flat.flip(dims=[1]), None)
        z_bwd = out_bwd.mean(dim=1)

        z = self.ssm_proj(torch.cat([z_fwd, z_bwd], dim=-1))

        out_t, self._t_state = self.t_ssm(z.unsqueeze(1), self._t_state)
        return out_t[:, -1, :]

    def apply_experts(self, ictcp: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply top-k expert deltas, grouped by unique expert (avoids 400-iter loop + GPU→CPU syncs)."""
        delta_acc = torch.zeros_like(ictcp)
        uniq = torch.unique(idx)
        for e in uniq:
            if e == -1:
                continue
            batch_inds, k_inds = torch.where(idx == e)
            w = weights[batch_inds, k_inds].view(-1, 1, 1, 1)
            inp = ictcp[batch_inds]
            delta_i = self.experts_i[e](inp[:, 0:1])
            delta_ct = self.experts_ct[e](inp[:, 1:2])
            delta_cp = self.experts_cp[e](inp[:, 2:3])
            delta = torch.cat([delta_i, delta_ct, delta_cp], dim=1)
            delta_acc[batch_inds] += w * delta
        return ictcp + delta_acc

    def forward(self, x: torch.Tensor, ssm_only: bool = False) -> torch.Tensor | None:
        z_t = self.forward_ssm_ictcp(x)

        if ssm_only:
            return None

        z_spatial = self.spatial_stats(x[:, 0:1])
        router_in = torch.cat([z_t, z_spatial], dim=-1)

        idx, weights, logits = self.router(router_in, x, k=self.n_active, threshold=self.routing_threshold)

        self._last_expert_idx = idx.detach().cpu()
        self._balancing_loss = self.router.load_balancing_loss(logits)

        cleaned_ictcp = self.apply_experts(x, idx, weights)
        return cleaned_ictcp
