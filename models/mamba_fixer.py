import torch
import torch.nn as nn

from models.components import (
    DilatedHDCStream,
    MoERouter,
    DownsampleChain,
    PatchEmbed,
    SequenceProcessor,
    yuv_to_ictcp,
    ictcp_to_yuv,
)


class MambaFixer(nn.Module):
    def __init__(self, num_features: int = 64, state_dimension: int = 32,
                 num_features_stream: int = 2, num_experts: int = 42,
                 dilation_rates: list[int] | None = None):
        super().__init__()
        self.num_features = num_features
        self.n_experts = num_experts

        self.downsample = DownsampleChain()
        self.patch_embed = PatchEmbed(1, num_features, 4)

        self.h_ssm = SequenceProcessor(num_features, state_dimension)
        self.v_ssm = SequenceProcessor(num_features, state_dimension)
        self.t_ssm = SequenceProcessor(num_features, state_dimension)

        self.router = MoERouter(num_features, num_experts)

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
        self._t_state = torch.zeros(batch_size, d, device=dev)
        self._has_state = True

    def _ensure_state(self, B: int, dev: torch.device):
        if not self._has_state:
            d = self._t_state.shape[-1]
            self._t_state = torch.zeros(B, d, device=dev)
            self._has_state = True
        elif self._t_state.shape[0] != B:
            self._t_state = self._t_state[:1].expand(B, -1).contiguous()

    def forward_ssm(self, x: torch.Tensor, pre_ictcp: bool = False) -> torch.Tensor:
        ictcp = x if pre_ictcp else yuv_to_ictcp(x)
        return self.forward_ssm_ictcp(ictcp)

    def forward_ssm_ictcp(self, ictcp: torch.Tensor) -> torch.Tensor:
        B = ictcp.shape[0]
        dev = ictcp.device
        self._ensure_state(B, dev)

        h_state = torch.zeros(B, self._t_state.shape[-1], device=dev)
        v_state = torch.zeros(B, self._t_state.shape[-1], device=dev)

        i_ch = ictcp[:, 0:1]
        ds = self.downsample(i_ch)
        feat = self.patch_embed(ds)

        B_embed, C_embed, H_embed, W_embed = feat.shape
        seq_h = feat.view(B_embed, C_embed, -1).transpose(1, 2)
        seq_v = feat.transpose(2, 3).contiguous().view(B_embed, C_embed, -1).transpose(1, 2)

        z_h, _ = self.h_ssm(seq_h, h_state)
        z_v, _ = self.v_ssm(seq_v, v_state)

        z = z_h + z_v

        z_t, self._t_state = self.t_ssm(z.unsqueeze(1), self._t_state)
        return z_t

    def apply_experts(self, ictcp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # Vectorized expert application
        B = ictcp.shape[0]
        final_output = torch.zeros_like(ictcp)
        for e in range(self.n_experts):
            mask = (idx == e)
            if mask.any():
                batch_indices = mask.nonzero(as_tuple=True)[0]
                inp = ictcp[batch_indices]
                delta_i = self.experts_i[e](inp[:, 0:1])
                delta_ct = self.experts_ct[e](inp[:, 1:2])
                delta_cp = self.experts_cp[e](inp[:, 2:3])
                final_output[batch_indices] = inp + torch.cat([delta_i, delta_ct, delta_cp], dim=1)
        return final_output

    def forward(self, x: torch.Tensor, ssm_only: bool = False, pre_ictcp: bool = False) -> torch.Tensor | None:
        ictcp = x if pre_ictcp else yuv_to_ictcp(x)

        z_t = self.forward_ssm_ictcp(ictcp)

        if ssm_only:
            return None

        idx, logits = self.router(z_t, ictcp)

        self._last_expert_idx = idx.detach().cpu()
        self._balancing_loss = self.router.load_balancing_loss(logits)

        cleaned_ictcp = self.apply_experts(ictcp, idx)

        return ictcp_to_yuv(cleaned_ictcp)
