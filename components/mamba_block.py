import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

from .fast_ssm import FastSSM


class SSMBlock(nn.Module):
    def __init__(self, d_model: int = 16, d_state: int = 16):
        super().__init__()
        if HAS_MAMBA:
            self.ssm = MambaSSM(d_model=d_model, d_state=d_state)
        else:
            self.ssm = FastSSM(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        # Always project hidden state to d_state for consistent routing
        self.state_proj = nn.Linear(d_model, d_state)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if HAS_MAMBA:
            out = self.ssm(x)
            new_state = self.state_proj(out[:, -1])
        else:
            if state is None:
                state = x.new_zeros(x.size(0), self.ssm.d_state)
            out, new_state = self.ssm(x, state)
        out = self.norm(out)
        return out, new_state


class SequenceProcessor(nn.Module):
    def __init__(self, d_model: int = 16, d_state: int = 16):
        super().__init__()
        self.ssm = SSMBlock(d_model, d_state)

    def forward(self, x: torch.Tensor, h_state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out, new_state = self.ssm(x, h_state)
        return out, new_state
