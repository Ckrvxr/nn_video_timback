import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, n_features: int = 16, n_experts: int = 42):
        super().__init__()
        self.router = nn.Linear(n_features, n_experts)
        self.n_experts = n_experts

    def forward(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z_t.squeeze(1)
        logits = self.router(x)
        idx = logits.argmax(-1)
        return idx, logits

    def load_balancing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=-1).mean(dim=0)
        target = torch.ones(self.n_experts, device=logits.device) / self.n_experts
        return F.kl_div((weights + 1e-8).log(), target, reduction='batchmean')
