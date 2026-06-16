import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, n_features: int = 16, n_experts: int = 42, stat_features: int = 6):
        super().__init__()
        total_in = n_features + stat_features
        hidden_dim = max(16, total_in // 2)
        self.router = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_experts)
        )
        self.n_experts = n_experts

    def forward(self, z_t: torch.Tensor, ictcp: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x_feat = z_t.squeeze(1)
        if ictcp is not None:
            mean = ictcp.mean(dim=(2, 3))  # [B, 3]
            std = ictcp.std(dim=(2, 3))    # [B, 3]
            stats = torch.cat([mean, std], dim=1)  # [B, 6]
        else:
            stats = torch.zeros(x_feat.size(0), 6, device=z_t.device, dtype=z_t.dtype)
            
        x = torch.cat([x_feat, stats], dim=1)  # [B, n_features + 6]
        logits = self.router(x)
        idx = logits.argmax(-1)
        return idx, logits

    def load_balancing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(logits, dim=-1).mean(dim=0)
        target = torch.ones(self.n_experts, device=logits.device) / self.n_experts
        return F.kl_div((weights + 1e-8).log(), target, reduction='batchmean')
