import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    def __init__(self, n_features: int = 16, n_experts: int = 100, stat_features: int = 6):
        super().__init__()
        total_in = n_features + stat_features
        hidden_dim = max(16, total_in // 2)
        self.router = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_experts)
        )
        self.n_experts = n_experts
        self.temperature = 1.0

    def forward(self, z_t: torch.Tensor, ictcp: torch.Tensor | None = None,
                k: int = 4, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_feat = z_t.squeeze(1)
        if ictcp is not None:
            mean = ictcp.mean(dim=(2, 3))  # [B, 3]
            var = torch.clamp(ictcp.var(dim=(2, 3), unbiased=False), min=0.0)
            std = torch.sqrt(var + 1e-8)    # [B, 3]
            stats = torch.cat([mean, std], dim=1)  # [B, 6]
        else:
            stats = torch.zeros(x_feat.size(0), 6, device=z_t.device, dtype=z_t.dtype)

        x = torch.cat([x_feat, stats], dim=1)  # [B, n_features + 6]
        logits = self.router(x)

        probs = F.softmax(logits / self.temperature, dim=-1)

        if threshold < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

            top_probs = sorted_probs[:, :k]
            top_idx = sorted_idx[:, :k]

            cum_probs = torch.cumsum(top_probs, dim=-1)
            prev_cum_probs = torch.cat([torch.zeros_like(cum_probs[:, :1]), cum_probs[:, :-1]], dim=-1)

            active_mask = (prev_cum_probs < threshold)
            active_mask[:, 0] = True

            idx = torch.where(active_mask, top_idx, torch.tensor(-1, device=logits.device))
            weights = torch.where(active_mask, top_probs, torch.tensor(0.0, device=logits.device, dtype=top_probs.dtype))
        else:
            weights, idx = torch.topk(probs, k, dim=-1)

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        return idx, weights, logits

    def load_balancing_loss(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        weights = probs.mean(dim=0)
        target = torch.ones(self.n_experts, device=logits.device) / self.n_experts
        return (weights - target).pow(2).sum() * self.n_experts
