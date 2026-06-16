import torch
import torch.nn as nn


class SAMAffine(nn.Module):
    def __init__(self, n_features: int = 16):
        super().__init__()
        self.mapping = nn.Linear(n_features, n_features * 2)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.mapping(z)
        gamma, beta = ab.chunk(2, dim=-1)
        return gamma, beta
