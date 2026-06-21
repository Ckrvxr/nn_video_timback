import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred_cur: torch.Tensor,
        pred_prev: torch.Tensor,
        target_cur: torch.Tensor,
        target_prev: torch.Tensor,
    ) -> torch.Tensor:
        pred_delta = pred_cur - pred_prev
        target_delta = target_cur - target_prev
        loss = torch.abs(pred_delta - target_delta).mean()
        return loss * self.weight
