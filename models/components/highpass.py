import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPassFilter(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('kernel', k.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1)
