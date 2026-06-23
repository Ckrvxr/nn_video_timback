from tests.helpers import mock_triton, mock_mamba_ssm
mock_triton()
mock_mamba_ssm()

import torch
from components import Timback


def test_timback_forward(small_timback, device):
    x = torch.randn(2, 3, 32, 32, device=device)
    small_timback.reset_state(2, device)
    out = small_timback(x)
    assert out.shape == (2, 3, 32, 32)


def test_timback_adaptive_routing(device):
    model = Timback(
        num_features=16, state_dimension=8, num_features_stream=2,
        num_experts=10, n_active=4, routing_threshold=0.85,
    ).to(device)
    model.eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    model.reset_state(2, device)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)
    last_idx = model._last_expert_idx
    assert last_idx.shape == (2, 4)
    assert ((last_idx >= -1) & (last_idx < 10)).all()
