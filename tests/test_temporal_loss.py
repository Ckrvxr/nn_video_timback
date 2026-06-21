from tests.helpers import mock_triton
mock_triton()

import torch
from utils.training.losses.temporal import TemporalConsistencyLoss


def test_temporal_zero_loss_when_delta_matches():
    criterion = TemporalConsistencyLoss()
    pred_cur = torch.randn(2, 3, 16, 16)
    pred_prev = torch.randn(2, 3, 16, 16)
    loss = criterion(pred_cur, pred_prev, pred_cur, pred_prev)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_temporal_scalar_output():
    criterion = TemporalConsistencyLoss()
    pred_cur = torch.randn(2, 3, 16, 16)
    pred_prev = torch.randn(2, 3, 16, 16)
    target_cur = torch.randn(2, 3, 16, 16)
    target_prev = torch.randn(2, 3, 16, 16)
    loss = criterion(pred_cur, pred_prev, target_cur, target_prev)
    assert loss.dim() == 0


def test_temporal_positive_loss_for_mismatched_delta():
    criterion = TemporalConsistencyLoss()
    pred_cur = torch.randn(2, 3, 16, 16)
    pred_prev = torch.zeros(2, 3, 16, 16)
    target_cur = torch.ones(2, 3, 16, 16)
    target_prev = torch.zeros(2, 3, 16, 16)
    loss = criterion(pred_cur, pred_prev, target_cur, target_prev)
    assert loss.item() > 0.0


def test_temporal_gradient_flow():
    criterion = TemporalConsistencyLoss()
    pred_cur = torch.randn(2, 3, 16, 16, requires_grad=True)
    pred_prev = torch.randn(2, 3, 16, 16, requires_grad=True)
    target_cur = torch.randn(2, 3, 16, 16)
    target_prev = torch.randn(2, 3, 16, 16)
    loss = criterion(pred_cur, pred_prev, target_cur, target_prev)
    loss.backward()
    for param, name in [(pred_cur, 'pred_cur'), (pred_prev, 'pred_prev')]:
        assert param.grad is not None, f"{name} has no grad"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN grad"
        assert not torch.isinf(param.grad).any(), f"{name} has inf grad"


def test_temporal_weight_scales_loss():
    pred_cur = torch.randn(2, 3, 16, 16)
    pred_prev = torch.randn(2, 3, 16, 16)
    target_cur = torch.randn(2, 3, 16, 16)
    target_prev = torch.randn(2, 3, 16, 16)

    loss_half = TemporalConsistencyLoss(weight=0.5)(pred_cur, pred_prev, target_cur, target_prev)
    loss_double = TemporalConsistencyLoss(weight=2.0)(pred_cur, pred_prev, target_cur, target_prev)
    assert torch.isclose(loss_double, loss_half * 4.0, atol=1e-6)


def test_temporal_zero_weight():
    criterion = TemporalConsistencyLoss(weight=0.0)
    pred_cur = torch.randn(2, 3, 16, 16)
    pred_prev = torch.randn(2, 3, 16, 16)
    target_cur = torch.ones(2, 3, 16, 16)
    target_prev = torch.zeros(2, 3, 16, 16)
    loss = criterion(pred_cur, pred_prev, target_cur, target_prev)
    assert loss.item() == 0.0
