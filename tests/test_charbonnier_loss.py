from tests.helpers import mock_triton
mock_triton()

import torch
from utils.training.losses.charbonnier import CharbonnierLoss


def test_charbonnier_perfect_match_gives_eps():
    """Charbonnier has an epsilon floor: perfect match → sqrt(eps)."""
    criterion = CharbonnierLoss(eps=1e-4)
    pred = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, pred)
    assert torch.isclose(loss, torch.tensor(1e-2), rtol=0.1)


def test_charbonnier_positive_loss_for_mismatch():
    criterion = CharbonnierLoss()
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    assert loss.item() > 0.0


def test_charbonnier_scalar_output():
    criterion = CharbonnierLoss()
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    assert loss.dim() == 0


def test_charbonnier_gradient_flow():
    criterion = CharbonnierLoss()
    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_charbonnier_eps_controls_floor():
    """eps adds a floor: perfect match gives sqrt(eps), mismatch is dominated by |diff|."""
    criterion_small = CharbonnierLoss(eps=1e-8)
    criterion_large = CharbonnierLoss(eps=1.0)
    pred = torch.randn(2, 3, 16, 16)
    target = pred.clone()

    loss_small = criterion_small(pred, target)
    loss_large = criterion_large(pred, target)
    assert torch.isclose(loss_small, torch.tensor(1e-4), atol=0.5e-4)
    assert torch.isclose(loss_large, torch.tensor(1.0), atol=0.1)


def test_charbonnier_stability_large_values():
    criterion = CharbonnierLoss()
    pred = torch.tensor([[[[1e5, -1e5]]]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    loss = criterion(pred, target)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_charbonnier_batch_independence():
    criterion = CharbonnierLoss()
    pred = torch.randn(4, 3, 8, 8)
    target = torch.zeros(4, 3, 8, 8)
    full_loss = criterion(pred, target)
    per_sample = torch.stack([criterion(pred[i:i+1], target[i:i+1]) for i in range(4)])
    assert torch.isclose(full_loss, per_sample.mean(), atol=1e-6)
