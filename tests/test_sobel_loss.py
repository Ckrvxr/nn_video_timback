from tests.helpers import mock_triton
mock_triton()

import pytest
import torch
from utils.training.losses.sobel import SobelLoss


def test_sobel_zero_loss_at_perfect():
    criterion = SobelLoss()
    pred = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, pred)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_sobel_scalar_output():
    criterion = SobelLoss()
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    assert loss.dim() == 0


def test_sobel_gradient_flow():
    criterion = SobelLoss()
    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_sobel_all_channels_affect_loss():
    """Every channel contributes to the sobel loss."""
    criterion = SobelLoss()
    pred = torch.randn(1, 3, 16, 16)
    target = pred.clone()
    # Wiping any single channel (including luma) increases loss
    for ch in range(3):
        target2 = target.clone()
        target2[:, ch] = 0.0
        loss = criterion(pred, target2)
        assert loss.item() > 0.0, f"Channel {ch} should affect loss"


def test_sobel_single_channel_works():
    """Loss works with single-channel input."""
    criterion = SobelLoss()
    x = torch.randn(1, 1, 16, 16)
    loss = criterion(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    loss2 = criterion(x, torch.randn_like(x))
    assert loss2.item() > 0.0


def test_sobel_loss_positive_for_mismatch():
    criterion = SobelLoss()
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    assert loss.item() > 0.0
