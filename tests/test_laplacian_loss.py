from tests.helpers import mock_triton
mock_triton()

import pytest
import torch
from utils.training.losses.laplacian import LaplacianPyramidLoss


def test_laplacian_zero_loss_at_perfect_match():
    for n_levels in [1, 3, 5]:
        criterion = LaplacianPyramidLoss(n_levels=n_levels)
        x = torch.randn(2, 3, 64, 64)
        loss = criterion(x, x)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), f"failed at n_levels={n_levels}"


def test_laplacian_scalar_output():
    criterion = LaplacianPyramidLoss()
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss = criterion(pred, target)
    assert loss.dim() == 0


def test_laplacian_gradient_flow():
    criterion = LaplacianPyramidLoss()
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_laplacian_custom_weights():
    weights = [0.5, 0.3, 0.2]
    criterion = LaplacianPyramidLoss(n_levels=3, weights=weights)
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    loss = criterion(pred, target)
    assert loss.item() >= 0.0


def test_laplacian_minimum_size_for_n_levels():
    """Input must be at least 2^n_levels on each spatial dim."""
    criterion = LaplacianPyramidLoss(n_levels=3)
    x = torch.randn(1, 3, 8, 8)
    loss = criterion(x, x)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    with pytest.raises(RuntimeError):
        tiny = torch.randn(1, 3, 3, 8)
        criterion(tiny, tiny)


def test_laplacian_positive_loss_for_mismatch():
    criterion = LaplacianPyramidLoss()
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss = criterion(pred, target)
    assert loss.item() > 0.0


def test_laplacian_default_weights_decay():
    criterion = LaplacianPyramidLoss(n_levels=4)
    expected = [1.0, 0.5, 0.25, 0.125]
    actual = criterion._weights.tolist()
    assert actual == pytest.approx(expected)
