from tests.helpers import mock_triton
mock_triton()

import pytest
import torch
from utils.training.losses.wavelet import WaveletLoss


def test_wavelet_zero_loss_at_perfect():
    criterion = WaveletLoss(n_levels=3)
    pred = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, pred)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_wavelet_scalar_output():
    criterion = WaveletLoss()
    pred = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 3, 32, 32)
    loss = criterion(pred, target)
    assert loss.dim() == 0


def test_wavelet_gradient_flow():
    criterion = WaveletLoss()
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_wavelet_custom_weights():
    weights = [1.0, 0.5, 0.25]
    criterion = WaveletLoss(n_levels=3, weights=weights)
    pred = torch.randn(2, 3, 32, 32)
    target = pred + 0.1
    loss = criterion(pred, target)
    assert loss.item() > 0.0


def test_wavelet_loss_positive_for_mismatch():
    criterion = WaveletLoss(n_levels=3)
    pred = torch.randn(2, 3, 16, 16)
    target = torch.randn(2, 3, 16, 16)
    loss = criterion(pred, target)
    assert loss.item() > 0.0


def test_wavelet_different_sizes():
    criterion = WaveletLoss(n_levels=2)
    for h, w in [(32, 32), (33, 33), (32, 48), (15, 15)]:
        pred = torch.randn(1, 3, h, w)
        target = torch.randn(1, 3, h, w)
        loss = criterion(pred, target)
        assert loss.item() > 0.0


def test_wavelet_default_weights_decay():
    criterion = WaveletLoss(n_levels=4)
    expected = torch.tensor([1.0, 0.5, 0.25, 0.125])
    assert torch.allclose(criterion._weights, expected, atol=1e-6)
