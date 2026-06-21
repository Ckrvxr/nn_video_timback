from tests.helpers import mock_triton
mock_triton()

import torch
from utils.training.losses.fft import FFTLoss


def test_fft_loss_correctness():
    criterion = FFTLoss()
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss = criterion(pred, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()

    pred_perfect = target.clone().requires_grad_(True)
    loss_perfect = criterion(pred_perfect, target)
    assert torch.isclose(loss_perfect, torch.tensor(0.0), atol=1e-6)


def test_fft_loss_preserves_dtype():
    criterion = FFTLoss()
    pred = torch.randn(2, 3, 16, 16, dtype=torch.float16)
    target = torch.randn(2, 3, 16, 16, dtype=torch.float16)
    loss = criterion(pred, target)
    assert loss.dtype == torch.float16


def test_fft_loss_non_square():
    criterion = FFTLoss()
    pred = torch.randn(1, 3, 16, 32)
    target = torch.randn(1, 3, 16, 32)
    loss = criterion(pred, target)
    assert loss.item() >= 0.0


def test_fft_loss_minimum_size():
    criterion = FFTLoss()
    pred = torch.randn(1, 1, 2, 2)
    target = torch.randn(1, 1, 2, 2)
    loss = criterion(pred, target)
    assert loss.item() >= 0.0
