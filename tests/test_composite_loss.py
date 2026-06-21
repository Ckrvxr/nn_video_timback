from tests.helpers import mock_triton
mock_triton()

import pytest
import torch
from utils.training.losses.composite import CompositeLoss


@pytest.fixture
def pred_target():
    B, C, H, W = 2, 3, 32, 32
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randn(B, C, H, W)
    return pred, target


@pytest.fixture
def pred_target_seq():
    B, C, H, W = 2, 3, 32, 32
    pred_cur = torch.randn(B, C, H, W, requires_grad=True)
    pred_prev = torch.randn(B, C, H, W, requires_grad=True)
    target_cur = torch.randn(B, C, H, W)
    target_prev = torch.randn(B, C, H, W)
    return pred_cur, pred_prev, target_cur, target_prev


def test_composite_default_config_uses_only_charbonnier(pred_target):
    pred, target = pred_target
    criterion = CompositeLoss({'charbonnier': 1.0})
    losses = criterion(pred, target)
    assert set(losses.keys()) == {'char', 'total'}
    assert losses['char'].item() > 0.0
    assert losses['total'].item() == losses['char'].item()


def test_composite_dict_keys(pred_target):
    pred, target = pred_target
    criterion = CompositeLoss({
        'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1,
    })
    losses = criterion(pred, target)
    expected = {'char', 'wavelet', 'sobel', 'fft', 'total'}
    assert set(losses.keys()) == expected


def test_composite_total_is_sum(pred_target):
    pred, target = pred_target
    criterion = CompositeLoss({
        'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1,
    })
    losses = criterion(pred, target)
    components = {k: v for k, v in losses.items() if k != 'total'}
    assert torch.isclose(losses['total'], sum(components.values()), atol=1e-6)


def test_composite_weights_scale_contributions(pred_target):
    pred, target = pred_target
    w_char = 2.0
    w_wavelet = 0.0
    criterion = CompositeLoss({'charbonnier': w_char, 'wavelet': w_wavelet})
    losses = criterion(pred, target)
    char_noscale = CompositeLoss({'charbonnier': 1.0})(pred, target)['char']
    assert torch.isclose(losses['char'], char_noscale * w_char, atol=1e-6)


def test_composite_temporal_included_when_prev_provided(pred_target_seq):
    pred_cur, pred_prev, target_cur, target_prev = pred_target_seq
    criterion = CompositeLoss({
        'charbonnier': 1.0, 'temporal_consistency': 0.5,
    })
    losses_with = criterion(pred_cur, target_cur, pred_prev, target_prev)
    assert 'temporal_consistency' in losses_with

    losses_without = criterion(pred_cur, target_cur)
    assert 'temporal_consistency' not in losses_without


def test_composite_gradient_flow(pred_target):
    pred, target = pred_target
    criterion = CompositeLoss({
        'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1,
    })
    losses = criterion(pred, target)
    losses['total'].backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_composite_gradient_flow_with_temporal(pred_target_seq):
    pred_cur, pred_prev, target_cur, target_prev = pred_target_seq
    criterion = CompositeLoss({
        'charbonnier': 1.0, 'temporal_consistency': 0.5,
    })
    losses = criterion(pred_cur, target_cur, pred_prev, target_prev)
    losses['total'].backward()
    for param, name in [(pred_cur, 'pred_cur'), (pred_prev, 'pred_prev')]:
        assert param.grad is not None, f"{name} has no grad"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN grad"
        assert not torch.isinf(param.grad).any(), f"{name} has inf grad"


def test_composite_zero_config_single_loss():
    criterion = CompositeLoss({'charbonnier': 0.0})
    pred = torch.randn(2, 3, 16, 16, requires_grad=True)
    target = torch.randn(2, 3, 16, 16)
    losses = criterion(pred, target)
    assert torch.isclose(losses['char'], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(losses['total'], torch.tensor(0.0), atol=1e-6)


def test_composite_empty_config_defaults_to_charbonnier(pred_target):
    """Empty config defaults charbonnier=1.0, so total > 0."""
    pred, target = pred_target
    criterion = CompositeLoss({})
    losses = criterion(pred, target)
    assert set(losses.keys()) == {'char', 'total'}
    assert losses['total'].item() > 0.0
    assert losses['total'].item() == losses['char'].item()
