from tests.helpers import mock_triton, mock_mamba_ssm
mock_triton()
mock_mamba_ssm()

import torch
import pytest
from yaml import safe_load
from models import Timback


@pytest.fixture
def small_model():
    model = Timback(
        num_features=16, state_dimension=8, num_features_stream=2,
        num_experts=4, n_active=2,
    )
    model.eval()
    return model


def test_validate_forward_runs(small_model):
    x = torch.randn(1, 3, 64, 64)
    small_model.reset_state(1, torch.device('cpu'))
    pred = small_model(x)
    assert pred.shape == x.shape
    assert not torch.isnan(pred).any()
    assert not torch.isinf(pred).any()


def test_validate_model_script_entry_point():
    """Verify the script's run_model function format loads a model and produces pred."""
    from pathlib import Path
    import numpy as np

    device = torch.device('cpu')
    config = safe_load(open('configs/prod.yaml'))
    model = Timback(
        num_features=config['model_architecture'].get('num_features', 16),
        state_dimension=config['model_architecture'].get('state_dimension', 16),
    ).to(device)
    model.eval()
    model.reset_state(1, device)

    lr = np.random.randn(9, 3, 64, 64).astype(np.float32)
    lr_t = torch.from_numpy(lr).float().unsqueeze(0).to(device)
    center_idx = 4
    pred = model(lr_t[:, center_idx].to(memory_format=torch.channels_last))
    assert pred.shape == (1, 3, 64, 64)
    diff = pred - lr_t[:, center_idx]
    assert diff.shape == pred.shape
