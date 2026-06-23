import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers import mock_triton, mock_mamba_ssm
mock_triton()
mock_mamba_ssm()

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def val_video_path():
    return PROJECT_ROOT / 'data' / 'val' / 'HR' / 'Hoppers_2026_seg17.mkv'


@pytest.fixture
def val_dir():
    return PROJECT_ROOT / 'data' / 'val'


@pytest.fixture
def composite_loss(device):
    from utils.training.losses.composite import CompositeLoss
    return CompositeLoss({
        'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1,
    })
