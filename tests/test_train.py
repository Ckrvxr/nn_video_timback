"""Tests for training control flow: multi-dataset loading, pause/exit."""
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from tests.helpers import mock_triton
mock_triton()

import torch
import torch.nn as nn
from scripts import train
from utils.training import cli


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1).half()
        self._balancing_loss = torch.tensor(0.0, requires_grad=True)

    def train(self, mode=True):
        pass

    def forward(self, x, **kwargs):
        return self.conv(x)

    def reset_state(self, batch_size, device):
        pass


@patch('utils.training.setup.create_dataloader')
@patch('scripts.train.train_epoch')
@patch('scripts.train.validate')
@patch('scripts.train.save_checkpoint')
@patch('scripts.train.safe_load')
@patch('scripts.train.open')
def test_multi_dataset_sequential(
    mock_open,
    mock_safe_load,
    mock_save_checkpoint,
    mock_validate,
    mock_train_epoch,
    mock_create_dataloader,
):
    mock_config = {
        'output_directory': 'runs/mamba_test',
        'random_seed': 42,
        'dataset': {
            'dataset_paths': ['./data/datasets/ds_a', './data/datasets/ds_b'],
            'patch_size': 256,
            'num_frames': 5,
            'num_workers': 2,
            'validation_batch_size': 2,
            'validation_num_workers': 2,
        },
        'model_architecture': {
            'model_name': 'timback',
            'num_features': 64,
            'state_dimension': 32,
            'num_features_stream': 2,
            'num_experts': 100,
            'n_active': 4,
            'dilation_rates': [1, 2, 4, 32],
        },
        'training_settings': {
            'batch_size': 4,
            'num_epochs': 1,
            'learning_rate': 2e-4,
            'min_learning_rate': 1e-6,
            'weight_decay': 1e-4,
            'adam_beta1': 0.9,
            'adam_beta2': 0.99,
            'gradient_clipping_threshold': 1.0,
            'use_mixed_precision': True,
        },
        'loss_weights': {
            'charbonnier': 1.0,
        },
        'logging_settings': {
            'validation_interval': 1,
        },
    }
    mock_safe_load.return_value = mock_config
    mock_validate.return_value = (30.0, 0.9, 80.0)
    mock_train_epoch.return_value = 0.05

    mock_args = MagicMock()
    mock_args.config = 'dummy.yaml'
    mock_args.resume = None
    mock_args.pretrained = None
    mock_args.device = 'cpu'
    mock_args.seed = 42

    train.EXIT_FLAG = False

    def mock_exists(self):
        if self.name in ('.pause', '.exit'):
            return False
        return True

    with patch('scripts.train.parse_args', return_value=mock_args), \
         patch('scripts.train.Path.mkdir') as mock_mkdir, \
         patch('scripts.train.Path.exists', new=mock_exists), \
         patch('scripts.train.Path.glob', return_value=[]):
        train.main()

        ds_call = call(
            datasets=['./data/datasets/ds_a', './data/datasets/ds_b'],
            batch_size=4, patch_size=256, frames=5, workers=2,
            is_train=True, segment_repeat=1, sequential=False,
            data_type='compressed', max_cached_segments=8,
        )
        mock_create_dataloader.assert_has_calls([ds_call])
        assert mock_train_epoch.call_count == 1


def test_pause_and_exit_triggers():
    cli.EXIT_FLAG = False

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        pause_file = run_dir / '.pause'
        exit_file = run_dir / '.exit'

        model = DummyModel()
        config = {
            'loss_weights': {},
            'training_settings': {
                'gradient_accumulation_steps': 1,
                'gradient_clipping_threshold': 1.0,
                'batch_size': 2,
            },
            'model_architecture': {
                'model_name': 'timback',
                'num_experts': 100,
                'n_active': 4,
            },
            'dataset': {
                'num_frames': 5,
                'sequential_mode': False,
            }
        }

        batches = [
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
        ]

        class ControlDataLoader:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __iter__(self):
                for idx, item in enumerate(self.data):
                    if idx == 1:
                        pause_file.touch()
                        def remove_pause():
                            time.sleep(1.5)
                            if pause_file.exists():
                                pause_file.unlink()
                        threading.Thread(target=remove_pause, daemon=True).start()
                    elif idx == 2:
                        exit_file.touch()
                    yield item

        loader = ControlDataLoader(batches)

        def criterion(pred, target):
            loss_val = torch.mean((pred - target) ** 2)
            return {'total': loss_val, 'moe': torch.tensor(0.0, requires_grad=True)}

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = 'cpu'

        start_time = time.perf_counter()
        loss = train.train_epoch(
            model=model, loader=loader, criterion=criterion,
            optimizer=optimizer, device=device, config=config, run_dir=run_dir,
        )
        duration = time.perf_counter() - start_time

        assert duration >= 1.5, f"Training did not pause! Duration: {duration:.2f}s"
        assert cli.EXIT_FLAG is True
        assert not exit_file.exists()
        assert not pause_file.exists()
