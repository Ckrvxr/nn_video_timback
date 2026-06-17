"""Test multi-dataset sequential loading (live decode, no cache)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Full mock of mamba_ssm + triton to avoid GPU dependencies in unit tests
_sm = MagicMock()
_sm.selective_scan_fn = lambda *a, **kw: (MagicMock(), MagicMock())
_sm.mamba_inner_fn = lambda *a, **kw: MagicMock()
sys.modules['mamba_ssm'] = _sm
sys.modules['mamba_ssm.ops'] = MagicMock()
sys.modules['mamba_ssm.ops.selective_scan_interface'] = _sm
sys.modules['mamba_ssm.ops.triton'] = MagicMock()
sys.modules['mamba_ssm.ops.triton.layer_norm'] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import train


@patch('scripts.train.create_dataloader')
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
            'model_name': 'mamba_fixer',
            'num_features': 64,
            'state_dimension': 32,
            'num_features_stream': 2,
            'num_experts': 42,
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

        # Verify dataloader was created per-dataset path with persistent_workers=False
        ds_a_call = call(
            datasets=['./data/datasets/ds_a'],
            batch_size=4, patch_size=256, frames=5, workers=2,
            is_train=True, clip_repeat=1, sequential=False,
            persistent_workers=False,
        )
        ds_b_call = call(
            datasets=['./data/datasets/ds_b'],
            batch_size=4, patch_size=256, frames=5, workers=2,
            is_train=True, clip_repeat=1, sequential=False,
            persistent_workers=False,
        )
        mock_create_dataloader.assert_has_calls([ds_a_call, ds_b_call], any_order=True)
        assert mock_train_epoch.call_count >= 2

        print("Multi-dataset sequential test passed!")


if __name__ == '__main__':
    test_multi_dataset_sequential()
