import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock Triton for Windows / non-Triton environments
class TritonMock(MagicMock):
    @classmethod
    def jit(cls, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator

sys.modules['triton'] = TritonMock()
sys.modules['triton.language'] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import train

@patch('scripts.train.warm_blosc_cache')
@patch('scripts.train.clear_dataset_cache')
@patch('scripts.train.create_dataloader')
@patch('scripts.train.train_epoch')
@patch('scripts.train.validate')
@patch('scripts.train.save_checkpoint')
@patch('scripts.train.safe_load')
@patch('scripts.train.open')
def test_multi_dataset_sequential_cache(
    mock_open,
    mock_safe_load,
    mock_save_checkpoint,
    mock_validate,
    mock_train_epoch,
    mock_create_dataloader,
    mock_clear_dataset_cache,
    mock_warm_blosc_cache
):
    # Setup mock config with 2 dataset paths
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
            'checkpoint_save_interval': 1,
            'validation_interval': 1,
        }
    }
    mock_safe_load.return_value = mock_config

    # Mock return values for validation and train_epoch
    mock_validate.return_value = (30.0, 0.9, 80.0)
    mock_train_epoch.return_value = 0.05

    # Mock command line arguments
    mock_args = MagicMock()
    mock_args.config = 'dummy.yaml'
    mock_args.resume = None
    mock_args.pretrained = None
    mock_args.device = 'cpu'
    mock_args.seed = 42

    # Reset exit flag
    train.EXIT_FLAG = False

    # Side effect for Path.exists to prevent fake pause/exit detections
    def mock_exists(self):
        if self.name in ('.pause', '.exit'):
            return False
        return True

    with patch('scripts.train.parse_args', return_value=mock_args), \
         patch('scripts.train.Path.mkdir') as mock_mkdir, \
         patch('scripts.train.Path.exists', new=mock_exists), \
         patch('scripts.train.Path.glob', return_value=[]):

        # Call main
        train.main()

        # Check that we did NOT call warm_blosc_cache globally at startup
        # (It should only be called inside the dataset loop, which sets target_dataset_path)
        startup_warm_calls = [
            call for call in mock_warm_blosc_cache.call_args_list 
            if len(call.args) == 1 and not call.kwargs.get('target_dataset_path')
        ]
        assert len(startup_warm_calls) == 0, "warm_blosc_cache should not be called globally for all datasets at startup"

        # Check that warm_blosc_cache was called once for each dataset path during training
        mock_warm_blosc_cache.assert_any_call(mock_config, target_dataset_path='./data/datasets/ds_a')
        mock_warm_blosc_cache.assert_any_call(mock_config, target_dataset_path='./data/datasets/ds_b')

        # Check that clear_dataset_cache was called once for each dataset path after training on it
        mock_clear_dataset_cache.assert_any_call('./data/datasets/ds_a')
        mock_clear_dataset_cache.assert_any_call('./data/datasets/ds_b')

        # Verify that create_dataloader was called with single dataset paths (with persistent_workers=False)
        mock_create_dataloader.assert_any_call(
            datasets=['./data/datasets/ds_a'],
            batch_size=4,
            patch_size=256,
            frames=5,
            workers=2,
            is_train=True,
            cache_max_videos=None,
            clip_repeat=1,
            sequential=False,
            persistent_workers=False
        )

        # Since validation_interval=1, check that we also sequentially warmed, validated, and cleared for validation
        mock_warm_blosc_cache.assert_any_call(mock_config, target_dataset_path='./data/datasets/ds_a')
        mock_warm_blosc_cache.assert_any_call(mock_config, target_dataset_path='./data/datasets/ds_b')

        # Total warm calls should be: 2 for train, 2 for validation
        assert mock_warm_blosc_cache.call_count == 4

        # Total clear calls should be: 2 for train, 2 for validation
        assert mock_clear_dataset_cache.call_count == 4

        print("Multi-dataset sequential cache test passed successfully!")

if __name__ == '__main__':
    test_multi_dataset_sequential_cache()
