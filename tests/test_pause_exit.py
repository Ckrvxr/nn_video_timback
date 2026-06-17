import sys
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

# Mock Triton for Windows/non-Triton environments
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

import torch
import torch.nn as nn
from scripts import train

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self._balancing_loss = torch.tensor(0.0, requires_grad=True)
    def train(self, mode=True):
        pass
    def forward(self, x, **kwargs):
        # x is [B, C, H, W]
        return self.conv(x)
    def reset_state(self, batch_size, device):
        pass

def test_pause_and_exit_triggers():
    # Reset exit flag at start
    train.EXIT_FLAG = False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        pause_file = run_dir / '.pause'
        exit_file = run_dir / '.exit'
        
        # Prepare mock objects for train_epoch
        model = DummyModel()
        
        # Create a simple config
        config = {
            'training_settings': {
                'gradient_accumulation_steps': 1,
                'gradient_clipping_threshold': 1.0,
                'batch_size': 2,
            },
            'model_architecture': {
                'model_name': 'mamba_fixer',
                'num_experts': 100,
                'n_active': 4,
            },
            'dataset': {
                'num_frames': 5,
                'sequential_mode': False,
            }
        }
        
        # Simple batches
        batches = [
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
            {'lr_frames': torch.zeros(2, 5, 3, 32, 32), 'hr': torch.zeros(2, 3, 32, 32)},
        ]
        
        # A custom iterator that pauses/exits training using file controls
        class ControlDataLoader:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __iter__(self):
                for idx, item in enumerate(self.data):
                    if idx == 1:
                        # Before returning 2nd batch, trigger a pause file
                        pause_file.touch()
                        # Start a thread to remove the pause file after 1.5 seconds
                        def remove_pause():
                            time.sleep(1.5)
                            if pause_file.exists():
                                pause_file.unlink()
                        threading.Thread(target=remove_pause, daemon=True).start()
                    elif idx == 2:
                        # Before returning 3rd batch, trigger exit file
                        exit_file.touch()
                    yield item

        loader = ControlDataLoader(batches)
        
        # Mock criterion (MSE loss to preserve gradient computation) and optimizer
        def criterion(pred, target):
            loss_val = torch.mean((pred - target) ** 2)
            return {'total': loss_val, 'moe': torch.tensor(0.0, requires_grad=True)}
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = 'cpu'
        
        # Run train_epoch
        start_time = time.perf_counter()
        loss = train.train_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            run_dir=run_dir
        )
        duration = time.perf_counter() - start_time
        
        # Assertions:
        # 1. The training should have paused for at least 1.5 seconds
        assert duration >= 1.5, f"Training did not pause! Duration: {duration:.2f}s"
        # 2. The training epoch should have terminated early after the 2nd batch (idx=1 is processed, idx=2 triggers exit)
        assert train.EXIT_FLAG is True, "EXIT_FLAG was not set to True by .exit trigger"
        # 3. The exit file should have been deleted (consumed)
        assert not exit_file.exists(), ".exit file was not deleted after being processed"
        # 4. The pause file should have been deleted by our thread
        assert not pause_file.exists(), ".pause file still exists"
        
        print("Pause and exit triggers test passed successfully!")

if __name__ == "__main__":
    test_pause_and_exit_triggers()
