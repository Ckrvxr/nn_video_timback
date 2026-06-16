import sys
from pathlib import Path
from unittest.mock import MagicMock

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

import torch
from losses.fft import FFTLoss

def test_fft_loss_correctness():
    # Instantiate the frequency domain FFT loss
    criterion = FFTLoss()
    
    # Generate mock tensors (batch size 2, channels 3, height 32, width 32)
    # Require gradients for prediction to check backpropagation
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    
    # Compute loss
    loss = criterion(pred, target)
    
    # 1. Assertions on loss properties
    assert isinstance(loss, torch.Tensor), "FFTLoss should return a PyTorch Tensor"
    assert loss.dim() == 0, "FFTLoss should return a scalar tensor"
    assert loss.item() >= 0.0, "FFTLoss value should be non-negative"
    
    # 2. Check backpropagation / gradient flow
    loss.backward()
    assert pred.grad is not None, "Gradients should propagate to the prediction tensor"
    assert not torch.isnan(pred.grad).any(), "Gradients should not contain NaNs"
    assert not torch.isinf(pred.grad).any(), "Gradients should not contain Infs"
    
    # 3. Test perfect match case
    pred_perfect = target.clone().requires_grad_(True)
    loss_perfect = criterion(pred_perfect, target)
    assert torch.isclose(loss_perfect, torch.tensor(0.0), atol=1e-6), "FFTLoss for identical inputs should be zero"
    
    print("FFTLoss verification and gradient check passed successfully!")

if __name__ == '__main__':
    test_fft_loss_correctness()
