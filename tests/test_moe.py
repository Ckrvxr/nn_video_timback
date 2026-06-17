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
from models.components.moe import MoERouter

def test_moe_router_flat_gradients():
    # Instantiate the MoE Router
    router = MoERouter(n_features=16, n_experts=4, stat_features=6)
    
    # 1. Test forward pass with flat/constant ictcp features
    z_t = torch.randn(2, 1, 16)  # [B, 1, n_features]
    ictcp = torch.zeros(2, 3, 32, 32, requires_grad=True)  # Completely flat/constant inputs
    
    idx, logits = router(z_t, ictcp)
    
    assert idx.shape == (2,)
    assert logits.shape == (2, 4)
    
    # 2. Backward pass to check gradient stability (must not be NaN or Inf)
    loss = logits.sum()
    loss.backward()
    
    assert ictcp.grad is not None, "Gradients should propagate to the ictcp tensor"
    assert not torch.isnan(ictcp.grad).any(), "Gradients must not contain NaNs under flat inputs"
    assert not torch.isinf(ictcp.grad).any(), "Gradients must not contain Infs under flat inputs"
    
    # 3. Test load balancing loss
    lbl = router.load_balancing_loss(logits)
    assert lbl.dim() == 0
    assert lbl.item() >= -1.0
    
    print("MoERouter flat gradient stability check passed successfully!")

if __name__ == '__main__':
    test_moe_router_flat_gradients()
