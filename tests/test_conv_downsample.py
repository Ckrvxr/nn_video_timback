import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models import MambaFixer

def test_mamba_fixer_forward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize model with test config settings
    print("Initializing MambaFixer with num_features=16...")
    model = MambaFixer(num_features=16, state_dimension=8, num_features_stream=2, num_experts=4, n_active=2).to(device)
    model.eval()
    
    # Create random input tensor (batch_size=2, channels=3, height=32, width=32)
    x = torch.randn(2, 3, 32, 32, device=device)
    
    # Reset state
    model.reset_state(2, device)
    
    # Run forward pass
    print("Running forward pass...")
    out = model(x)
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 3, 32, 32), f"Expected shape (2, 3, 32, 32), got {out.shape}"
    print("Test passed successfully!")

if __name__ == '__main__':
    test_mamba_fixer_forward()
