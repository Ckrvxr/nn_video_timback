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
import torch.nn as nn
from utils.sam import SAM

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize with static weights so tests are deterministic
        self.w = nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))
    def forward(self, x):
        return x * self.w

def test_sam_optimizer_math():
    model = SimpleModel()
    
    # Instantiate SAM wrapper over standard SGD optimizer
    optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD, rho=0.5, lr=0.1)
    
    inputs = torch.tensor([1.0, 2.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    # ------------------
    # Step 1: First Pass
    # ------------------
    outputs = model(inputs)
    loss = torch.sum((outputs - targets) ** 2)
    # loss = (2*1 - 0)^2 + (3*2 - 0)^2 = 4 + 36 = 40
    assert torch.isclose(loss, torch.tensor(40.0))
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    # dLoss/dw_0 = 2 * (2*1) * 1 = 4
    # dLoss/dw_1 = 2 * (3*2) * 2 = 24
    assert torch.isclose(model.w.grad[0], torch.tensor(4.0))
    assert torch.isclose(model.w.grad[1], torch.tensor(24.0))
    
    # Save original weights before SAM perturbation
    original_w = model.w.data.clone()
    
    # Call first step (climb to adversarial weights)
    optimizer.first_step(zero_grad=True)
    
    # Ggrad norm = sqrt(4^2 + 24^2) = sqrt(16 + 576) = sqrt(592) = 24.33
    # scale = rho / grad_norm = 0.5 / 24.33 = 0.02055
    # e_w_0 = 4 * 0.02055 = 0.0822
    # e_w_1 = 24 * 0.02055 = 0.4932
    # w_0_new = 2.0 + 0.0822 = 2.0822
    # w_1_new = 3.0 + 0.4932 = 3.4932
    assert torch.isclose(model.w[0], original_w[0] + 4.0 * (0.5 / 592**0.5), rtol=1e-4)
    assert torch.isclose(model.w[1], original_w[1] + 24.0 * (0.5 / 592**0.5), rtol=1e-4)
    
    # -------------------
    # Step 2: Second Pass
    # -------------------
    # Gradients should have been zeroed by first_step(zero_grad=True)
    assert model.w.grad is None or torch.all(model.w.grad == 0)
    
    outputs_adv = model(inputs)
    loss_adv = torch.sum((outputs_adv - targets) ** 2)
    loss_adv.backward()
    
    # Save adversarial gradients
    adv_grad = model.w.grad.clone()
    
    # Call second step (restores original weights and performs SGD update using adversarial gradients)
    optimizer.second_step(zero_grad=True)
    
    # Verify that original weights were restored and updated with SGD:
    # w_new = original_w - lr * adv_grad
    expected_w = original_w - 0.1 * adv_grad
    assert torch.allclose(model.w, expected_w)
    
    print("SAM mathematical optimizer test passed successfully!")

def test_sam_invalid_gradients():
    def get_setup():
        m = SimpleModel()
        opt = SAM(m.parameters(), base_optimizer=torch.optim.SGD, rho=0.5, lr=0.1)
        return m, opt

    # Case 1: Zero gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([0.0, 0.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    # Since grad_norm is 0, perturbation should be skipped and weights should not change
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)
    
    # Case 2: NaN gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([float('nan'), 1.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    # Since grad_norm has NaN, perturbation is skipped
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)
    
    # Case 3: Inf gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([float('inf'), 1.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    # Since grad_norm is Inf, perturbation is skipped
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)
    
    print("SAM invalid gradient stability test passed successfully!")

if __name__ == '__main__':
    test_sam_optimizer_math()
    test_sam_invalid_gradients()
