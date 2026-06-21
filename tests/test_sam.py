from tests.helpers import mock_triton
mock_triton()

import torch
import torch.nn as nn
from utils.training.sam import SAM


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float32))

    def forward(self, x):
        return x * self.w


def test_sam_optimizer_math():
    model = SimpleModel()
    optimizer = SAM(model.parameters(), base_optimizer=torch.optim.SGD, rho=0.5, lr=0.1)

    inputs = torch.tensor([1.0, 2.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 0.0], dtype=torch.float32)

    outputs = model(inputs)
    loss = torch.sum((outputs - targets) ** 2)
    assert torch.isclose(loss, torch.tensor(40.0))

    optimizer.zero_grad()
    loss.backward()
    assert torch.isclose(model.w.grad[0], torch.tensor(4.0))
    assert torch.isclose(model.w.grad[1], torch.tensor(24.0))

    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=True)

    grad_norm = (4.0 ** 2 + 24.0 ** 2) ** 0.5
    scale = 0.5 / grad_norm
    assert torch.isclose(model.w[0], original_w[0] + 4.0 * scale, rtol=1e-4)
    assert torch.isclose(model.w[1], original_w[1] + 24.0 * scale, rtol=1e-4)

    assert model.w.grad is None or torch.all(model.w.grad == 0)

    outputs_adv = model(inputs)
    loss_adv = torch.sum((outputs_adv - targets) ** 2)
    loss_adv.backward()
    adv_grad = model.w.grad.clone()
    optimizer.second_step(zero_grad=True)

    expected_w = original_w - 0.1 * adv_grad
    assert torch.allclose(model.w, expected_w)


def test_sam_invalid_gradients():
    def get_setup():
        m = SimpleModel()
        opt = SAM(m.parameters(), base_optimizer=torch.optim.SGD, rho=0.5, lr=0.1)
        return m, opt

    # Zero gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([0.0, 0.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)

    # NaN gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([float('nan'), 1.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)

    # Inf gradients
    model, optimizer = get_setup()
    optimizer.zero_grad()
    model.w.grad = torch.tensor([float('inf'), 1.0])
    original_w = model.w.data.clone()
    optimizer.first_step(zero_grad=False)
    assert torch.all(model.w == original_w)
    optimizer.second_step(zero_grad=True)
