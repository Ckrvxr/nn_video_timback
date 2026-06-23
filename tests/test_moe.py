from tests.helpers import mock_triton
mock_triton()

import torch
from components.moe import MoERouter


def test_moe_router_flat_gradients():
    router = MoERouter(n_features=16, n_experts=4, stat_features=6)
    z_t = torch.randn(2, 1, 16)
    ictcp = torch.zeros(2, 3, 32, 32, requires_grad=True)

    idx, weights, logits = router(z_t, ictcp)

    assert idx.shape == (2, 4)
    assert weights.shape == (2, 4)
    assert logits.shape == (2, 4)

    loss = logits.sum()
    loss.backward()
    assert ictcp.grad is not None
    assert not torch.isnan(ictcp.grad).any()
    assert not torch.isinf(ictcp.grad).any()

    lbl = router.load_balancing_loss(logits)
    assert lbl.dim() == 0
    assert lbl.item() >= -1.0
