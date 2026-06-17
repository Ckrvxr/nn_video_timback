import torch


def _hilbert_d(n: int, x: int, y: int) -> int:
    d = 0
    s = n >> 1
    while s:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_flat_indices(H: int, W: int) -> torch.Tensor:
    N = 1 << (max(H, W) - 1).bit_length()
    order = torch.zeros(H, W, dtype=torch.long)
    for y in range(H):
        for x in range(W):
            order[y, x] = _hilbert_d(N, x, y)
    return order.view(-1).argsort()
