import torch


def fbin_loss(pred: torch.Tensor, target: torch.Tensor,
              grid_sizes: list[int] | None = None,
              eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier-style loss at grid-frequency bins.

    Computes sqrt(|spec_p - spec_t|^2 + eps) - sqrt(eps) at FFT bins
    corresponding to each grid size g.  Same gradient as Charbonnier
    but zero at perfect reconstruction.
    """
    if grid_sizes is None:
        grid_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    spec_p = torch.fft.rfft2(pred.float(), norm='ortho')
    spec_t = torch.fft.rfft2(target.float(), norm='ortho')

    B, C, H, W = pred.shape
    W2 = spec_p.shape[-1]

    loss = torch.tensor(0.0, device=pred.device)
    n = 0
    bias = eps ** 0.5
    for g in grid_sizes:
        ky, kx = H // g, W // g
        if 0 < ky < H and 0 < kx < W2:
            diff = (spec_p[:, :, ky, kx] - spec_t[:, :, ky, kx]).abs()
            loss = loss + (torch.sqrt(diff ** 2 + eps) - bias).mean()
            n += 1

    return loss / max(n, 1)
