"""Measure VRAM during realistic training simulation."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from utils.data.dataset import CompressedVideoDataset, SequentialVideoBatchSampler, collate_video
from torch.utils.data import DataLoader
from models import Timback
from utils.training.losses.composite import CompositeLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_train_model_forward():
    model = Timback(16, 8, 2, 4, 2).to('cuda').train()
    x = torch.randn(1, 3, 64, 64, device='cuda')
    model.reset_state(1, 'cuda')
    y = model(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

    c = CompositeLoss({'charbonnier': 1.0, 'wavelet': 0.5}, device='cuda')
    loss_dict = c(y, x)
    loss_dict['total'].backward()
    assert y.grad is None  # non-leaf, but params should have grad
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def run_profile():
    torch.backends.cudnn.benchmark = True
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    val_dir = Path(__file__).resolve().parent.parent / 'data' / 'val'

    vram0 = torch.cuda.memory_allocated() / 1024**3
    print(f'Initial VRAM: {vram0:.3f}GB')

    ds = CompressedVideoDataset(
        datasets=[str(val_dir)],
        patch_size=512, frames=9, is_train=True)
    sampler = SequentialVideoBatchSampler(ds, batch_size=4)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_video, num_workers=0)
    it = iter(loader)

    model = Timback(64, 32, 2, 100, 2, [1, 2, 4, 32]).to('cuda').train()
    c = CompositeLoss({
        'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1,
    }, device='cuda')

    print(f'Model loaded VRAM: {torch.cuda.memory_allocated()/1024**3:.3f}GB')

    for step in range(10):
        b = next(it)
        lr = b['lr_frames'].cuda(non_blocking=True)
        hr = b['hr'].cuda(non_blocking=True)
        torch.cuda.synchronize()

        for t in range(9):
            if t == 4:
                pred = model(lr[:, t])
                ld = c(pred, hr)
                ld['total'].backward()
            else:
                with torch.no_grad():
                    model.forward_ssm_ictcp(lr[:, t])
        torch.cuda.synchronize()

        if step % 2 == 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        torch.cuda.synchronize()

        if step in (0, 1, 4, 9):
            vram = torch.cuda.memory_allocated() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f'step {step}:  VRAM={vram:.2f}GB  peak={peak:.2f}GB')

    print()
    print(f'VRAM peak:  {torch.cuda.max_memory_allocated()/1024**3:.2f}GB')
    print(f'VRAM final: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
    print(f'VRAM diff:  +{torch.cuda.memory_allocated()/1024**3 - vram0:.3f}GB')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for VRAM profiling.")
        sys.exit(1)
    run_profile()
