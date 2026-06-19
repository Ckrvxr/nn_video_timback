"""Measure VRAM during realistic training simulation."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.backends.cudnn.benchmark = True
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

from utils.dataset import AV1CompressedVideoDataset, SequentialVideoBatchSampler, collate_vsr
from torch.utils.data import DataLoader
from models import MambaFixer
from losses.composite import CompositeLoss

vram0 = torch.cuda.memory_allocated() / 1024**3
print(f'Initial VRAM: {vram0:.3f}GB')

ds = AV1CompressedVideoDataset(
    datasets=[r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val'],
    patch_size=512, frames=9, is_train=True)
sampler = SequentialVideoBatchSampler(ds, batch_size=4)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_vsr, num_workers=0)
it = iter(loader)

model = MambaFixer(64, 32, 2, 100, 2, [1, 2, 4, 32]).to('cuda').train()
c = CompositeLoss({'charbonnier': 1.0, 'laplacian': 0.5, 'color_tight': 0.1, 'fft': 0.1}, device='cuda')

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
                model(lr[:, t], ssm_only=True)
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
