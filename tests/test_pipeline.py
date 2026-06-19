"""End-to-end pipeline test: decode → ICtCp → grid crop → Dataloader → model."""
import sys, time
from pathlib import Path

# Don't mock mamba_ssm — let it fail naturally so FastSSM fallback is used
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from utils.video_loader import probe_frame_count, probe_resolution
from utils.frame_cache import LazyFrameRange
from utils.dataset import AV1CompressedVideoDataset, VideoBatchSampler, collate_vsr
from torch.utils.data import DataLoader
from models import MambaFixer
from losses.composite import CompositeLoss

val_dir = Path(r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val')
hr = val_dir / 'HR' / 'Hoppers_2026_seg17.mkv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('=' * 50)

# ── 1. LazyFrameRange full decode ────────────────────────────
print('\n[1] LazyFrameRange decode')
n = probe_frame_count(str(hr))
h, w = probe_resolution(str(hr))
fr = LazyFrameRange(str(hr), n, window_size=9)
t0 = time.perf_counter()
for i in range(n):
    f = fr[i]
t1 = time.perf_counter()
print(f'    {n} frames in {t1-t0:.1f}s ({n/(t1-t0):.0f} fps)')
assert f.shape == (h, w, 3)

# ── 2. ICtCp range ───────────────────────────────────────────
print('\n[2] ICtCp range')
print(f'    I:   [{f[:,:,0].min():.4f}, {f[:,:,0].max():.4f}]')
print(f'    Ct:  [{f[:,:,1].min():.4f}, {f[:,:,1].max():.4f}]')
print(f'    Cp:  [{f[:,:,2].min():.4f}, {f[:,:,2].max():.4f}]')
assert f[:,:,0].min() >= -0.05

# ── 3. Dataset + grid crop ───────────────────────────────────
print('\n[3] Dataset + grid crop')
ds = AV1CompressedVideoDataset(datasets=[str(val_dir)], patch_size=512, frames=9, is_train=True)
v = ds.videos[0]
gy, gx = v['h'] // 512, v['w'] // 512
print(f'    Grid: {gy}x{gx}={gy*gx} cells/frame, total samples={len(ds)}')
assert gy * gx == 28

# ── 4. Dataloader (single-worker to avoid hang on Windows fork) ──
print('\n[4] DataLoader(num_workers=0)')
sampler = VideoBatchSampler(ds, batch_size=4, clip_repeat=1)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_vsr, num_workers=0)
loader_iter = iter(loader)
t0 = time.perf_counter()
batches = [next(loader_iter) for _ in range(3)]
t1 = time.perf_counter()
for b in batches:
    assert b['lr_frames'].shape == (4, 9, 3, 512, 512)
    assert b['hr'].shape == (4, 3, 512, 512)
print(f'    3 batches in {t1-t0:.2f}s ({3/(t1-t0):.1f} batch/s)')

# ── 5. CUDA transfer ─────────────────────────────────────────
print(f'\n[5] Transfer to {device}')
b = batches[0]
t0 = time.perf_counter()
lr_cuda = b['lr_frames'].to(device, non_blocking=True)
hr_cuda = b['hr'].to(device, non_blocking=True)
_ = lr_cuda.sum() + hr_cuda.sum()
t1 = time.perf_counter()
print(f'    Transfer: {t1-t0:.4f}s')
print(f'    lr: {lr_cuda.shape} {lr_cuda.device}')

# ── 6. Model forward (synthetic data, small model) ──────────
print('\n[6] Model forward')
model = MambaFixer(64, 32, 2, 4, 2, [1, 2, 4]).to(device)
x = torch.randn((4, 3, 512, 512), device=device).to(memory_format=torch.channels_last)
t0 = time.perf_counter()
model.reset_state(4, device)
pred = model(x)
t1 = time.perf_counter()
print(f'    Forward: {t1-t0:.4f}s, pred={pred.shape}')
assert pred.shape == (4, 3, 512, 512)

# ── 7. Loss backward ─────────────────────────────────────────
print('\n[7] Loss backward')
criterion = CompositeLoss({'charbonnier': 1.0, 'laplacian': 0.5}, device=device)
loss_dict = criterion(pred, x)
loss_dict['total'].backward()
print(f'    loss={loss_dict["total"].item():.6f}')
for k, v in loss_dict.items():
    print(f'      {k}={v.item():.6f}')

print()
print('=' * 50)
print('ALL 7/7 STEPS PASSED')
