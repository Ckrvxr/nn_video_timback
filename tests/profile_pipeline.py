"""Profile each pipeline stage precisely."""
import sys, time, gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

val_dir = Path(r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val')
hr = val_dir / 'HR' / 'Hoppers_2026_seg17.mkv'

N = 10  # number of frames per profile batch
torch.cuda.synchronize() if torch.cuda.is_available() else None

# ═══════════════════════════════════════════════════════════════
# 1. av.open overhead
# ═══════════════════════════════════════════════════════════════
import av
t0 = time.perf_counter()
for _ in range(N):
    c = av.open(str(hr))
    c.close()
t1 = time.perf_counter()
print(f'[1] av.open:            {(t1-t0)/N*1000:.1f}ms')

# ═══════════════════════════════════════════════════════════════
# 2. HEVC decode (sequential, no YUV extract)
# ═══════════════════════════════════════════════════════════════
t0 = time.perf_counter()
for _ in range(3):
    with av.open(str(hr)) as c:
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        list(c.decode(video=0))
t1 = time.perf_counter()
torch.cuda.synchronize() if torch.cuda.is_available() else None
# 87 * 3 = 261 frames
print(f'[2] HEVC decode:        {(t1-t0)/261*1000:.1f}ms/frame  ({(t1-t0)/3:.1f}s for 87 frames)')

# ═══════════════════════════════════════════════════════════════
# 3. HEVC decode + YUV plane read (the full load_video_frame_range)
# ═══════════════════════════════════════════════════════════════
from utils.video_loader import load_video_frame_range

test_ranges = [(0, 9), (40, 49), (78, 87)]
for label, lo, hi in [('sequential [0,9)', *test_ranges[0]),
                       ('seek [40,49)', *test_ranges[1]),
                       ('seek [78,87)', *test_ranges[2])]:
    t0 = time.perf_counter()
    for _ in range(3):
        f = load_video_frame_range(str(hr), lo, hi)
        _ = len(f)
    t1 = time.perf_counter()
    n_frames = hi - lo
    print(f'[3] {label:22s}: {(t1-t0)/3:.3f}s  ({n_frames} frames, {(t1-t0)/3/n_frames*1000:.1f}ms/frame)')

# ═══════════════════════════════════════════════════════════════
# 4. np.stack (YUV list → batch array)
# ═══════════════════════════════════════════════════════════════
yuv_frames = load_video_frame_range(str(hr), 0, 9)
t0 = time.perf_counter()
for _ in range(N):
    a = np.stack(yuv_frames, axis=0)
t1 = time.perf_counter()
print(f'[4] np.stack(9×4K):    {(t1-t0)/N*1000:.1f}ms')

# ═══════════════════════════════════════════════════════════════
# 5. YUV→ICtCp (CUDA)
# ═══════════════════════════════════════════════════════════════
from utils.frame_cache import LazyFrameRange
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Warmup
_ = LazyFrameRange._yuv_to_ictcp(a, bits=10)
torch.cuda.synchronize() if torch.cuda.is_available() else None

t0 = time.perf_counter()
for _ in range(5):
    _ = LazyFrameRange._yuv_to_ictcp(a, bits=10)
t1 = time.perf_counter()
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f'[5] yuv_to_ictcp CUDA: {(t1-t0)/5:.3f}s  ({(t1-t0)/5/9*1000:.1f}ms/frame)')

# ═══════════════════════════════════════════════════════════════
# 6. Full __getitem__: decode → ICtCp → grid crop 1 cell
# ═══════════════════════════════════════════════════════════════
from utils.dataset import AV1CompressedVideoDataset
ds = AV1CompressedVideoDataset(datasets=[str(val_dir)], patch_size=512, frames=9, is_train=True)

# Simulate one __getitem__ call
item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40, ds.videos[0]['ds_root'], 0, 0)
# Warmup
_ = ds[item]
t0 = time.perf_counter()
for _ in range(3):
    _ = ds[item]
t1 = time.perf_counter()
print(f'[6] __getitem__ (cache): {(t1-t0)/3*1000:.1f}ms  (warm cache)')

# Cold (different frame, different grid cell → new decode)
item2 = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 50, ds.videos[0]['ds_root'], 2, 3)
t0 = time.perf_counter()
_ = ds[item2]
t1 = time.perf_counter()
print(f'[6] __getitem__ (cold): {(t1-t0)*1000:.0f}ms  (new decode+ICtCp)')

# ═══════════════════════════════════════════════════════════════
# 7. Model forward on CUDA (synthetic data)
# ═══════════════════════════════════════════════════════════════
from models import MambaFixer
model = MambaFixer(64, 32, 2, 100, 4, [1, 2, 4, 32]).to('cuda')
x = torch.randn((4, 3, 512, 512), device='cuda')
model.reset_state(4, 'cuda')

# Warmup
_ = model(x)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(10):
    _ = model(x)
t1 = time.perf_counter()
torch.cuda.synchronize()
print(f'[7] Model forward:     {(t1-t0)/10*1000:.1f}ms  (4×512², 100 experts, full model)')

# ═══════════════════════════════════════════════════════════════
# 8. Loss backward
# ═══════════════════════════════════════════════════════════════
from losses.composite import CompositeLoss
criterion = CompositeLoss({'charbonnier': 1.0, 'laplacian': 0.5, 'fft': 0.1}, device='cuda')
pred = model(x)
target = torch.randn((4, 3, 512, 512), device='cuda')

t0 = time.perf_counter()
for _ in range(10):
    loss_dict = criterion(pred, target)
    loss_dict['total'].backward(retain_graph=True)
t1 = time.perf_counter()
torch.cuda.synchronize()
print(f'[8] Loss F+B:          {(t1-t0)/10*1000:.1f}ms  (4×512², char+lap+fft)')

# ═══════════════════════════════════════════════════════════════
# 9. DataLoader with SequentialVideoBatchSampler
# ═══════════════════════════════════════════════════════════════
from utils.dataset import SequentialVideoBatchSampler, collate_vsr
from torch.utils.data import DataLoader

sampler = SequentialVideoBatchSampler(ds, batch_size=4)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_vsr, num_workers=0)
loader_iter = iter(loader)

t0 = time.perf_counter()
b1 = next(loader_iter)
t1 = time.perf_counter()
print(f'[9] First batch (cold):       {t1-t0:.1f}s')

t0 = time.perf_counter()
b2 = next(loader_iter)
t1 = time.perf_counter()
print(f'[9] Second batch (incr):      {t1-t0:.1f}s')

t0 = time.perf_counter()
b3 = next(loader_iter)
t1 = time.perf_counter()
print(f'[9] Third batch (incr):       {t1-t0:.1f}s')

t0 = time.perf_counter()
b4 = next(loader_iter)
t1 = time.perf_counter()
print(f'[9] Fourth batch (incr):      {t1-t0:.1f}s')

t0 = time.perf_counter()
b5 = next(loader_iter)
t1 = time.perf_counter()
print(f'[9] Fifth batch (incr):       {t1-t0:.1f}s')

print()
print('Done profiling.')
