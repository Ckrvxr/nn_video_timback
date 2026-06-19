"""Test crop-first YUV→ICtCp pipeline."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from utils.frame_cache import LazyFrameRange
from utils.dataset import AV1CompressedVideoDataset, SequentialVideoBatchSampler, collate_vsr, _yuv_to_ictcp
from torch.utils.data import DataLoader

hr = r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val\HR\Hoppers_2026_seg17.mkv'

# 1. LazyFrameRange stores YUV
fr = LazyFrameRange(hr, 87, window_size=9)
f = fr[0]
print('[1] Cache dtype:', f.dtype, 'shape:', f.shape, 'MB:', f.nbytes / 1024**2)
assert f.dtype == np.uint16

# 2. crop-first ICtCp
yuv_crop = f[0:512, 0:512]
ictcp = _yuv_to_ictcp(yuv_crop[np.newaxis, ...])
print('[2] Crop ICtCp:', ictcp.shape, ictcp.dtype)

# 3. Dataset __getitem__
ds = AV1CompressedVideoDataset(
    datasets=[r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val'],
    patch_size=512, frames=9, is_train=True)
item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40,
        ds.videos[0]['ds_root'], 0, 0)
t0 = time.perf_counter()
out = ds[item]
t1 = time.perf_counter()
print('[3] __getitem__: {:.3f}s'.format(t1 - t0))
print('    lr_frames:', out['lr_frames'].shape, out['lr_frames'].dtype)
print('    hr:', out['hr'].shape, out['hr'].dtype)
print('    I range: [{:.4f}, {:.4f}]'.format(out['hr'][0].min(), out['hr'][0].max()))
assert out['lr_frames'].dtype == torch.float32

# 4. DataLoader
sampler = SequentialVideoBatchSampler(ds, batch_size=4)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_vsr, num_workers=0)
it = iter(loader)
t0 = time.perf_counter()
for i in range(3):
    b = next(it)
    lr = b['lr_frames'].cuda()
    hr = b['hr'].cuda()
    _ = lr.sum() + hr.sum()
torch.cuda.synchronize()
t1 = time.perf_counter()
print('[4] 3 batches: {:.2f}s  ({:.2f}s/batch)'.format(t1 - t0, (t1 - t0) / 3))

# 5. Model forward (verify end-to-end)
from models import MambaFixer
model = MambaFixer(64, 32, 2, 100, 2, [1, 2, 4, 32]).to('cuda').eval()
x = out['hr'].unsqueeze(0).cuda()
model.reset_state(1, 'cuda')
with torch.no_grad():
    p = model(x)
print('[5] Model forward: pred={} I=[{:.4f},{:.4f}]'.format(
    p.shape, p[0, 0].min(), p[0, 0].max()))

print('\nALL OK')
