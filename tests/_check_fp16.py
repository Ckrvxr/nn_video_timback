"""Verify fp16 cache works correctly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from utils.video_loader import load_video_frame_range
from utils.frame_cache import LazyFrameRange

hr = r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val\HR\Hoppers_2026_seg17.mkv'

fr = LazyFrameRange(hr, 87, window_size=9)
f = fr[0]

cached = fr._cache[0]
print('Cache dtype:', cached.dtype)
print('Cache shape:', cached.shape)
print('Cache MB:', cached.nbytes / 1024**2)
assert cached.dtype == np.float16

# Verify it still works through the grid crop pipeline
from utils.dataset import AV1CompressedVideoDataset
ds = AV1CompressedVideoDataset(
    datasets=[r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val'],
    patch_size=512, frames=9, is_train=True)
item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40,
        ds.videos[0]['ds_root'], 0, 0)
out = ds[item]
print('lr_frames:', out['lr_frames'].shape, out['lr_frames'].dtype)
print('hr:', out['hr'].shape, out['hr'].dtype)
assert out['lr_frames'].dtype == torch.float32
print('fp16 cache OK')
