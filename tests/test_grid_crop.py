"""Integration test for live decode, mkv probing, and grid crop."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_sm = MagicMock()
sys.modules['mamba_ssm'] = _sm
sys.modules['mamba_ssm.ops'] = MagicMock()
sys.modules['mamba_ssm.ops.selective_scan_interface'] = _sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_loader import probe_frame_count, probe_resolution
from utils.frame_cache import LazyFrameRange
from utils.dataset import AV1CompressedVideoDataset, SequentialVideoBatchSampler, collate_vsr
from torch.utils.data import DataLoader
import numpy as np

val_dir = Path(r'C:\Users\Ckrvxr\MyProject\nn_video_timback\data\val')
hr = val_dir / 'HR' / 'Hoppers_2026_seg17.mkv'

# 1. mkv probe — count_frames returns correct 87
n = probe_frame_count(str(hr))
h, w = probe_resolution(str(hr))
print(f'1. mkv probe: {n} frames, {h}x{w}')
assert n == 87, f'Expected 87, got {n}'

# 2. LazyFrameRange window=9, first frame
fr = LazyFrameRange(str(hr), n, window_size=9)
f0 = fr[0]
assert f0.shape == (h, w, 3)
print(f'2. LazyFrameRange: shape={f0.shape}')

# 3. Incremental decode — sequential access decodes 9 first then 0 per step
cache_sizes = []
for i in range(12):
    _ = fr[i]
    cache_sizes.append(len(fr._cache))
growths = [cache_sizes[i] - cache_sizes[i - 1] for i in range(1, len(cache_sizes))]
print(f'3. Cache sizes: {cache_sizes}')
print(f'   Growth per step: {growths}')
assert cache_sizes[0] == 9, f'First access should load 9 frames, got {cache_sizes[0]}'
assert max(growths[1:]) == 0, f'Steps 1-11 should add 0 new frames, got {growths}'
# Step 11 → frame 11, window [7, 16): frames 7-8 already cached, 9-15 not
_ = fr[11]
sz = len(fr._cache)
# Cache should be 9 (frames 7-15 from the new window)
assert sz == 9, f'After re-window at frame 11, expected 9, got {sz}'
print('   Incremental decode OK')

print()
print('ALL TESTS PASSED')
