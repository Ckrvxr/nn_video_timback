"""Profile data pipeline stages: decode, color conversion, DataLoader, model, and loss."""
import sys
import time
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np


def test_profile_pipeline_model_forward():
    from models import MambaFixer
    from utils.training.losses.composite import CompositeLoss
    model = MambaFixer(16, 8, 2, 4, 2)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    model.reset_state(1, 'cpu')
    y = model(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

    c = CompositeLoss({'charbonnier': 1.0})
    loss_dict = c(y, x)
    assert loss_dict['total'].item() >= 0.0


def test_profile_pipeline_color_conversion():
    from models.components.color_space import yuv_to_ictcp_np, rgb_to_ictcp_np, ictcp_to_rgb_np
    yuv = np.random.randint(0, 256, (1, 64, 64, 3)).astype(np.uint8)
    ictcp = yuv_to_ictcp_np(yuv)
    assert ictcp.shape == yuv.shape
    assert not np.isnan(ictcp).any()

    rgb = np.random.uniform(0.0, 1.0, (1, 64, 64, 3)).astype(np.float32)
    ictcp_rgb = rgb_to_ictcp_np(rgb)
    assert ictcp_rgb.shape == rgb.shape
    rgb_back = ictcp_to_rgb_np(ictcp_rgb)
    assert np.allclose(rgb, rgb_back, atol=1e-5)


def run_profile():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    hr = PROJECT_ROOT / 'data' / 'val' / 'HR' / 'Hoppers_2026_seg17.mkv'
    N = 10

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 1. av.open overhead
    print("--- 1. av.open overhead ---")
    import av
    t0 = time.perf_counter()
    for _ in range(N):
        c = av.open(str(hr))
        c.close()
    t1 = time.perf_counter()
    print(f"  av.open:            {(t1-t0)/N*1000:.1f}ms")

    # 2. HEVC decode (sequential, no YUV extract)
    print("\n--- 2. HEVC decode ---")
    t0 = time.perf_counter()
    for _ in range(3):
        with av.open(str(hr)) as c:
            s = c.streams.video[0]
            s.thread_type = 'AUTO'
            list(c.decode(video=0))
    t1 = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  HEVC decode:        {(t1-t0)/261*1000:.1f}ms/frame  ({(t1-t0)/3:.1f}s for 87 frames)")

    # 3. Full decode + YUV plane read
    print("\n--- 3. Decode + YUV plane read ---")
    from utils.data.video_loader import load_video_frame_range
    test_ranges = [(0, 9), (40, 49), (78, 87)]
    for label, lo, hi in [
        ('sequential [0,9)', *test_ranges[0]),
        ('seek [40,49)', *test_ranges[1]),
        ('seek [78,87)', *test_ranges[2]),
    ]:
        t0 = time.perf_counter()
        for _ in range(3):
            f = load_video_frame_range(str(hr), lo, hi)
            _ = len(f)
        t1 = time.perf_counter()
        n_frames = hi - lo
        print(f"  {label:22s}: {(t1-t0)/3:.3f}s  ({n_frames} frames, {(t1-t0)/3/n_frames*1000:.1f}ms/frame)")

    # 4. np.stack
    print("\n--- 4. np.stack ---")
    yuv_frames = load_video_frame_range(str(hr), 0, 9)
    t0 = time.perf_counter()
    for _ in range(N):
        a = np.stack(yuv_frames, axis=0)
    t1 = time.perf_counter()
    print(f"  np.stack(9x4K):    {(t1-t0)/N*1000:.1f}ms")

    # 5. YUV -> ICtCp CUDA
    print("\n--- 5. Color conversion ---")
    from utils.data.dataset import _yuv_to_ictcp
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    _ = _yuv_to_ictcp(a)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(5):
        _ = _yuv_to_ictcp(a)
    t1 = time.perf_counter()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  YUV->ICtCp CUDA:   {(t1-t0)/5:.3f}s  ({(t1-t0)/5/9*1000:.1f}ms/frame)")

    # 6. Full __getitem__
    print("\n--- 6. Dataset __getitem__ ---")
    from utils.data.dataset import CompressedVideoDataset
    ds = CompressedVideoDataset(
        datasets=[str(PROJECT_ROOT / 'data' / 'val')], patch_size=512, frames=9, is_train=True)
    item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40, ds.videos[0]['ds_root'], 0, 0)
    _ = ds[item]
    t0 = time.perf_counter()
    for _ in range(3):
        _ = ds[item]
    t1 = time.perf_counter()
    print(f"  __getitem__ (warm):  {(t1-t0)/3*1000:.1f}ms")
    item2 = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 50, ds.videos[0]['ds_root'], 2, 3)
    t0 = time.perf_counter()
    _ = ds[item2]
    t1 = time.perf_counter()
    print(f"  __getitem__ (cold):  {(t1-t0)*1000:.0f}ms")

    # 7. Model forward
    print("\n--- 7. Model forward ---")
    from models import MambaFixer
    model = MambaFixer(64, 32, 2, 100, 4, [1, 2, 4, 32]).to('cuda')
    x = torch.randn((4, 3, 512, 512), device='cuda')
    model.reset_state(4, 'cuda')
    _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = model(x)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"  Model forward:      {(t1-t0)/10*1000:.1f}ms  (4x512^2, 100 experts)")

    # 8. Loss backward
    print("\n--- 8. Loss backward ---")
    from utils.losses.composite import CompositeLoss
    criterion = CompositeLoss({'charbonnier': 1.0, 'wavelet': 0.5, 'sobel': 0.05, 'fft': 0.1}, device='cuda')
    pred = model(x)
    target = torch.randn((4, 3, 512, 512), device='cuda')
    t0 = time.perf_counter()
    for _ in range(10):
        loss_dict = criterion(pred, target)
        loss_dict['total'].backward(retain_graph=True)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"  Loss F+B:           {(t1-t0)/10*1000:.1f}ms  (char+lap+fft)")

    # 9. DataLoader batches
    print("\n--- 9. DataLoader batches (SequentialVideoBatchSampler) ---")
    from utils.data.dataset import SequentialVideoBatchSampler, collate_video
    from torch.utils.data import DataLoader
    sampler = SequentialVideoBatchSampler(ds, batch_size=4)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_video, num_workers=0)
    loader_iter = iter(loader)
    t0 = time.perf_counter()
    b1 = next(loader_iter)
    t1 = time.perf_counter()
    print(f"  First batch (cold):       {t1-t0:.1f}s")
    for i in range(2, 6):
        t0 = time.perf_counter()
        _ = next(loader_iter)
        t1 = time.perf_counter()
        print(f"  Batch {i} (incr):          {t1-t0:.1f}s")

    # 10. Color conversion throughput benchmark
    print("\n--- 10. Color conversion throughput ---")
    from models.components.color_space import (
        yuv_to_ictcp_np, rgb_to_ictcp_np, ictcp_to_rgb_np,
        yuv_to_ictcp, rgb_to_ictcp, ictcp_to_rgb,
    )
    yuv = np.random.randint(0, 256, (N, 1080, 1920, 3)).astype(np.uint8)
    rgb = np.random.uniform(0.0, 1.0, (N, 1080, 1920, 3)).astype(np.float32)
    yuv_t = (torch.from_numpy(yuv.astype(np.float32)).permute(0, 3, 1, 2).contiguous() / 127.5 - 1.0).to('cuda')
    rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().to('cuda')

    def bench(name, fn, warmup=5):
        for i in range(min(warmup, N)):
            fn(yuv[i] if 'YUV' in name else rgb[i])
        t0 = time.perf_counter()
        for i in range(N):
            fn(yuv[i] if 'YUV' in name else rgb[i])
        t = time.perf_counter() - t0
        print(f"  {name:30s}  {N/t:8.1f} fps  {t/N*1000:6.2f} ms/frame")

    def bench_torch(name, fn, tensor, warmup=5):
        with torch.no_grad():
            for i in range(min(warmup, N)):
                fn(tensor[i:i+1])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(N):
                fn(tensor[i:i+1])
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
        print(f"  {name:30s}  {N/t:8.1f} fps  {t/N*1000:6.2f} ms/frame")

    print(f"  Resolution: 1920x1080, {N} frames")
    print("  -- numpy --")
    bench("yuv_to_ictcp_np", yuv_to_ictcp_np)
    bench("rgb_to_ictcp_np", rgb_to_ictcp_np)
    bench("ictcp_to_rgb_np", ictcp_to_rgb_np)
    print("  -- torch JIT (CUDA) --")
    bench_torch("yuv_to_ictcp", yuv_to_ictcp, yuv_t)
    bench_torch("rgb_to_ictcp", rgb_to_ictcp, rgb_t)
    bench_torch("ictcp_to_rgb", ictcp_to_rgb, rgb_t)

    print("\nDone profiling.")


if __name__ == '__main__':
    run_profile()
