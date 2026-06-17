"""Benchmark throughput of all color conversion functions."""
import sys; sys.path.insert(0, '.')
import time
import numpy as np
import torch
from models.components.color_space import (
    yuv_to_ictcp_np, rgb_to_ictcp_np, ictcp_to_rgb_np,
    yuv_to_ictcp, rgb_to_ictcp, ictcp_to_rgb,
)

H, W, N = 1080, 1920, 100

yuv = np.random.randint(0, 256, (N, H, W, 3)).astype(np.uint8)
rgb = np.random.uniform(0.0, 1.0, (N, H, W, 3)).astype(np.float32)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
yuv_t = (torch.from_numpy(yuv.astype(np.float32)).permute(0,3,1,2).contiguous() / 127.5 - 1.0).to(device)
rgb_t = torch.from_numpy(rgb).permute(0,3,1,2).contiguous().to(device)


def bench(name, fn, warmup=5):
    for i in range(min(warmup, N)):
        fn(yuv[i] if 'YUV' in name else rgb[i])
    t0 = time.perf_counter()
    for i in range(N):
        fn(yuv[i] if 'YUV' in name else rgb[i])
    t = time.perf_counter() - t0
    fps = N / t
    ms = t / N * 1000
    print(f"  {name:30s}  {fps:8.1f} fps  {ms:6.2f} ms/frame")


def bench_torch(name, fn, tensor, warmup=5):
    with torch.no_grad():
        for i in range(min(warmup, N)):
            fn(tensor[i:i+1])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for i in range(N):
            fn(tensor[i:i+1])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.perf_counter() - t0
    fps = N / t
    ms = t / N * 1000
    print(f"  {name:30s}  {fps:8.1f} fps  {ms:6.2f} ms/frame")


if __name__ == '__main__':
    print(f"Resolution: {W}x{H}, {N} frames")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    print("── numpy (float32) ──")
    bench("yuv_to_ictcp_np", yuv_to_ictcp_np)
    bench("rgb_to_ictcp_np", rgb_to_ictcp_np)
    bench("ictcp_to_rgb_np", ictcp_to_rgb_np)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"── torch JIT (float32, {dev}) ──")
    bench_torch("yuv_to_ictcp", yuv_to_ictcp, yuv_t)
    bench_torch("rgb_to_ictcp", rgb_to_ictcp, rgb_t)
    bench_torch("ictcp_to_rgb", ictcp_to_rgb, rgb_t)
