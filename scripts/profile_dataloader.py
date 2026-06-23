import gc
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_WARMUP = 5
N_MEASURE = 30


def profile_decode(dataset_path: str):
    """Measure raw FFV1 decode throughput via load_video_frame_range."""
    from utils.data.preproc_dataset import RawVideoDataset
    from utils.data.video_loader import load_video_frame_range

    ds = RawVideoDataset(dataset_path, frames=9)
    sample = ds.samples[0]
    hr_path = sample['hr_path']

    times = []
    for i in range(N_WARMUP + N_MEASURE):
        t0 = time.perf_counter()
        _ = load_video_frame_range(hr_path, 0, 5)
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(t)

    mean = statistics.mean(times)
    fps = 5.0 / mean
    return fps, mean


def profile_ictcp(dataset_path: str):
    """Measure batch_yuv_to_ictcp conversion throughput."""
    import torch
    from utils.data.ictcp import batch_yuv_to_ictcp
    from utils.data.video_loader import load_video_frame_range

    ds = RawVideoDataset(dataset_path, frames=9)
    sample = ds.samples[0]
    path = sample['hr_path']
    frames = load_video_frame_range(path, 0, 9)
    yuv = torch.from_numpy(np.stack(frames, axis=0))
    device = torch.device('cuda')

    for _ in range(N_WARMUP):
        batch_yuv_to_ictcp(yuv, device)

    times = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        batch_yuv_to_ictcp(yuv, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean = statistics.mean(times)
    return 1.0 / mean, mean


def profile_dataset(dataset_path: str):
    """Measure RawVideoDataset.__getitem__ throughput (single-threaded)."""
    import torch
    from utils.data.preproc_dataset import RawVideoDataset

    ds = RawVideoDataset(dataset_path, frames=9)
    device = torch.device('cuda')

    times = []
    for i in range(N_WARMUP + N_MEASURE):
        idx = i % len(ds)
        t0 = time.perf_counter()
        _ = ds[idx]
        t = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(t)

    mean = statistics.mean(times)
    return 1.0 / mean, mean


def profile_dataloader(dataset_path: str, workers: int, batch_size: int = 4):
    """Measure DataLoader throughput with given worker count."""
    from utils.data.dataset import create_dataloader

    def _run():
        loader = create_dataloader(
            datasets=[dataset_path],
            batch_size=batch_size,
            patch_size=512,
            frames=9,
            workers=workers,
            is_train=True,
            data_type="raw",
            segment_repeat=1,
        )
        n = 0
        t0 = time.perf_counter()
        for batch in loader:
            n += batch['lr_frames'].size(0)
        elapsed = time.perf_counter() - t0
        return n / elapsed if elapsed > 0 else 0

    # Warmup
    _run()
    gc.collect()
    if workers > 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _run()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profile data loading pipeline')
    parser.add_argument('dataset', nargs='?', default='./datasets/synthetic',
                        help='Dataset path (default: ./datasets/synthetic)')
    parser.add_argument('--decode-only', action='store_true',
                        help='Only run decode benchmark')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 1, 2, 4],
                        help='Worker counts to test (default: 0 1 2 4)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for DataLoader (default: 4)')
    args = parser.parse_args()

    dp = args.dataset
    if not Path(dp).exists():
        print(f"Dataset not found: {dp}")
        sys.exit(1)

    print("=" * 60)
    print("  Data Pipeline Profiler")
    print(f"  Dataset: {dp}")
    print("=" * 60)

    # 1. Decode
    print("\n[1/4] FFV1 decode (load_video_frame_range)")
    fps, t = profile_decode(dp)
    print(f"        {fps:>8.1f} frames/sec  (avg {t*1000:.1f}ms per 5-frame batch)")

    # 2. ICtCp
    print("\n[2/4] ICtCp conversion (batch_yuv_to_ictcp)")
    try:
        import torch
        if not torch.cuda.is_available():
            print("        SKIP — no CUDA available")
        else:
            hz, t = profile_ictcp(dp)
            print(f"        {hz:>8.1f} conversions/sec  (avg {t*1000:.1f}ms per 9-frame batch)")
    except Exception as e:
        print(f"        SKIP — {e}")

    # 3. Dataset (single-threaded)
    print("\n[3/4] RawVideoDataset.__getitem__")
    sps, t = profile_dataset(dp)
    print(f"        {sps:>8.1f} samples/sec  (avg {t*1000:.1f}ms per sample)")

    # 4. DataLoader (various workers)
    print("\n[4/4] DataLoader throughput")
    for w in args.workers:
        gc.collect()
        if w > 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            sps = profile_dataloader(dp, workers=w, batch_size=args.batch_size)
            print(f"        workers={w}: {sps:>8.1f} samples/sec  (batch_size={args.batch_size})")
        except Exception as e:
            print(f"        workers={w}: ERROR — {e}")

    print("\n" + "=" * 60)
    print("  Done.")


if __name__ == '__main__':
    main()
