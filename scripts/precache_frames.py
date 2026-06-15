#! /usr/bin/env python3
"""Pre-decode all video frames to Blosc cache for zero-decoding training.

Usage:
    pixi run python scripts/precache_frames.py --datasets ./data/datasets/realisvideo-4k
    pixi run python scripts/precache_frames.py --datasets ./data/datasets/realisvideo-4k --variants HR av1_crf32
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from utils.blosc_cache import BloscCache
from utils.video_loader import load_video_frames_raw, probe_frame_count

ASSUMED_HW = (2160, 3840)


def cache_one_video(args: tuple) -> tuple:
    video_path, cache_root = args
    cache = BloscCache(cache_root)
    cp = cache.cache_path(video_path)
    if cp.exists():
        n = probe_frame_count(str(video_path))
        raw = n * ASSUMED_HW[0] * ASSUMED_HW[1] * 3
        return (str(video_path), 'skip', n, 0, raw)
    raw_frames = load_video_frames_raw(str(video_path))
    h, w = raw_frames[0][0].shape[:2]
    raw = len(raw_frames) * h * w * 3
    cache.put_raw(video_path, raw_frames)
    size = cp.stat().st_size
    return (str(video_path), 'cached', len(raw_frames), size, raw)


def fmt_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def main():
    from utils.console import console

    parser = argparse.ArgumentParser(description='Pre-cache video as .blp files')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--variants', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    for ds_root_str in args.datasets:
        ds_root = Path(ds_root_str)
        cache_root = ds_root / 'cache'

        if args.variants:
            variant_dirs = [ds_root / v for v in args.variants]
        else:
            variant_dirs = sorted(
                d for d in ds_root.iterdir()
                if d.is_dir() and d.name != 'cache'
            )

        todos = []
        for vd in variant_dirs:
            for mp4 in sorted(vd.glob('*.mp4')):
                cp = BloscCache(cache_root).cache_path(mp4)
                if not cp.exists():
                    todos.append((str(mp4), str(cache_root)))

        n_total = len(todos)
        if not todos:
            console.success(f'{ds_root.name} — all cached.')
            continue

        n_workers = min(args.workers, n_total)
        start = time.perf_counter()
        done = 0
        total_bytes = 0
        total_raw = 0

        pbar = tqdm(total=n_total, desc=ds_root.name, unit='file')
        try:
            if n_workers > 1:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    fut_map = {pool.submit(cache_one_video, t): t for t in todos}
                    for fut in as_completed(fut_map):
                        path, status, n, comp_b, raw_b = fut.result()
                        done += 1
                        total_bytes += comp_b
                        total_raw += raw_b
                        pbar.update(1)
                        pbar.set_postfix(last=Path(path).name, status=status)
            else:
                for todo in todos:
                    path, status, n, comp_b, raw_b = cache_one_video(todo)
                    done += 1
                    total_bytes += comp_b
                    total_raw += raw_b
                    pbar.update(1)
                    pbar.set_postfix(last=Path(path).name, status=status)
        except KeyboardInterrupt:
            console.warning(f'Interrupted — {done} videos, {total_bytes/(1<<30):.1f}GB')
            pbar.close()
            return

        pbar.close()
        if done:
            elapsed = time.perf_counter() - start
            savings = (total_raw - total_bytes) / total_raw * 100 if total_raw else 0
            console.success(f'{done} videos  {fmt_time(elapsed)}  '
                            f'{total_bytes/(1<<30):.1f}GB  ({savings:.0f}% saved)')


if __name__ == '__main__':
    main()
