#! /usr/bin/env python3
"""Pre-decode all video frames to Blosc cache for zero-decoding training.

Usage:
    pixi run python scripts/precache_frames.py --datasets ./data/datasets/realisvideo-4k
    pixi run python scripts/precache_frames.py --datasets ./data/datasets/realisvideo-4k --variants HR av1_crf32
"""
import argparse
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Console

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
    parser = argparse.ArgumentParser(description='Pre-cache video as .blp files')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--variants', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    console = Console()

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
            console.print(f'[green]{ds_root.name}[/] — all cached.')
            continue

        n_workers = min(args.workers, n_total)
        start = time.perf_counter()
        done = 0
        total_bytes = 0
        total_raw = 0
        recent = deque(maxlen=4)

        progress = Progress(
            TextColumn('[cyan]{task.description}[/]'),
            BarColumn(),
            TextColumn('{task.completed}/{task.total}'),
            TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
            TimeElapsedColumn(), TextColumn('•'), TimeRemainingColumn(),
            TextColumn('• {task.fields[r]:.1f}x •{task.fields[g]}'),
        )
        task = progress.add_task(ds_root.name, total=n_total, r=1.0, g='0GB')

        def build_table():
            t = Table(box=None, show_header=True, header_style='dim', padding=(0, 1, 0, 1))
            t.add_column('File', no_wrap=True)
            t.add_column('Orig', justify='right')
            t.add_column('Comp', justify='right')
            t.add_column('Ratio', justify='right')
            t.add_column('Disk', justify='right')
            t.add_column('ETA', justify='right')
            for entry in recent:
                fname, raw_b, comp_b, _, _, _, _, cumul = entry
                r = raw_b / comp_b if comp_b else 0
                t.add_row(f'{fname}', f'{raw_b/1e6:.0f}M', f'{comp_b/1e6:.0f}M',
                          f'{r:.1f}x', f'{cumul/1e9:.1f}G', '')
            return t

        def update():
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed else 0
            gb = total_bytes / (1 << 30)
            ratio = total_raw / total_bytes if total_bytes else 0
            progress.update(task, completed=done, r=ratio, g=f'{gb:.1f}GB')
            t = build_table()
            if done == n_total:
                est = gb
                eta_str = ''
            elif done:
                est = gb / done * n_total
                eta_sec = (n_total - done) / rate if rate else 0
                eta_str = fmt_time(eta_sec)
            else:
                est = 0
                eta_str = ''
            t.add_row('[dim]──[/]'*5)
            r = total_raw / total_bytes if total_bytes else 0
            t.add_row(f'[bold]{done}/{n_total}[/]',
                      f'[bold]{total_raw/1e6:.0f}M[/]', f'[bold]{total_bytes/1e6:.0f}M[/]',
                      f'[bold]{r:.1f}x[/]', f'[bold]~{est:.0f}G[/]',
                      f'{eta_str}')
            return t

        def report(path, status, n, comp_b, raw_b):
            nonlocal done, total_bytes, total_raw
            done += 1
            total_bytes += comp_b
            total_raw += raw_b
            recent.append((Path(path).name, raw_b, comp_b, status, n, path, False, total_bytes))

        try:
            with Live(update(), refresh_per_second=2.5, transient=True, console=console) as live:
                if n_workers > 1:
                    with ProcessPoolExecutor(max_workers=n_workers) as pool:
                        fut_map = {pool.submit(cache_one_video, t): t for t in todos}
                        for fut in as_completed(fut_map):
                            report(*fut.result())
                            live.update(update())
                else:
                    for todo in todos:
                        report(*cache_one_video(todo))
                        live.update(update())
        except KeyboardInterrupt:
            console.print(f'\n[red]Interrupted[/] — {done} videos, {total_bytes/(1<<30):.1f}GB')

        if done:
            elapsed = time.perf_counter() - start
            savings = (total_raw - total_bytes) / total_raw * 100 if total_raw else 0
            console.print(f'[green]Done[/]  {done} videos  {fmt_time(elapsed)}  '
                          f'{total_bytes/(1<<30):.1f}GB  ({savings:.0f}% saved)')


if __name__ == '__main__':
    main()
