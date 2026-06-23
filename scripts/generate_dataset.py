import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.data.probe_utils import (
    sanitize, discover_inputs, probe_video, get_video_resolution,
    _probe_bit_depth, color_conversion_filter, COLOR_TAGS, ENCODER_WEIGHTS,
)
from utils.data.encode_worker import _plan_segments, process_segment, count_segments
from utils.data.dataset_prep import preprocess_dataset


def main():
    parser = argparse.ArgumentParser(
        description='Generate AV1 compressed video dataset with random variants')
    parser.add_argument('-i', '--input', required=True, action='append',
                        help='Input video file, directory, or glob pattern')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: ./datasets/{name})')
    parser.add_argument('--name', type=str, default='dataset',
                        help='Dataset name (default: dataset)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Downsampling factor (<1 = multiply, >1 = divide; 0.5 = 2x down)')
    parser.add_argument('--slice-frames', type=int, default=30,
                        help='Frames per output window (default: 30)')
    parser.add_argument('--num-slices', type=int, default=50,
                        help='Number of segments per video (default: 50)')
    parser.add_argument('--num-frames', type=int, default=30,
                        help='Patch window size in frames (default: 30)')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Spatial patch size in pixels (default: 512)')
    parser.add_argument('--num-variants', type=int, default=1,
                        help='Number of LR variants per segment (default: 1)')
    parser.add_argument('--encoders', type=str, nargs='+',
                        default=['av1', 'h265', 'h264'],
                        choices=['av1', 'h265', 'h264'],
                        help='Encoders to randomly pick from (default: all)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel videos (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--colorspace', type=str, default='bt2020pq',
                        choices=['bt2020pq', 'passthrough'],
                        help='Target color space (default: bt2020pq)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Force re-encode all segments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without encoding')
    parser.add_argument('--preprocess', action='store_true',
                        help='Generate .npy cache (default: skip, use CPU real-time decoding)')
    args = parser.parse_args()
    args.scale_factor = int(round(1 / args.scale)) if args.scale < 1 else int(args.scale)

    args.name = sanitize(args.name)
    if args.output_dir is None:
        args.output_dir = f'./datasets/{args.name}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    videos = discover_inputs(args.input)
    if not videos:
        print(f'No supported video files found in: {args.input}')
        sys.exit(1)

    print(f'Found {len(videos)} video(s)')
    if args.dry_run:
        print(f'Output: {args.output_dir}')
        n_slices = args.num_slices if args.num_slices is not None else 3
        print(f'Scale: {args.scale} ({args.scale_factor}x down), Segments: {n_slices}×{args.slice_frames}f, '
              f'Variants: {args.num_variants}')
        print(f'Encoders: {", ".join(args.encoders)}')
        print(f'Colorspace: {args.colorspace}, Seed: {args.seed}')
        for v in videos:
            print(f'  {v}')
        return

    t0 = time.perf_counter()

    all_segments = []
    for v in videos:
        try:
            segs = _plan_segments(v, args)
            if segs:
                all_segments.extend(segs)
        except ValueError as e:
            print(f'  {v.name}: {e}')

    if not all_segments:
        print('No segments to generate')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        with tqdm.tqdm(total=len(all_segments), desc='Generating', unit='seg') as pbar:
            futures = {}
            for s in all_segments:
                f = executor.submit(process_segment,
                    s['video_path'], s['seg_idx'], s['start_frame'],
                    s['gxi'], s['gyi'], s['fps'], s['source_bits'],
                    s['lr_pix_opts'], s['lr_scale'], args)
                futures[f] = s

            for future in as_completed(futures):
                s = futures[future]
                try:
                    info = future.result()
                    pbar.set_postfix_str(f'{s["video_path"].stem}_seg{s["seg_idx"]}: {info}')
                except Exception as e:
                    pbar.set_postfix_str(f'seg{s["seg_idx"]}: {e}')
                pbar.update(1)

    shutil.rmtree(str(Path(args.output_dir) / '.tmp'), ignore_errors=True)

    if args.preprocess:
        preprocess_dataset(args.output_dir)

    elapsed = time.perf_counter() - t0
    print(f'\nDone in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
