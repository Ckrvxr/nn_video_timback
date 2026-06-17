import argparse
import glob
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED_EXTS = {'.mp4', '.mkv', '.mov', '.webm', '.avi'}
PIX_FMT = 'yuv444p10le'
COLOR_TAGS = [
    '-color_primaries', 'bt2020',
    '-color_trc', 'smpte2084',
    '-colorspace', 'bt2020nc',
    '-color_range', 'pc',
]

AV1_PRESETS = [8, 9, 10, 11, 12]
X265_PRESETS = ['medium', 'slow', 'veryslow', 'fast']
X264_PRESETS = ['medium', 'slow', 'veryslow', 'fast']
VP9_CPU_USED = [0, 1, 2, 3, 4]

ENCODER_WEIGHTS = {
    'av1': 3,
    'h265': 3,
    'h264': 3,
    'vp9': 1,
}


def sanitize(name: str) -> str:
    return re.sub(r'[^\w.\-]', '_', name)


def discover_inputs(paths: list[str]) -> list[Path]:
    files = []
    seen = set()
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            if pp.suffix.lower() in SUPPORTED_EXTS and pp not in seen:
                files.append(pp)
                seen.add(pp)
        elif pp.is_dir():
            for ext in SUPPORTED_EXTS:
                for f in sorted(pp.rglob(f'*{ext}')):
                    if f not in seen:
                        files.append(f)
                        seen.add(f)
        else:
            matched = sorted(Path(fp) for fp in glob.glob(p, recursive=True))
            for f in matched:
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and f not in seen:
                    files.append(f)
                    seen.add(f)
    return files


def _run_ffprobe(video_path: Path, args: list[str]) -> str:
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-of', 'csv=p=0'] + args + [str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip()


def probe_video(video_path: Path) -> tuple[float, int]:
    fps = 30.0
    total_frames = 0
    out = _run_ffprobe(video_path,
                       ['-show_entries', 'stream=r_frame_rate,nb_frames'])
    try:
        if out:
            parts = out.split(',')
            fr = parts[0]
            if '/' in fr:
                n, d = fr.split('/')
                fps = float(n) / float(d) if float(d) > 0 else 30.0
            else:
                fps = float(fr) if fr else 30.0
            if len(parts) > 1 and parts[1] and parts[1].isdigit():
                total_frames = int(parts[1])
    except Exception:
        fps = 30.0

    if total_frames == 0:
        out = _run_ffprobe(video_path,
                           ['-count_packets',
                            '-show_entries', 'stream=nb_read_packets'])
        try:
            out = out.rstrip(',')
            if out and out.isdigit():
                total_frames = int(out)
        except Exception:
            pass
    return fps, total_frames


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    out = _run_ffprobe(video_path,
                       ['-show_entries', 'stream=width,height'])
    try:
        if out:
            parts = out.split(',')
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 1920, 1080


def _probe_color(video_path: Path) -> dict:
    out = _run_ffprobe(video_path,
                       ['-show_entries',
                        'stream=color_primaries,color_trc,color_space,color_range'])
    parts = out.split(',') if out else [''] * 4
    return {
        'primaries': parts[0] if len(parts) > 0 else '',
        'trc': parts[1] if len(parts) > 1 else '',
        'space': parts[2] if len(parts) > 2 else '',
        'range': parts[3] if len(parts) > 3 else '',
    }


def _have_zscale() -> bool:
    try:
        r = subprocess.run(['ffmpeg', '-filters'], capture_output=True,
                           text=True, timeout=10)
        return 'zscale' in r.stdout
    except Exception:
        return False


def color_conversion_filter(video_path: Path) -> tuple[str | None, str]:
    color = _probe_color(video_path)
    p, t, s = color['primaries'], color['trc'], color['space']
    p_ok = 'bt2020' in p
    t_ok = 'smpte2084' in t
    s_ok = 'bt2020nc' in s
    if p_ok and t_ok and s_ok:
        return None, f'already BT.2020 PQ ({p}/{t}/{s})'
    if not p_ok and (not p or p == 'unspecified'):
        return None, f'color_primaries unspecified ({p}); skipping conversion'
    if not t_ok and (not t or t == 'unspecified'):
        return None, f'color_trc unspecified ({t}); skipping conversion'
    if not s_ok and (not s or s == 'unspecified'):
        return None, f'color_space unspecified ({s}); skipping conversion'
    if not _have_zscale():
        raise RuntimeError(
            f'Color conversion needed but zscale (libzimg) not available.\n'
            f'  Input: {p}/{t}/{s}\n'
            f'  Install ffmpeg with --enable-libzimg or use --colorspace passthrough')
    return (
        'zscale=transfer=smpte2084:primaries=bt2020:matrix=bt2020nc:range=full',
        f'converting {p}/{t}/{s} → bt2020/smpte2084/bt2020nc',
    )


def make_scale_filter(w: int, h: int, scale: int) -> str | None:
    if scale <= 1:
        return None
    out_w = w // scale
    out_h = h // scale
    return f'scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int'


def _random_film_grain(rng: random.Random) -> int:
    p = rng.random()
    if p < 0.60:
        return 0
    if p < 0.85:
        return rng.randint(1, 15)
    return rng.randint(16, 50)


def _ffmpeg(cmd: list[str], desc: str, timeout: int = 7200):
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f'{desc} failed: ' + result.stderr.decode(errors='replace')[-500:])


def process_video(video_path: Path, args) -> int:
    name = sanitize(video_path.stem)
    rng = random.Random(hash(f'{name}_{args.seed}'))

    fps, total_frames = probe_video(video_path)
    w, h = get_video_resolution(video_path)
    if total_frames == 0:
        return 0

    lr_scale = make_scale_filter(w, h, args.scale)
    output_dir = Path(args.output_dir)

    trim = max(1, total_frames // 100)
    valid_start = trim
    valid_end = total_frames - trim
    valid_range = valid_end - valid_start

    if valid_range <= 0:
        return 0

    num_slices = args.num_slices if args.num_slices is not None else 3
    total_needed = num_slices * args.slice_frames
    if total_needed > valid_range:
        raise ValueError(
            f'{video_path.name}: {num_slices}×{args.slice_frames}={total_needed} frames needed, '
            f'but only {valid_range} frames available after trimming '
            f'[{trim}]×2 from {total_frames}')

    remaining_gap = valid_range - total_needed
    cuts = sorted(rng.random() for _ in range(num_slices))
    props = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, num_slices)] + [1 - cuts[-1]]
    gaps = [int(round(remaining_gap * p)) for p in props]
    diff = remaining_gap - sum(gaps)
    if diff != 0:
        gaps[rng.randint(0, num_slices)] += diff

    starts = []
    pos = valid_start + gaps[0]
    for i in range(num_slices):
        starts.append(pos)
        pos += args.slice_frames + gaps[i + 1]

    # Determine color conversion (once per video)
    if args.colorspace == 'passthrough':
        conv_filter = None
        conv_desc = 'passthrough mode, no conversion'
    else:
        conv_filter, conv_desc = color_conversion_filter(video_path)
    print(f'  {video_path.name}: {conv_desc}')

    hr_dir = output_dir / 'HR'
    segments_created = 0

    for seg_idx, start_frame in enumerate(starts):
        seg_name = f'{name}_seg{seg_idx}'
        hr_path = hr_dir / f'{seg_name}.mp4'

        if hr_path.exists() and not args.no_resume:
            segments_created += 1
            continue

        # Build filter chain: select + optional color conversion
        select = f'select=between(n\\,{start_frame}\\,{start_frame + args.slice_frames - 1})'
        vf_parts = [select]
        if conv_filter:
            vf_parts.append(conv_filter)
        vf_filter = ','.join(vf_parts)

        # ── HR: lossless FFV1, BT.2020 PQ yuv444p10le ────────────────
        hr_dir.mkdir(parents=True, exist_ok=True)
        hr_cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-vf', vf_filter,
            '-vsync', '0',
            '-c:v', 'ffv1',
            '-pix_fmt', PIX_FMT,
            *COLOR_TAGS,
            '-an',
            '-frames:v', str(args.slice_frames),
            str(hr_path),
        ]
        _ffmpeg(hr_cmd, f'HR {seg_name}', timeout=300)

        # ── LR: re-encode HR with random codec + params ────────────
        encoder_pairs = [(e, ENCODER_WEIGHTS[e]) for e in args.encoders if e in ENCODER_WEIGHTS]
        if not encoder_pairs:
            encoder_pairs = [('av1', 1)]
        enc_pool = [e for e, _ in encoder_pairs]
        enc_w = [w for _, w in encoder_pairs]

        for v in range(args.num_variants):
            codec = rng.choices(enc_pool, weights=enc_w, k=1)[0]

            if codec == 'av1':
                crf = rng.randint(30, 63)
                preset = rng.choice(AV1_PRESETS)
                keyint = rng.choice([32, 64, 96, 128, 160, 300])
                film_grain = _random_film_grain(rng)
                dir_name = f'av1_crf{crf}_p{preset}_gop{keyint}_fg{film_grain}'
                lr_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(hr_path),
                    '-c:v', 'libsvtav1',
                    '-crf', str(crf),
                    '-g', str(keyint),
                    '-pix_fmt', PIX_FMT,
                    '-svtav1-params',
                    f'tune=0:preset={preset}:film_grain={film_grain}',
                    *COLOR_TAGS, '-an',
                ]

            elif codec == 'h265':
                crf = rng.randint(18, 40)
                preset = rng.choice(X265_PRESETS)
                keyint = rng.choice([32, 64, 96, 128, 160, 300])
                dir_name = f'h265_crf{crf}_p{preset}_gop{keyint}'
                lr_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(hr_path),
                    '-c:v', 'libx265',
                    '-preset', preset,
                    '-crf', str(crf),
                    '-pix_fmt', PIX_FMT,
                    '-x265-params', f'keyint={keyint}:no-open-gop=1',
                    *COLOR_TAGS, '-an',
                ]

            elif codec == 'h264':
                crf = rng.randint(18, 40)
                preset = rng.choice(X264_PRESETS)
                keyint = rng.choice([32, 64, 96, 128, 160, 300])
                dir_name = f'h264_crf{crf}_p{preset}_gop{keyint}'
                lr_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(hr_path),
                    '-c:v', 'libx264',
                    '-preset', preset,
                    '-crf', str(crf),
                    '-g', str(keyint),
                    '-pix_fmt', PIX_FMT,
                    *COLOR_TAGS, '-an',
                ]

            elif codec == 'vp9':
                crf = rng.randint(15, 40)
                preset = rng.choice(VP9_CPU_USED)
                keyint = rng.choice([32, 64, 96, 128, 160, 300])
                dir_name = f'vp9_crf{crf}_cpu{preset}_gop{keyint}'
                lr_cmd = [
                    'ffmpeg', '-y',
                    '-i', str(hr_path),
                    '-c:v', 'libvpx-vp9',
                    '-crf', str(crf),
                    '-g', str(keyint),
                    '-pix_fmt', PIX_FMT,
                    '-deadline', 'good',
                    '-cpu-used', str(preset),
                    *COLOR_TAGS, '-an',
                ]

            variant_dir = output_dir / dir_name
            lr_path = variant_dir / f'{seg_name}.mp4'
            if lr_path.exists() and not args.no_resume:
                continue
            if lr_scale:
                lr_cmd += ['-vf', lr_scale]
            lr_cmd += [str(lr_path)]
            _ffmpeg(lr_cmd, f'{codec} {seg_name} v{v}')

        segments_created += 1

    return segments_created


def count_segments(video_path: Path, args) -> int:
    _, total_frames = probe_video(video_path)
    if total_frames == 0:
        return 0
    trim = max(1, total_frames // 100)
    valid_range = total_frames - 2 * trim
    if valid_range <= 0:
        return 0
    num_slices = args.num_slices if args.num_slices is not None else 3
    total_needed = num_slices * args.slice_frames
    if total_needed > valid_range:
        return 0
    return num_slices


def main():
    parser = argparse.ArgumentParser(
        description='Generate AV1 compressed video dataset with random variants')
    parser.add_argument('-i', '--input', required=True, action='append',
                        help='Input video file, directory, or glob pattern')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory (default: ./data/{name})')
    parser.add_argument('--name', type=str, default='dataset',
                        help='Dataset name (default: dataset)')
    parser.add_argument('--scale', type=int, default=1,
                        help='Downsampling factor (2 = half resolution)')
    parser.add_argument('--slice-frames', type=int, default=90,
                        help='Frames per segment (default: 90)')
    parser.add_argument('--num-slices', type=int, default=None,
                        help='Number of segments per video (default: 3)')
    parser.add_argument('--num-variants', type=int, default=1,
                        help='Number of LR variants per segment (default: 1)')
    parser.add_argument('--encoders', type=str, nargs='+',
                        default=['av1', 'h265', 'h264', 'vp9'],
                        choices=['av1', 'h265', 'h264', 'vp9'],
                        help='Encoders to randomly pick from (default: all)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel videos (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--colorspace', type=str, default='bt2020pq',
                        choices=['bt2020pq', 'passthrough'],
                        help='Target color space (default: bt2020pq)')
    parser.add_argument('--no-resume', action='store_true',
                        help='Force re-encode all segments')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without encoding')
    args = parser.parse_args()

    args.name = sanitize(args.name)
    if args.output_dir is None:
        args.output_dir = f'./data/{args.name}'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    videos = discover_inputs(args.input)
    if not videos:
        print(f'No supported video files found in: {args.input}')
        sys.exit(1)

    print(f'Found {len(videos)} video(s)')
    if args.dry_run:
        print(f'Output: {args.output_dir}')
        n_slices = args.num_slices if args.num_slices is not None else 3
        print(f'Scale: {args.scale}x, Segments: {n_slices}×{args.slice_frames}f, '
              f'Variants: {args.num_variants}')
        print(f'Encoders: {", ".join(args.encoders)}')
        print(f'Colorspace: {args.colorspace}, Seed: {args.seed}')
        for v in videos:
            print(f'  {v}')
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.perf_counter()

    total_segments = sum(count_segments(v, args) for v in videos)
    if total_segments == 0:
        print('No segments to generate (all videos too short or 0 frames)')
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_video, v, args): v for v in videos}
        with tqdm.tqdm(total=total_segments, desc='Generating', unit='seg') as pbar:
            for future in as_completed(futures):
                v = futures[future]
                try:
                    segs = future.result()
                    if segs > 0:
                        pbar.set_postfix_str(f'{v.stem}')
                    pbar.update(segs)
                except Exception as e:
                    pbar.set_postfix_str(f'{v.stem}: FAIL - {e}')

    elapsed = time.perf_counter() - t0
    print(f'\nDone in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
