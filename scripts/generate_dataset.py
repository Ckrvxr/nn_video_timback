import argparse
import glob
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import tqdm
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SUPPORTED_EXTS = {'.mp4', '.mkv', '.mov', '.webm', '.avi'}
COLOR_TAGS = [
    '-color_primaries', 'bt2020',
    '-color_trc', 'smpte2084',
    '-colorspace', 'bt2020nc',
    '-color_range', 'pc',
]

LR_PIX_WEIGHTS = {
    12: [('yuv420p12le', 5), ('yuv420p10le', 4), ('yuv420p', 1)],
    10: [('yuv420p10le', 7), ('yuv420p', 3)],
     8: [('yuv420p', 1)],
}

AV1_PRESETS = [8, 9, 10, 11, 12]
X265_PRESETS = ['medium', 'slow', 'fast']
X264_PRESETS = ['medium', 'slow', 'veryslow', 'fast']

ENCODER_WEIGHTS = {
    'av1': 3,
    'h265': 3,
    'h264': 3,
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
        capture_output=True, text=True, timeout=120,
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
                           ['-show_entries', 'format=duration'])
        try:
            if out and out.replace('.', '', 1).lstrip('-').isdigit():
                dur = float(out)
                total_frames = int(round(dur * fps))
        except Exception:
            pass
    return fps, total_frames


def _probe_bit_depth(video_path: Path) -> int:
    out = _run_ffprobe(video_path,
                       ['-show_entries', 'stream=bits_per_raw_sample,pix_fmt'])
    if not out:
        return 10
    parts = out.split(',')
    pix = parts[0]
    bits = parts[1] if len(parts) > 1 else ''
    if bits and bits.isdigit():
        return int(bits)
    for b in ['16', '12', '10', '9']:
        if b in pix:
            return int(b)
    return 8


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


def _plan_segments(video_path: Path, args) -> list[dict] | None:
    name = sanitize(video_path.stem)
    rng = random.Random(hash(f'{name}_{args.seed}'))

    fps, total_frames = probe_video(video_path)
    w, h = get_video_resolution(video_path)
    if total_frames == 0:
        return None

    source_bits = _probe_bit_depth(video_path)
    hr_pix = 'yuv444p' + (f'{source_bits}le' if source_bits > 8 else '')

    trim = max(1, total_frames // 100)
    valid_start = trim
    valid_end = total_frames - trim
    valid_range = valid_end - valid_start
    if valid_range <= 0:
        return None

    num_slices = args.num_slices if args.num_slices is not None else 3
    lr_padding = 90
    seg_frames = args.num_frames + 2 * lr_padding
    if seg_frames > valid_range:
        raise ValueError(
            f'{video_path.name}: {seg_frames} frames needed per segment, '
            f'but only {valid_range} frames available after trimming '
            f'[{trim}]×2 from {total_frames}')

    stride = valid_range / num_slices
    starts = []
    for i in range(num_slices):
        nominal = valid_start + int(round(i * stride))
        jitter = int(round(stride * rng.uniform(-0.2, 0.2)))
        start = max(valid_start, min(valid_end - seg_frames, nominal + jitter))
        starts.append(start)

    print(f'  {video_path.name}: source_bits={source_bits}, HR={hr_pix}')

    patch_size = getattr(args, 'patch_size', 512)
    scale = args.scale
    src_gx = w // patch_size
    src_gy = h // patch_size
    lr_gx = (w // scale) // patch_size
    lr_gy = (h // scale) // patch_size
    max_gx = min(src_gx, lr_gx)
    max_gy = min(src_gy, lr_gy)
    if max_gy < 1 or max_gx < 1:
        return None

    lr_pix_opts = LR_PIX_WEIGHTS.get(source_bits, LR_PIX_WEIGHTS[10])
    lr_scale = make_scale_filter(w, h, args.scale)

    segments = []
    for seg_idx, start_frame in enumerate(starts):
        seg_rng = random.Random(hash(f'{name}_{seg_idx}_{args.seed}'))
        gyi = seg_rng.randint(0, max(1, max_gy) - 1)
        gxi = seg_rng.randint(0, max(1, max_gx) - 1)
        segments.append({
            'video_path': video_path,
            'seg_idx': seg_idx,
            'start_frame': start_frame,
            'gxi': gxi,
            'gyi': gyi,
            'fps': fps,
            'source_bits': source_bits,
            'lr_pix_opts': lr_pix_opts,
            'lr_scale': lr_scale,
        })

    return segments


def process_segment(video_path: Path, seg_idx: int, start_frame: int,
                    gxi: int, gyi: int, fps: float, source_bits: int,
                    lr_pix_opts: list[tuple[str, int]], lr_scale: str | None,
                    args) -> bool:
    name = sanitize(video_path.stem)
    rng = random.Random(hash(f'{name}_{seg_idx}_{args.seed}'))
    seg_name = f'{name}_seg{seg_idx}'

    num_frames = getattr(args, 'num_frames', 30)
    patch_size = getattr(args, 'patch_size', 512)
    lr_padding = 90
    lr_frames = num_frames + 2 * lr_padding
    output_dir = Path(args.output_dir)
    tmp_dir = output_dir / '.tmp'

    lr_pool = [p for p, _ in lr_pix_opts]
    lr_pw = [w for _, w in lr_pix_opts]

    cx = gxi * patch_size
    cy = gyi * patch_size
    center = start_frame + lr_padding

    scale = args.scale
    if scale > 1:
        hr_vf = f'crop={patch_size * scale}:{patch_size * scale}:{cx}:{cy},scale={patch_size}:{patch_size}'
        lr_vf = f'crop={patch_size}:{patch_size}:{cx // scale}:{cy // scale}'
    else:
        hr_vf = f'crop={patch_size}:{patch_size}:{cx}:{cy}'
        lr_vf = f'crop={patch_size}:{patch_size}:{cx}:{cy}'

    encoder_pairs = [(e, ENCODER_WEIGHTS[e]) for e in args.encoders if e in ENCODER_WEIGHTS]
    if not encoder_pairs:
        encoder_pairs = [('av1', 1)]
    enc_pool = [e for e, _ in encoder_pairs]
    enc_w = [w for _, w in encoder_pairs]

    for v in range(args.num_variants):
        for _attempt in range(3):
            try:
                codec = rng.choices(enc_pool, weights=enc_w, k=1)[0]
                lr_pix = rng.choices(lr_pool, weights=lr_pw, k=1)[0]
                seek_time = start_frame / fps

                if codec == 'av1':
                    crf = int(round(max(18, min(61, rng.gauss(30, 6)))))
                    preset = rng.choice(AV1_PRESETS)
                    keyint = rng.choice([32, 64, 96, 128, 160, 300])
                    film_grain = _random_film_grain(rng)
                    dir_name = f'av1_crf{crf}_p{preset}_gop{keyint}_fg{film_grain}_{lr_pix}'
                    lr_cmd = [
                        'ffmpeg', '-y',
                        '-ss', f'{seek_time:.6f}',
                        '-i', str(video_path),
                        '-c:v', 'libsvtav1',
                        '-crf', str(crf), '-g', str(keyint),
                        '-pix_fmt', lr_pix,
                        '-svtav1-params',
                        f'tune=0:preset={preset}:film_grain={film_grain}',
                        '-frames:v', str(lr_frames),
                        *COLOR_TAGS, '-an',
                    ]
                elif codec == 'h265':
                    crf = int(round(max(18, min(61, rng.gauss(29, 6)))))
                    preset = rng.choice(X265_PRESETS)
                    keyint = rng.choice([32, 64, 96, 128, 160, 300])
                    dir_name = f'h265_crf{crf}_p{preset}_gop{keyint}_{lr_pix}'
                    lr_cmd = [
                        'ffmpeg', '-y',
                        '-ss', f'{seek_time:.6f}',
                        '-i', str(video_path),
                        '-c:v', 'libx265',
                        '-preset', preset, '-crf', str(crf),
                        '-pix_fmt', lr_pix,
                        '-x265-params', f'keyint={keyint}:no-open-gop=1',
                        '-frames:v', str(lr_frames),
                        *COLOR_TAGS, '-an',
                    ]
                elif codec == 'h264':
                    crf = int(round(max(18, min(61, rng.gauss(28, 6)))))
                    preset = rng.choice(X264_PRESETS)
                    keyint = rng.choice([32, 64, 96, 128, 160, 300])
                    dir_name = f'h264_crf{crf}_p{preset}_gop{keyint}_{lr_pix}'
                    lr_cmd = [
                        'ffmpeg', '-y',
                        '-ss', f'{seek_time:.6f}',
                        '-i', str(video_path),
                        '-c:v', 'libx264',
                        '-preset', preset, '-crf', str(crf),
                        '-g', str(keyint),
                        '-pix_fmt', lr_pix,
                        '-frames:v', str(lr_frames),
                        *COLOR_TAGS, '-an',
                    ]

                if lr_scale:
                    lr_cmd += ['-vf', lr_scale]
                lr_raw = tmp_dir / f'{seg_name}_{v}.mp4'
                if not lr_raw.exists() or args.no_resume:
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    _ffmpeg(lr_cmd + [str(lr_raw)], f'{codec} {seg_name} v{v}')

                h_start = center / fps
                h_end = (center + num_frames) / fps
                name_dir = f'{int(h_start//60):02d}m{int(h_start%60):02d}s_{int(h_end//60):02d}m{int(h_end%60):02d}s_{dir_name}'
                final_dir = output_dir / name_dir
                if (final_dir / 'meta.json').exists() and not args.no_resume:
                    break
                final_dir.mkdir(parents=True, exist_ok=True)

                _ffmpeg([
                    'ffmpeg', '-y',
                    '-ss', f'{h_start:.6f}',
                    '-i', str(video_path),
                    '-vf', hr_vf,
                    '-c:v', 'ffv1',
                    '-frames:v', str(num_frames),
                    *COLOR_TAGS, '-an',
                    str(final_dir / 'HR.mkv'),
                ], f'FFV1 HR {seg_name}')

                _ffmpeg([
                    'ffmpeg', '-y',
                    '-ss', f'{lr_padding / fps:.6f}',
                    '-i', str(lr_raw),
                    '-vf', lr_vf,
                    '-c:v', 'ffv1',
                    '-frames:v', str(num_frames),
                    *COLOR_TAGS, '-an',
                    str(final_dir / 'LR.mkv'),
                ], f'FFV1 LR {seg_name} v{v}')
                break  # all ffmpeg calls succeeded
            except RuntimeError:
                if _attempt == 2:
                    raise  # give up after 3 attempts

    return True


def count_segments(video_path: Path, args) -> int:
    _, total_frames = probe_video(video_path)
    if total_frames == 0:
        return 0
    trim = max(1, total_frames // 100)
    valid_range = total_frames - 2 * trim
    if valid_range <= 0:
        return 0
    num_slices = args.num_slices if args.num_slices is not None else 3
    seg_frames = args.num_frames + 2 * 90
    total_needed = num_slices * seg_frames
    if total_needed > valid_range:
        return 0
    return num_slices


def _batch_yuv_to_ictcp_crop_gpu(yuv_frames: list, crop_info: tuple,
                                 batch_size: int = 8, patch_size: int = 512) -> np.ndarray:
    """Decode YUV in batches, GPU convert to ICtCp, crop 512², accumulate.

    crop_info = (crop_y, crop_x, num_frames)
    Returns (num_frames, 3, 512, 512) float16.
    """
    import torch
    crop_y, crop_x, total_n = crop_info
    out = np.empty((total_n, 3, patch_size, patch_size), dtype=np.float16)

    for start in range(0, total_n, batch_size):
        end = min(start + batch_size, total_n)
        chunk = yuv_frames[start:end]
        batch = np.stack(chunk, axis=0)

        bits = 8
        if batch.dtype == np.uint16:
            max_val = int(batch.max())
            bits = 12 if max_val > 1023 else 10
        peak = float((1 << bits) - 1)
        center = float(1 << (bits - 1))

        t = torch.from_numpy(batch.astype(np.float32, copy=False)).cuda()
        t = t.permute(0, 3, 1, 2).contiguous()
        t[:, 0:1] = t[:, 0:1] / peak * 255.0
        t[:, 1:] = (t[:, 1:] - center) / peak * 255.0 + 128.0
        t = t / 127.5 - 1.0

        from models.components.color_space import yuv_to_ictcp
        with torch.no_grad():
            ictcp = yuv_to_ictcp(t).cpu().numpy().astype(np.float16)

        out[start:end] = ictcp[:, :, crop_y:crop_y + patch_size, crop_x:crop_x + patch_size]

    return out


def _preprocess_one_segment(seg_dir: Path, crop_info: tuple,
                            num_frames: int = 30, patch_size: int = 512):
    """Read HR.mkv/LR.mp4 from seg_dir → ICtCp → hr.npy/lr.npy/meta.json.

    Files are pre-trimmed to the exact window, so always read from frame 0.
    """
    from utils.data.video_loader import load_video_frame_range

    yuv = load_video_frame_range(str(seg_dir / 'HR.mkv'), 0, num_frames)
    hr_p = _batch_yuv_to_ictcp_crop_gpu(yuv, crop_info[:2] + (num_frames,))
    np.save(str(seg_dir / 'hr.npy'), hr_p)
    del yuv, hr_p

    lr_path = seg_dir / 'LR.mkv'
    if lr_path.exists():
        yuv = load_video_frame_range(str(lr_path), 0, num_frames)
        lr_p = _batch_yuv_to_ictcp_crop_gpu(yuv, crop_info[:2] + (num_frames,))
        np.save(str(seg_dir / 'lr.npy'), lr_p)
        del yuv, lr_p

    with open(str(seg_dir / 'meta.json'), 'w') as f:
        json.dump({'window_start': crop_info[2], 'grid_y': crop_info[3],
                   'grid_x': crop_info[4], 'gy': crop_info[5], 'gx': crop_info[6],
                   'num_frames': num_frames}, f, indent=2)


def preprocess_dataset(data_dir: str, num_frames: int = 30, patch_size: int = 512):
    """Walk named dirs, ICtCp HR.mkv+LR.mp4 → hr.npy/lr.npy/meta.json in-place."""
    from utils.data.video_loader import probe_resolution, probe_frame_count
    data_path = Path(data_dir)
    dirs = sorted(d for d in data_path.iterdir()
                  if d.is_dir() and d.name != '.tmp')

    import tqdm
    for seg_dir in tqdm.tqdm(dirs, desc='Preprocessing', unit='seg'):
        if (seg_dir / 'meta.json').exists():
            continue
        hr_path = seg_dir / 'HR.mkv'
        if not hr_path.exists():
            continue

        h, w = probe_resolution(str(hr_path))
        gy, gx = h // patch_size, w // patch_size

        # Files are pre-cropped; meta.json for record only
        rng = random.Random(hash(seg_dir.name))
        gyi = rng.randint(0, max(1, gy) - 1)
        gxi = rng.randint(0, max(1, gx) - 1)
        crop_info = (0, 0, 0, gyi, gxi, gy, gx)

        _preprocess_one_segment(seg_dir, crop_info, num_frames, patch_size)


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
    parser.add_argument('--slice-frames', type=int, default=30,
                        help='Frames per output window (default: 30)')
    parser.add_argument('--num-slices', type=int, default=None,
                        help='Number of segments per video (default: 3)')
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
                    future.result()
                    pbar.set_postfix_str('')
                except Exception as e:
                    pbar.set_postfix_str(f'seg{s["seg_idx"]}: {e}')
                pbar.update(1)

    shutil.rmtree(str(Path(args.output_dir) / '.tmp'), ignore_errors=True)

    preprocess_dataset(args.output_dir)

    elapsed = time.perf_counter() - t0
    print(f'\nDone in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
