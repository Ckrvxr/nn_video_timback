import json
import random
import subprocess
import time
from pathlib import Path

from .probe_utils import (
    sanitize, probe_video, get_video_resolution, _probe_bit_depth,
    make_scale_filter, _random_film_grain, _ffmpeg,
    LR_PIX_WEIGHTS, COLOR_TAGS, AV1_PRESETS, X265_PRESETS, X264_PRESETS, ENCODER_WEIGHTS,
)


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
    sf = args.scale_factor
    src_gx = w // patch_size
    lr_gx = (w // sf) // patch_size
    src_gy = h // patch_size
    lr_gy = (h // sf) // patch_size
    max_gx = min(src_gx, lr_gx)
    max_gy = min(src_gy, lr_gy)
    if max_gy < 1 or max_gx < 1:
        return None

    lr_pix_opts = LR_PIX_WEIGHTS.get(source_bits, LR_PIX_WEIGHTS[10])
    lr_scale = make_scale_filter(w, h, args.scale_factor)

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
                    args) -> str:
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

    sf = args.scale_factor
    if sf > 1:
        hr_vf = f'crop={patch_size * sf}:{patch_size * sf}:{cx}:{cy},scale={patch_size}:{patch_size}'
        lr_vf = f'crop={patch_size}:{patch_size}:{cx // sf}:{cy // sf}'
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
                    '-pix_fmt', 'yuv444p12le',
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
                with open(str(final_dir / 'meta.json'), 'w') as f:
                    json.dump({'num_frames': num_frames}, f, indent=2)
                break
            except RuntimeError:
                if _attempt == 2:
                    raise

    return dir_name


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
