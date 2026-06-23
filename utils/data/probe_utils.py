import glob
import random
import re
import subprocess
from pathlib import Path


SUPPORTED_EXTS = {'.mp4', '.mkv', '.mov', '.webm', '.avi'}
COLOR_TAGS = [
    '-color_primaries', 'bt2020',
    '-color_trc', 'smpte2084',
    '-colorspace', 'bt2020nc',
    '-color_range', 'pc',
]

LR_PIX_WEIGHTS = {
    12: [('yuv420p12le', 5), ('yuv444p12le', 0.1), ('yuv420p10le', 4), ('yuv420p', 1)],
    10: [('yuv420p10le', 7), ('yuv444p10le', 0.15), ('yuv420p', 3)],
     8: [('yuv420p', 1), ('yuv444p', 0.02)],
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


def make_scale_filter(w: int, h: int, scale_factor: int) -> str | None:
    if scale_factor <= 1:
        return None
    out_w = w // scale_factor
    out_h = h // scale_factor
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
