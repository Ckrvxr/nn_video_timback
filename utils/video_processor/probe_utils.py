import random
import re
import subprocess
from pathlib import Path


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
X264_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow', 'placebo']

ENCODER_WEIGHTS = {'av1': 0.5, 'h265': 0.3, 'h264': 0.2}


def sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def discover_inputs(inputs: list[str]) -> list[Path]:
    videos = []
    for inp in inputs:
        p = Path(inp).expanduser().resolve()
        if p.is_file():
            videos.append(p)
        elif p.is_dir():
            videos.extend(sorted(p.rglob('*')))
        else:
            for f in Path('.').glob(inp):
                videos.append(f)
    supported = ('.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf', '.ts', '.mts')
    return sorted(set(v for v in videos if v.suffix.lower() in supported and v.is_file()))


def probe_video(video_path: Path) -> tuple[float, int]:
    result = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate,nb_frames',
         '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True, timeout=30)
    parts = result.stdout.strip().split(',')
    if len(parts) >= 2 and parts[1].isdigit():
        fps_str = parts[0]
        if '/' in fps_str:
            n, d = map(int, fps_str.split('/'))
            fps = n / d if d else 30.0
        else:
            fps = float(fps_str)
        return fps, int(parts[1])
    # Fallback: probe duration and approximate
    r2 = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-show_entries', 'format=duration',
         '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True, timeout=30)
    dur = float(r2.stdout.strip())
    fps = 30.0
    return fps, int(dur * fps)


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height',
         '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True, timeout=30)
    parts = result.stdout.strip().split(',')
    return int(parts[0]), int(parts[1])


def _probe_bit_depth(video_path: Path) -> int:
    result = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=bits_per_raw_sample,pix_fmt',
         '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True, timeout=30)
    parts = result.stdout.strip().split(',')
    if parts[0] and parts[0] != 'N/A':
        return int(parts[0])
    pix_fmt = parts[1] if len(parts) > 1 else ''
    if '12' in pix_fmt:
        return 12
    if '10' in pix_fmt:
        return 10
    return 8


def _random_film_grain(rng: random.Random) -> int:
    p = rng.random()
    if p < 0.60:
        return 0
    if p < 0.85:
        return rng.randint(1, 15)
    return rng.randint(16, 50)


def make_scale_filter(w: int, h: int, scale_factor: int) -> str | None:
    if scale_factor <= 1:
        return None
    out_w = w // scale_factor
    out_h = h // scale_factor
    return f'scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int'


def color_conversion_filter(video_path: Path) -> tuple[str | None, str]:
    # Probe source color space and return appropriate conversion
    result = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=color_space,color_primaries,color_trc',
         '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True, timeout=15)
    parts = result.stdout.strip().split(',')
    csp = parts[0] if len(parts) > 0 else ''
    # Default to bt2020nc conversion
    return ('bt2020nc', 'bt2020pq')


def _ffmpeg(cmd: list[str], desc: str, timeout: int = 7200):
    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors='replace')[-500:]
        raise RuntimeError(f'{desc} failed:\n{stderr}')
