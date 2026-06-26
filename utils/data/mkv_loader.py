"""MKV video batch loader — ffmpeg pipe + numpy colour conversion."""

import json
import subprocess
from pathlib import Path

import numpy as np

from utils.colorspace.color_space import yuv_to_rgb_linear_np


def _yuv_to_rgb_cpu(yuv: np.ndarray) -> np.ndarray:
    """Convert YUV [N, H, W, 3] uint16 (yuv444p12le) → linear RGB float32 on CPU."""
    return yuv_to_rgb_linear_np(yuv, bits=12)


# ── FFmpeg pipe ───────────────────────────────────────────────────────

def _ffmpeg_to_yuv(path: Path) -> np.ndarray:
    """Decode MKV to raw YUV uint16 array [N, H, W, 3] via ffmpeg pipe.

    Always outputs yuv444p12le (12-bit); 8/10-bit inputs are up-sampled
    by ffmpeg automatically.
    """
    result = subprocess.run(
        ['ffprobe', '-v', 'error',
         '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height',
         '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True, timeout=30)
    parts = result.stdout.strip().split(',')
    width, height = int(parts[0]), int(parts[1])

    cmd = [
        'ffmpeg', '-vsync', '0', '-hide_banner',
        '-i', str(path),
        '-vf', "zscale=matrix=bt2020nc:transfer=bt709:primaries=bt709:range=full",
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv444p12le',
        '-s', f'{width}x{height}',
        'pipe:1',
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg decode failed for {path}: {proc.stderr.decode()[-500:]}')

    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    n_frames = raw.size // (3 * height * width)
    planes = raw[:n_frames * 3 * height * width].reshape(n_frames, 3, height, width)
    return np.transpose(planes, (0, 2, 3, 1))


# ── Public API ────────────────────────────────────────────────────────

def decode_yuv(lr_path, hr_path):
    """Decode a single clip → (lr_yuv, hr_yuv) as [N, H, W, 3] uint16 raw YUV."""
    return _ffmpeg_to_yuv(lr_path), _ffmpeg_to_yuv(hr_path)


def _decode_clip(lr_path, hr_path):
    """Decode a single clip → (lr, hr) as [N, H, W, 3] float32 linear RGB via numpy."""
    lr_yuv, hr_yuv = decode_yuv(lr_path, hr_path)
    return _yuv_to_rgb_cpu(lr_yuv), _yuv_to_rgb_cpu(hr_yuv)


def discover_clips(paths):
    """Scan dataset directories for clip dirs containing meta.json / LR.mkv / HR.mkv."""
    clips = []
    for p in paths:
        p = Path(p)
        if not p.is_dir():
            continue
        for entry in sorted(p.iterdir()):
            if not entry.is_dir():
                continue
            lr = entry / 'LR.mkv'
            hr = entry / 'HR.mkv'
            meta = entry / 'meta.json'
            if lr.exists() and hr.exists() and meta.exists():
                with open(meta) as f:
                    info = json.load(f)
                clips.append({
                    'lr_path': lr,
                    'hr_path': hr,
                    'n_frames': info.get('num_frames', 0),
                    'clip_name': entry.name,
                })
    return clips



