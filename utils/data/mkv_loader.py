"""MKV video batch loader — ffmpeg pipe + JAX GPU colour conversion."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from utils.colorspace.color_space import (
    _M_YUV2RGB, _M_RGB2LMS, _M_LMS2ICTCP,
    eotf_pq, oetf_pq,
)


# ── JIT-compiled GPU colour pipeline (12-bit native) ──────────────────

def _yuv12_to_ictcp(x: jnp.ndarray) -> jnp.ndarray:
    """YUV (12-bit normalized [0,1] / [-0.5,0.5]) → ICtCp.  NHWC."""
    rgb_nl = x @ _M_YUV2RGB.T
    rgb_nl = jnp.clip(rgb_nl, 0.0, None)
    rgb_lin = eotf_pq(rgb_nl)
    lms = rgb_lin @ _M_RGB2LMS.T
    lms_p = oetf_pq(lms)
    return lms_p @ _M_LMS2ICTCP.T


_JIT_YUV12_ICTCP = jax.jit(_yuv12_to_ictcp)


def _yuv_to_ictcp_gpu(yuv: np.ndarray) -> np.ndarray:
    """Convert YUV [N, H, W, 3] uint16 (yuv444p12le) → ICtCp float32 on GPU."""
    peak = jnp.float32(4095.0)
    center = jnp.float32(2048.0)
    yuv_f32 = jnp.array(yuv, dtype=jnp.float32)
    y = yuv_f32[..., 0:1] / peak
    u = (yuv_f32[..., 1:2] - center) / peak
    v = (yuv_f32[..., 2:3] - center) / peak
    norm = jnp.concatenate([y, u, v], axis=-1)
    ictcp = jax.block_until_ready(_JIT_YUV12_ICTCP(norm))
    return np.asarray(ictcp)


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
        '-f', 'rawvideo',
        '-pix_fmt', 'yuv444p12le',
        '-s', f'{width}x{height}',
        'pipe:1',
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg decode failed for {path}: {proc.stderr.decode()[-500:]}')

    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    n_frames = raw.size // (width * height * 3)
    return raw.reshape(n_frames, height, width, 3)


# ── Public API ────────────────────────────────────────────────────────

def _decode_clip(lr_path, hr_path):
    """Decode a single clip → (lr, hr) as [N, H, W, 3] float32 ICtCp via GPU."""
    lr_yuv = _ffmpeg_to_yuv(lr_path)
    hr_yuv = _ffmpeg_to_yuv(hr_path)
    return _yuv_to_ictcp_gpu(lr_yuv), _yuv_to_ictcp_gpu(hr_yuv)


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


def load_mkv_batch(paths, batch_size, shuffle=True, frames=1):
    """Generator that yields (lr_batch, hr_batch) float32 arrays from MKV clips.

    Each clip is decoded via ffmpeg pipe → numpy colour conversion (CPU).
    When shuffle=True (training), performs a single shuffled pass over the clips
    (the caller creates a fresh generator each epoch to reshuffle).
    """
    clips = discover_clips(paths)
    if not clips:
        raise RuntimeError(f'No MKV clips found in: {paths}')

    if shuffle:
        np.random.shuffle(clips)

    buf_lr = []
    buf_hr = []

    for clip in clips:
        lr_all, hr_all = _decode_clip(clip['lr_path'], clip['hr_path'])

        n = lr_all.shape[0]
        for i in range(0, n, frames):
            i_end = min(i + frames, n)
            if frames > 1:
                if i_end - i < frames:
                    break
                lr_concat = np.concatenate(lr_all[i:i_end], axis=-1)
                hr_center = hr_all[i + frames // 2]
            else:
                lr_concat = lr_all[i]
                hr_center = hr_all[i]

            buf_lr.append(lr_concat[np.newaxis, ...])
            buf_hr.append(hr_center[np.newaxis, ...])

            if len(buf_lr) >= batch_size:
                yield np.concatenate(buf_lr, axis=0), np.concatenate(buf_hr, axis=0)
                buf_lr.clear()
                buf_hr.clear()

    if buf_lr:
        yield np.concatenate(buf_lr, axis=0), np.concatenate(buf_hr, axis=0)
        buf_lr.clear()
        buf_hr.clear()
