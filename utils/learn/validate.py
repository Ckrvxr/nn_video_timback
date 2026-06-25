"""Model validation — YUV domain via ffmpeg (consistent path for pred & ref)."""

import gc
import os
import tempfile
from pathlib import Path

import jax
import numpy as np

from utils.colorspace import ictcp_to_yuv_np
from utils.console import console
from utils.data.mkv_loader import _decode_clip
from utils.evaluation.ffmpeg_metrics import ffmpeg_metrics
from utils.video_processor.probe_utils import get_video_resolution, probe_video


def _release_jax_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def _probe(path: Path) -> tuple[int, int, float]:
    w, h = get_video_resolution(path)
    fps, _ = probe_video(path)
    return w, h, fps


def validate_clip_yuv(jit_apply, params, lr_path, hr_path, batch_size=16):
    """Run model on one clip, compare pred(ICtCp→YUV) vs HR.mkv via ffmpeg."""
    lr_all, _ = _decode_clip(lr_path, hr_path)
    n_frames = lr_all.shape[0]
    if n_frames == 0:
        raise RuntimeError("Decoded clip has 0 frames")

    _, H, W = lr_all.shape[:3]

    pred_frames = []
    for i in range(0, n_frames, batch_size):
        batch = lr_all[i:i + batch_size]
        pred = np.array(jax.block_until_ready(jit_apply(params, batch)))
        pred = np.nan_to_num(pred, nan=0.0)
        pred_frames.append(pred)
    pred = np.concatenate(pred_frames, axis=0)

    pred_yuv = ictcp_to_yuv_np(pred, bits=12)

    tmp = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
    tmp.close()
    pred_yuv.tofile(tmp.name)

    try:
        fps = 60  # placeholder; ffmpeg matches frame count regardless
        metrics = ffmpeg_metrics(hr_path, tmp.name, pix_fmt='yuv444p12le',
                                 width=W, height=H, framerate=fps)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    return metrics


def validate(model, params, clips, batch_size, name):
    """Validate across all clips: model output → YUV → ffmpeg vs HR.mkv."""
    jit_apply = jax.jit(lambda p, x: model.apply(p, x))
    total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0}
    n = 0

    _release_jax_memory()

    for clip in clips:
        for attempt_bs in [batch_size, 1]:
            try:
                metrics = validate_clip_yuv(jit_apply, params,
                                            clip['lr_path'], clip['hr_path'],
                                            batch_size=attempt_bs)
                for k in total:
                    total[k] += metrics[k]
                n += 1
                break
            except Exception as e:
                err_msg = str(e)
                is_oom = 'OUT_OF_MEMORY' in err_msg or 'CUDA_ERROR_OUT_OF_MEMORY' in err_msg
                if is_oom and attempt_bs > 1:
                    console.warning(f"OOM validating {clip.get('clip_name', '?')}, retrying batch=1")
                    _release_jax_memory()
                    continue
                console.error(f"Validation failed for {clip.get('clip_name', '?')}: {e}")
                break
        _release_jax_memory()

    if n == 0:
        console.warning(f"No clips validated for {name}")
        return float('nan'), float('nan'), float('nan')
    return total['psnr'] / n, total['ssim'] / n, total['vmaf'] / n


def baseline_yuv(clip):
    """Compare LR.mkv vs HR.mkv directly via ffmpeg (no model)."""
    w, h, fps = _probe(clip['hr_path'])
    lr_ms = ffmpeg_metrics(clip['hr_path'], str(clip['lr_path']),
                           pix_fmt=None, width=w, height=h, framerate=fps)
    return lr_ms
