"""Model validation — YUV domain via ffmpeg (consistent path for pred & ref).

PSNR/SSIM from 12-bit rawvideo comparison.  VMAF from ICtCp→RGB→8-bit PNG
(old code path) since libvmaf expects SDR content.
"""

import gc
import os
import re
import subprocess
import tempfile
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from utils.colorspace import ictcp_to_yuv, ictcp_to_yuv_np, yuv_to_rgb
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


def _vmaf_8bit_png(pred_ictcp: np.ndarray, target_ictcp: np.ndarray) -> float:
    """VMAF via ICtCp → YUV → RGB (linear matrix) → uint8 PNG → ffmpeg libvmaf.

    Matches the old code's VMAF path: libvmaf expects 8-bit SDR input.
    """
    # ICtCp → YUV float32 [-1, 1] → RGB float32 [-1, 1]
    pred_yuv = np.array(ictcp_to_yuv(pred_ictcp))
    target_yuv = np.array(ictcp_to_yuv(target_ictcp))
    pred_rgb = np.array(yuv_to_rgb(pred_yuv))
    target_rgb = np.array(yuv_to_rgb(target_yuv))

    # [-1, 1] → uint8
    pred_u8 = ((pred_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)
    target_u8 = ((target_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # Take first frame, ensure HWC layout
    if pred_u8.ndim == 4:
        pred_u8, target_u8 = pred_u8[0], target_u8[0]
    if pred_u8.shape[0] in (1, 3):
        pred_u8 = pred_u8.transpose(1, 2, 0)
        target_u8 = target_u8.transpose(1, 2, 0)

    tmpdir = tempfile.mkdtemp()
    try:
        ref_path = os.path.join(tmpdir, 'ref.png')
        pred_path = os.path.join(tmpdir, 'pred.png')
        Image.fromarray(target_u8).save(ref_path)
        Image.fromarray(pred_u8).save(pred_path)

        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-i', ref_path, '-i', pred_path,
             '-lavfi', 'libvmaf', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=30,
        )
        m = re.search(r'VMAF score:\s*([\d.]+)', result.stderr or result.stdout)
        return float(m.group(1)) if m else 0.0
    finally:
        for f in [ref_path, pred_path]:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


def validate_clip_yuv(jit_apply, params, lr_path, hr_path, batch_size=16):
    """Run model on one clip: PSNR/SSIM from 12-bit YUV, VMAF from 8-bit PNG."""
    lr_all, hr_all = _decode_clip(lr_path, hr_path)
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

    # PSNR / SSIM via 12-bit YUV rawvideo
    pred_yuv = ictcp_to_yuv_np(pred, bits=12)
    tmp = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
    tmp.close()
    pred_yuv.tofile(tmp.name)
    try:
        metrics = ffmpeg_metrics(hr_path, tmp.name, pix_fmt='yuv444p12le',
                                 width=W, height=H, framerate=60)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # VMAF via 8-bit PNG (libvmaf expects SDR content)
    vmaf = _vmaf_8bit_png(pred, hr_all[:len(pred)])
    metrics['vmaf'] = vmaf

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
