"""Model validation — RGB float32 domain (no YUV round-trip ceiling).

PSNR/SSIM computed in float32 RGB via linear matrix (ictcp_to_yuv → yuv_to_rgb),
matching the old code's approach.  VMAF from ICtCp→RGB→8-bit PNG → ffmpeg libvmaf.
"""

import gc
import os
import re
import subprocess
import tempfile

import jax
import numpy as np
from PIL import Image

from utils.colorspace import ictcp_to_yuv, yuv_to_rgb
from utils.colorspace.color_space import eotf_pq_np
from utils.console import console
from utils.data.mkv_loader import _decode_clip
from utils.evaluation.metrics import calculate_psnr_batch, calculate_ssim_batch


def _release_jax_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def _vmaf_8bit_png(pred_ictcp: np.ndarray, target_ictcp: np.ndarray) -> float:
    """VMAF via ICtCp → YUV → RGB → PQ→sRGB tone map → uint8 PNG → ffmpeg libvmaf.

    libvmaf expects standard 8-bit SDR.  The linear matrix from ``yuv_to_rgb``
    gives PQ-encoded RGB; we apply PQ EOTF followed by sRGB gamma to produce
    proper SDR input.
    """
    pred_yuv = np.array(ictcp_to_yuv(pred_ictcp))
    target_yuv = np.array(ictcp_to_yuv(target_ictcp))
    pred_rgb = np.array(yuv_to_rgb(pred_yuv))
    target_rgb = np.array(yuv_to_rgb(target_yuv))

    # [-1, 1] → [0, 1] → PQ EOTF → linear light → sRGB gamma → uint8
    pred_rgb = eotf_pq_np((pred_rgb + 1) / 2)
    target_rgb = eotf_pq_np((target_rgb + 1) / 2)
    pred_u8 = (np.power(np.clip(pred_rgb / 10000, 0, 1), 1 / 2.2) * 255).clip(0, 255).astype(np.uint8)
    target_u8 = (np.power(np.clip(target_rgb / 10000, 0, 1), 1 / 2.2) * 255).clip(0, 255).astype(np.uint8)

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


def _clip_rgb_metrics(pred_ictcp: np.ndarray, target_ictcp: np.ndarray) -> dict:
    """Return ``{'psnr', 'ssim', 'vmaf'}`` from ICtCp tensors.

    PSNR/SSIM in float32 RGB (linear matrix, no PQ), no YUV ceiling.
    VMAF from 8-bit PNG screenshots.
    """
    pred_yuv = np.array(ictcp_to_yuv(pred_ictcp))
    target_yuv = np.array(ictcp_to_yuv(target_ictcp))
    pred_rgb = np.array(yuv_to_rgb(pred_yuv))
    target_rgb = np.array(yuv_to_rgb(target_yuv))

    psnr = float(calculate_psnr_batch(pred_rgb, target_rgb).mean())
    ssim = float(calculate_ssim_batch(pred_rgb, target_rgb).mean())
    vmaf = _vmaf_8bit_png(pred_ictcp, target_ictcp)

    return {'psnr': psnr, 'ssim': ssim, 'vmaf': vmaf}


def validate_clip_rgb(jit_apply, params, lr_path, hr_path, batch_size=16):
    """Run model on one clip: PSNR/SSIM in RGB float32, VMAF via PNG."""
    lr_all, hr_all = _decode_clip(lr_path, hr_path)
    n_frames = lr_all.shape[0]
    if n_frames == 0:
        raise RuntimeError("Decoded clip has 0 frames")

    pred_frames = []
    for i in range(0, n_frames, batch_size):
        batch = lr_all[i:i + batch_size]
        pred = np.array(jax.block_until_ready(jit_apply(params, batch)))
        pred = np.nan_to_num(pred, nan=0.0)
        pred_frames.append(pred)
    pred = np.concatenate(pred_frames, axis=0)

    return _clip_rgb_metrics(pred, hr_all[:len(pred)])


def validate(model, params, clips, batch_size, name):
    """Validate across all clips: RGB float32 PSNR/SSIM, PNG VMAF."""
    jit_apply = jax.jit(lambda p, x: model.apply(p, x))
    total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0}
    n = 0

    _release_jax_memory()

    for clip in clips:
        for attempt_bs in [batch_size, 1]:
            try:
                metrics = validate_clip_rgb(jit_apply, params,
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


def baseline_clip(clip):
    """Return baseline metrics (LR vs HR, RGB float32, no model)."""
    lr_all, hr_all = _decode_clip(clip['lr_path'], clip['hr_path'])
    return _clip_rgb_metrics(lr_all, hr_all)
