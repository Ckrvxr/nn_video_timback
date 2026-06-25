"""Model validation — RGB float32 domain (no YUV round-trip ceiling).

Matches the old code's approach: convert ICtCp → YUV → RGB via exact linear
matrices (no PQ curves, no clipping, no quantisation), then compute PSNR/SSIM
in float32 RGB and VMAF from 8-bit PNG frames.
"""

import gc
import os
import re
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from utils.colorspace import ictcp_to_yuv, yuv_to_rgb
from utils.console import console
from utils.data.mkv_loader import _decode_clip
from utils.evaluation.metrics import calculate_psnr_batch, calculate_ssim_batch


def _release_jax_memory():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def _compute_vmaf_rgb(pred_rgb: np.ndarray, ref_rgb: np.ndarray, tmpdir: str | None = None) -> float:
    """Compute VMAF from two RGB float32 [-1,1] frames via PNG screenshots.

    Matches the old code's VMAF approach: RGB float32 → uint8 PNG → ffmpeg libvmaf.
    """
    pred_u8 = ((pred_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)
    ref_u8 = ((ref_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)

    if pred_u8.ndim == 4:
        pred_u8 = pred_u8[0]
        ref_u8 = ref_u8[0]
    # PyAV/FFmpeg expects NHWC, yuv_to_rgb returns CHW → transpose
    if pred_u8.ndim == 3 and pred_u8.shape[0] in (1, 3):
        pred_u8 = pred_u8.transpose(1, 2, 0)
        ref_u8 = ref_u8.transpose(1, 2, 0)

    cleanup = False
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
        cleanup = True

    ref_path = os.path.join(tmpdir, 'ref.png')
    pred_path = os.path.join(tmpdir, 'pred.png')
    Image.fromarray(ref_u8).save(ref_path)
    Image.fromarray(pred_u8).save(pred_path)

    cmd = [
        'ffmpeg', '-hide_banner',
        '-i', ref_path,
        '-i', pred_path,
        '-lavfi', 'libvmaf',
        '-f', 'null', '-',
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        match = re.search(r'VMAF score:\s*([\d.]+)', result.stderr or result.stdout)
        vmaf = float(match.group(1)) if match else 0.0
    except Exception:
        vmaf = 0.0

    if cleanup:
        for f in [ref_path, pred_path]:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    return vmaf


def validate_clip_rgb(jit_apply, params, lr_path, hr_path, batch_size=16):
    """Run model on one clip, return RGB-domain PSNR/SSIM + VMAF.

    1. Decode clip → ICtCp
    2. Model forward → pred ICtCp
    3. pred, hr → ictcp_to_yuv → yuv_to_rgb → float32 RGB
    4. PSNR/SSIM from float32 RGB, VMAF from PNG frames
    """
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
    pred_ictcp = np.concatenate(pred_frames, axis=0)
    target_ictcp = hr_all[:len(pred_ictcp)]

    # Convert to float32 RGB (linear YUV→RGB, no PQ/EOTF)
    pred_yuv = np.array(ictcp_to_yuv(pred_ictcp))
    target_yuv = np.array(ictcp_to_yuv(target_ictcp))
    pred_rgb = np.array(yuv_to_rgb(pred_yuv))
    target_rgb = np.array(yuv_to_rgb(target_yuv))

    psnr = float(calculate_psnr_batch(pred_rgb, target_rgb).mean())
    ssim = float(calculate_ssim_batch(pred_rgb, target_rgb).mean())

    # VMAF from first frame of the clip (PNG-based)
    vmaf = _compute_vmaf_rgb(pred_rgb[:1], target_rgb[:1])

    return {'psnr': psnr, 'ssim': ssim, 'vmaf': vmaf}


def validate_rgb(model, params, clips, batch_size, name):
    """Primary validation: RGB float32 domain (matches old code)."""
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
                console.error(f"RGB validation failed for {clip.get('clip_name', '?')}: {e}")
                break
        _release_jax_memory()

    if n == 0:
        console.warning(f"No clips validated for {name}")
        return float('nan'), float('nan'), float('nan')
    return total['psnr'] / n, total['ssim'] / n, total['vmaf'] / n


def validate_ictcp(model, params, clips, batch_size, name):
    """Secondary validation: ICtCp float32 domain (training loss space, no ceiling)."""
    jit_apply = jax.jit(lambda p, x: model.apply(p, x))
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    _release_jax_memory()

    for clip in clips:
        try:
            lr_all, hr_all = _decode_clip(clip['lr_path'], clip['hr_path'])
            n_frames = lr_all.shape[0]
            pred_frames = []
            for i in range(0, n_frames, batch_size):
                batch = lr_all[i:i + batch_size]
                pred = jax.block_until_ready(jit_apply(params, batch))
                pred_frames.append(pred)
            pred = jnp.concatenate(pred_frames, axis=0)
            target = jnp.array(hr_all, dtype=jnp.float32)
            total_psnr += float(calculate_psnr_batch(pred, target).mean())
            total_ssim += float(calculate_ssim_batch(pred, target).mean())
            n += 1
        except Exception as e:
            console.error(f"ICtCp validation failed for {clip.get('clip_name', '?')}: {e}")

    if n == 0:
        return float('nan'), float('nan')
    return total_psnr / n, total_ssim / n
