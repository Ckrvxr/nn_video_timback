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
    """Best-effort attempt to free JAX compilation caches and trigger GC."""
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass


def _probe_video(path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) for a video file."""
    width, height = get_video_resolution(path)
    fps, _ = probe_video(path)
    return width, height, fps


def _safe_ictcp_to_yuv(ictcp: np.ndarray, bits: int = 12) -> np.ndarray:
    """Convert model output to YUV, sanitising non-finite values."""
    ictcp = np.nan_to_num(ictcp, nan=0.0)
    yuv = ictcp_to_yuv_np(ictcp, bits=bits)
    return yuv


def validate_clip(jit_apply, params, lr_path, hr_path, batch_size=16, baseline=False):
    """Run model on a single clip, save pred as YUV raw, return ffmpeg metrics."""
    tmp_path = None
    try:
        width, height, framerate = _probe_video(hr_path)

        if baseline:
            dist_path = str(lr_path)
            pix_fmt = None
        else:
            lr_all, _ = _decode_clip(lr_path, hr_path)
            n_frames = lr_all.shape[0]
            if n_frames == 0:
                raise RuntimeError("Decoded clip has 0 frames")

            pred_frames = []
            for i in range(0, n_frames, batch_size):
                batch = lr_all[i:i + batch_size]
                pred = np.array(jax.block_until_ready(jit_apply(params, batch)))
                pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
                pred_frames.append(pred)
            pred = np.concatenate(pred_frames, axis=0)

            yuv = _safe_ictcp_to_yuv(pred, bits=12)
            tmp = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            tmp.close()
            yuv.tofile(tmp.name)
            dist_path = tmp.name
            tmp_path = tmp.name
            pix_fmt = 'yuv444p12le'

        metrics = ffmpeg_metrics(hr_path, dist_path, pix_fmt=pix_fmt,
                                 width=width, height=height, framerate=framerate)
        # Treat all-zero metrics as a failure (ffmpeg likely errored out).
        if all(metrics[k] == 0.0 for k in metrics):
            raise RuntimeError("ffmpeg returned zero metrics; check stderr above")
        return metrics
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def validate(model, params, clips, batch_size, run_dir, name, baseline=False):
    """Validate across all clips in a dataset using ffmpeg."""
    # Re-use a single JIT-compiled apply function for the whole validation pass.
    jit_apply = jax.jit(lambda p, x: model.apply(p, x))
    total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0}
    n = 0

    _release_jax_memory()

    for clip in clips:
        # Try the requested batch size first; fall back to 1 on OOM.
        for attempt_bs in [batch_size, 1]:
            try:
                metrics = validate_clip(jit_apply, params,
                                        clip['lr_path'], clip['hr_path'],
                                        batch_size=attempt_bs, baseline=baseline)
                for k in total:
                    total[k] += metrics[k]
                n += 1
                break
            except Exception as e:
                err_msg = str(e)
                is_oom = 'OUT_OF_MEMORY' in err_msg or 'CUDA_ERROR_OUT_OF_MEMORY' in err_msg
                if is_oom and attempt_bs > 1:
                    console.warning(
                        f"OOM validating {clip.get('clip_name', '?')}, retrying with batch_size=1"
                    )
                    _release_jax_memory()
                    continue
                console.error(f"Validation failed for {clip.get('clip_name', '?')}: {e}")
                break
        _release_jax_memory()

    if n == 0:
        console.warning(f"No clips validated successfully for {name}")
        return float('nan'), float('nan'), float('nan')
    return total['psnr'] / n, total['ssim'] / n, total['vmaf'] / n
