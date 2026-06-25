import os
import tempfile
from pathlib import Path

import numpy as np

from utils.colorspace import ictcp_to_yuv_np
from utils.data.mkv_loader import _decode_clip
from utils.evaluation.ffmpeg_metrics import ffmpeg_metrics


def validate_clip(model, params, lr_path, hr_path, baseline=False):
    """Run model on a single clip, save pred as YUV raw, return ffmpeg metrics."""
    tmp_path = None
    try:
        if baseline:
            dist_path = str(lr_path)
            pix_fmt = None
        else:
            lr_all, _ = _decode_clip(lr_path, hr_path)

            pred_frames = []
            for i in range(0, lr_all.shape[0], 16):
                batch = lr_all[i:i + 16]
                pred = np.array(model.apply(params, batch))
                pred_frames.append(pred)
            pred = np.concatenate(pred_frames, axis=0)

            yuv = ictcp_to_yuv_np(pred, bits=12)
            tmp = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False)
            tmp.close()
            yuv.tofile(tmp.name)
            dist_path = tmp.name
            tmp_path = tmp.name
            pix_fmt = 'yuv444p12le'

        return ffmpeg_metrics(hr_path, dist_path, pix_fmt=pix_fmt)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def validate(model, params, clips, batch_size, run_dir, name, baseline=False):
    """Validate across all clips in a dataset using ffmpeg."""
    total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0}
    n = 0

    for clip in clips:
        metrics = validate_clip(model, params,
                                clip['lr_path'], clip['hr_path'],
                                baseline=baseline)
        for k in total:
            total[k] += metrics[k]
        n += 1

    n = max(n, 1)
    return total['psnr'] / n, total['ssim'] / n, total['vmaf'] / n
