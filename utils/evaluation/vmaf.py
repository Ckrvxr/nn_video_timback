import os
import re
import subprocess
import tempfile
import torch
from PIL import Image
import numpy as np

from utils.color_space import yuv_to_rgb


def _find_ffmpeg() -> str:
    paths = [
        r'C:\Users\Ckrvxr\MyApp\ffmpeg-8.1\bin\ffmpeg.exe',
        'ffmpeg',
    ]
    for p in paths:
        try:
            r = subprocess.run([p, '-filters'], capture_output=True, text=True, timeout=5)
            if 'libvmaf' in (r.stdout + r.stderr):
                return p
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return 'ffmpeg'


_FFMPEG_PATH = _find_ffmpeg()


@torch.no_grad()
def compute_vmaf(
    pred_yuv: torch.Tensor,
    ref_yuv: torch.Tensor,
    tmpdir: str | None = None,
    ffmpeg_path: str | None = None,
) -> float:
    if ffmpeg_path is None:
        ffmpeg_path = _FFMPEG_PATH
    pred_rgb = yuv_to_rgb(pred_yuv).cpu().numpy()
    ref_rgb = yuv_to_rgb(ref_yuv).cpu().numpy()

    pred_u8 = ((pred_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)
    ref_u8 = ((ref_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)

    if pred_u8.ndim == 4:
        pred_u8 = pred_u8[0]
        ref_u8 = ref_u8[0]
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
        ffmpeg_path, '-hide_banner',
        '-i', pred_path,
        '-i', ref_path,
        '-lavfi', '[0:v][1:v]libvmaf',
        '-f', 'null', '-',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

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

    match = re.search(r'VMAF score:\s*([\d.]+)', result.stderr or result.stdout)
    if match:
        return float(match.group(1))
    return 0.0
