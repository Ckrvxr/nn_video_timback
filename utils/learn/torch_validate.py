import gc
import os
import re
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

from utils.colorspace.color_space import ictcp_to_rgb_np
from utils.console import console
from utils.data.mkv_loader import _decode_clip
from utils.evaluation.torch_metrics import calculate_psnr_batch, calculate_ssim_batch
from utils.loss.vgg_perceptual import VGGDistance


def _vmaf_8bit_png(pred_ictcp: np.ndarray, target_ictcp: np.ndarray) -> float:
    pred_rgb = ictcp_to_rgb_np(pred_ictcp)
    target_rgb = ictcp_to_rgb_np(target_ictcp)
    pred_u8 = (np.power(pred_rgb, 1 / 2.2) * 255).clip(0, 255).astype(np.uint8)
    target_u8 = (np.power(target_rgb, 1 / 2.2) * 255).clip(0, 255).astype(np.uint8)
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


_VGG_CACHE = None

def _get_vgg(device):
    global _VGG_CACHE
    if _VGG_CACHE is None:
        _VGG_CACHE = VGGDistance().to(device)
    return _VGG_CACHE


def _clip_rgb_metrics(pred_ictcp: np.ndarray, target_ictcp: np.ndarray) -> dict:
    pred_rgb = ictcp_to_rgb_np(pred_ictcp)
    target_rgb = ictcp_to_rgb_np(target_ictcp)
    pred_t = torch.from_numpy(pred_rgb).permute(0, 3, 1, 2).float()
    target_t = torch.from_numpy(target_rgb).permute(0, 3, 1, 2).float()
    psnr = float(calculate_psnr_batch(pred_t, target_t, max_val=1.0).mean())
    ssim = float(calculate_ssim_batch(pred_t, target_t, max_val=1.0).mean())
    vmaf = _vmaf_8bit_png(pred_ictcp, target_ictcp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg = _get_vgg(device)
    with torch.no_grad():
        vgg_val = float(vgg(pred_t[:4].to(device), target_t[:4].to(device))) if len(pred_t) >= 4 else float(vgg(pred_t.to(device), target_t.to(device)))
    return {'psnr': psnr, 'ssim': ssim, 'vmaf': vmaf, 'vgg': vgg_val}


def validate_clip_rgb(model, device, lr_path, hr_path, batch_size=16):
    lr_all, hr_all = _decode_clip(lr_path, hr_path)
    n_frames = lr_all.shape[0]
    if n_frames == 0:
        raise RuntimeError("Decoded clip has 0 frames")

    dtype = next(model.parameters()).dtype

    pred_frames = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n_frames, batch_size):
            batch = lr_all[i:i + batch_size]
            batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().to(device, dtype=dtype)
            pred = model(batch_t)
            pred = pred.float().cpu().permute(0, 2, 3, 1).numpy()
            pred = np.nan_to_num(pred, nan=0.0)
            pred_frames.append(pred)
    pred = np.concatenate(pred_frames, axis=0)
    return _clip_rgb_metrics(pred, hr_all[:len(pred)])


def validate(model, device, clips, batch_size, name):
    total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0, 'vgg': 0.0}
    n = 0
    for clip in clips:
        for attempt_bs in [batch_size, 1]:
            try:
                metrics = validate_clip_rgb(model, device, clip['lr_path'], clip['hr_path'], batch_size=attempt_bs)
                for k in total:
                    total[k] += metrics[k]
                n += 1
                break
            except Exception as e:
                err_msg = str(e)
                is_oom = 'out of memory' in err_msg.lower()
                if is_oom and attempt_bs > 1:
                    console.warning(f"OOM validating {clip.get('clip_name', '?')}, retrying batch=1")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                console.error(f"Validation failed for {clip.get('clip_name', '?')}: {e}")
                break
        gc.collect()
        torch.cuda.empty_cache()

    if n == 0:
        console.warning(f"No clips validated for {name}")
        return float('nan'), float('nan'), float('nan'), float('nan')
    return total['psnr'] / n, total['ssim'] / n, total['vmaf'] / n, total.get('vgg', 0.0) / n


def baseline_clip(clip):
    lr_all, hr_all = _decode_clip(clip['lr_path'], clip['hr_path'])
    return _clip_rgb_metrics(lr_all, hr_all)
