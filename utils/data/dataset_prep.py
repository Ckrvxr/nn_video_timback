import json
import random
from pathlib import Path

import numpy as np
import tqdm

from .video_loader import load_video_frame_range
from .video_probe import probe_resolution, probe_frame_count


def _batch_yuv_to_ictcp_crop_gpu(yuv_frames: list, crop_info: tuple,
                                 batch_size: int = 8, patch_size: int = 512) -> np.ndarray:
    import torch
    crop_y, crop_x, total_n = crop_info
    out = np.empty((total_n, 3, patch_size, patch_size), dtype=np.float16)

    for start in range(0, total_n, batch_size):
        end = min(start + batch_size, total_n)
        chunk = yuv_frames[start:end]
        batch = np.stack(chunk, axis=0)

        bits = 8
        if batch.dtype == np.uint16:
            max_val = int(batch.max())
            bits = 12 if max_val > 1023 else 10
        peak = float((1 << bits) - 1)
        center = float(1 << (bits - 1))

        t = torch.from_numpy(batch.astype(np.float32, copy=False)).cuda()
        t = t.permute(0, 3, 1, 2).contiguous()
        t[:, 0:1] = t[:, 0:1] / peak * 255.0
        t[:, 1:] = (t[:, 1:] - center) / peak * 255.0 + 128.0
        t = t / 127.5 - 1.0

        from utils.color_space import yuv_to_ictcp
        with torch.no_grad():
            ictcp = yuv_to_ictcp(t).cpu().numpy().astype(np.float16)

        out[start:end] = ictcp[:, :, crop_y:crop_y + patch_size, crop_x:crop_x + patch_size]

    return out


def _preprocess_one_segment(seg_dir: Path, crop_info: tuple,
                            num_frames: int = 30, patch_size: int = 512):
    yuv = load_video_frame_range(str(seg_dir / 'HR.mkv'), 0, num_frames)
    hr_p = _batch_yuv_to_ictcp_crop_gpu(yuv, crop_info[:2] + (num_frames,))
    np.save(str(seg_dir / 'hr.npy'), hr_p)
    del yuv, hr_p

    lr_path = seg_dir / 'LR.mkv'
    if lr_path.exists():
        yuv = load_video_frame_range(str(lr_path), 0, num_frames)
        lr_p = _batch_yuv_to_ictcp_crop_gpu(yuv, crop_info[:2] + (num_frames,))
        np.save(str(seg_dir / 'lr.npy'), lr_p)
        del yuv, lr_p

    with open(str(seg_dir / 'meta.json'), 'w') as f:
        json.dump({'window_start': crop_info[2], 'grid_y': crop_info[3],
                   'grid_x': crop_info[4], 'gy': crop_info[5], 'gx': crop_info[6],
                   'num_frames': num_frames}, f, indent=2)


def preprocess_dataset(data_dir: str, num_frames: int = 30, patch_size: int = 512):
    data_path = Path(data_dir)
    dirs = sorted(d for d in data_path.iterdir()
                  if d.is_dir() and d.name != '.tmp')

    for seg_dir in tqdm.tqdm(dirs, desc='Preprocessing', unit='seg'):
        if (seg_dir / 'meta.json').exists():
            continue
        hr_path = seg_dir / 'HR.mkv'
        if not hr_path.exists():
            continue

        h, w = probe_resolution(str(hr_path))
        gy, gx = h // patch_size, w // patch_size

        rng = random.Random(hash(seg_dir.name))
        gyi = rng.randint(0, max(1, gy) - 1)
        gxi = rng.randint(0, max(1, gx) - 1)
        crop_info = (0, 0, 0, gyi, gxi, gy, gx)

        _preprocess_one_segment(seg_dir, crop_info, num_frames, patch_size)
