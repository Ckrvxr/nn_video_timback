from .probe_utils import sanitize, discover_inputs, probe_video, get_video_resolution, \
    _probe_bit_depth, color_conversion_filter, COLOR_TAGS, ENCODER_WEIGHTS, \
    LR_PIX_WEIGHTS, AV1_PRESETS, X265_PRESETS, X264_PRESETS, make_scale_filter


class CompressedVideoDataset:
    """Base class for compressed video datasets."""
    pass


def preprocess_dataset(output_dir: str):
    """Preprocess dataset to .npy cache (placeholder)."""
    import shutil
    import subprocess
    from pathlib import Path

    out = Path(output_dir)
    cache_dir = out.parent / f'{out.name}_npy'
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'Preprocessing {output_dir} → {cache_dir}')

    for clip_dir in sorted(out.iterdir()):
        if not clip_dir.is_dir():
            continue
        npy_path = cache_dir / f'{clip_dir.name}.npy'
        if npy_path.exists():
            continue
        lr_path = clip_dir / 'LR.mkv'
        hr_path = clip_dir / 'HR.mkv'
        if not lr_path.exists() or not hr_path.exists():
            continue
        # Use system ffmpeg to decode and convert to raw float
        # (Simplified - actual implementation would decode and write .npy)
        print(f'  {clip_dir.name}')
