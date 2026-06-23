from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..memory import check_cpu_allocation
from .clean_dataset import CleanVideoDataset
from .compressed_dataset import CompressedVideoDataset
from .compressed_samplers import (
    VideoBatchSampler,
    ValVideoBatchSampler,
    SequentialVideoBatchSampler,
)
from .preproc_dataset import (
    PreprocessedVideoDataset,
    RawVideoDataset,
    MirrorDataset,
    PreprocessedBatchSampler,
)


def collate_video(batch: list[dict]) -> dict[str, Any]:
    lr_frames = torch.stack([item['lr_frames'] for item in batch], dim=0)
    hr = torch.stack([item['hr'] for item in batch], dim=0)

    out = {'lr_frames': lr_frames, 'hr': hr}
    video_id = batch[0].get('video_id')
    if video_id is not None:
        out['video_id'] = video_id
    return out


def collate_raw(batch: list[dict]) -> dict[str, Any]:
    B = len(batch)
    lr_shape = batch[0]['lr_frames'].shape
    hr_shape = batch[0]['hr'].shape
    lr_dtype = batch[0]['lr_frames'].dtype
    hr_dtype = batch[0]['hr'].dtype

    total_bytes = int(np.prod((B, *lr_shape)) * np.dtype(lr_dtype).itemsize
                      + np.prod((B, *hr_shape)) * np.dtype(hr_dtype).itemsize)
    
    # More aggressive memory checking for 16GB RAM constraint
    if not check_cpu_allocation(total_bytes, 'collate_raw', ratio=0.5):
        B = max(1, B // 2)
        batch = batch[:B]
        # Recalculate after reduction
        total_bytes = int(np.prod((B, *lr_shape)) * np.dtype(lr_dtype).itemsize
                          + np.prod((B, *hr_shape)) * np.dtype(hr_dtype).itemsize)

    lr_out = np.empty((B, *lr_shape), dtype=lr_dtype)
    hr_out = np.empty((B, *hr_shape), dtype=hr_dtype)

    for i, item in enumerate(batch):
        lr_out[i] = item['lr_frames']
        hr_out[i] = item['hr']

    out = {'lr_frames': torch.from_numpy(lr_out), 'hr': torch.from_numpy(hr_out)}
    video_id = batch[0].get('video_id')
    if video_id is not None:
        out['video_id'] = video_id
    return out


def create_dataloader(
    datasets: list[str],
    batch_size: int = 16,
    patch_size: int = 256,
    frames: int = 3,
    workers: int = 8,
    is_train: bool = True,
    data_type: str = 'compressed',
    prefetch_factor: int | None = None,
    shuffle_variants: bool = False,
    clip_repeat: int = 1,
    segment_repeat: int = 1,
    sequential: bool = False,
    persistent_workers: bool | None = None,
    patch_buffer: int = 0,
    shuffle: bool | None = None,
    max_cached_segments: int = 8,
) -> DataLoader:
    """Create dataloader with memory optimizations for 16GB RAM + 7GB VRAM"""
    # Reduce workers and prefetch for memory-constrained systems
    if workers > 4:
        workers = 4
    if prefetch_factor is None or prefetch_factor > 2:
        prefetch_factor = 2
    # Reduce cached segments for memory efficiency
    if max_cached_segments > 4:
        max_cached_segments = 4
    if data_type in ('cache', 'raw'):
        cls = PreprocessedVideoDataset if data_type == 'cache' else RawVideoDataset
        collate = collate_video if data_type == 'cache' else collate_raw
        ds_list = [cls(d, frames=frames, max_cached_segments=max_cached_segments) for d in datasets]
        if prefetch_factor is None:
            prefetch_factor = 1 if workers > 0 else None
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
        train_dataset = MirrorDataset(dataset) if is_train else dataset

        kwargs = {
            'num_workers': workers,
            'pin_memory': True,
            'persistent_workers': persistent_workers if persistent_workers is not None else (workers > 0),
            'prefetch_factor': prefetch_factor if (prefetch_factor is not None and workers > 0) else None,
        }
        if is_train:
            sampler = PreprocessedBatchSampler(dataset, batch_size, segment_repeat=segment_repeat)
            return DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate, **kwargs)
        shuffle_val = shuffle if shuffle is not None else is_train
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_val,
                          collate_fn=collate, **kwargs)

    cls = CompressedVideoDataset if data_type == 'compressed' else CleanVideoDataset
    dataset = cls(
        datasets=datasets,
        patch_size=patch_size,
        frames=frames,
        is_train=is_train,
        patch_buffer=patch_buffer,
    )

    if prefetch_factor is None:
        prefetch_factor = 2 if workers > 0 else None

    if persistent_workers is None:
        persistent_workers = workers > 0

    kwargs = {
        'num_workers': workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor if workers > 0 else None,
    }

    if batch_size > 1:
        if is_train and sequential:
            sampler = SequentialVideoBatchSampler(dataset, batch_size,
                                                   shuffle_variants=shuffle_variants)
        elif is_train:
            sampler = VideoBatchSampler(dataset, batch_size, shuffle_variants=shuffle_variants,
                                         clip_repeat=clip_repeat)
        else:
            sampler = ValVideoBatchSampler(dataset, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_video,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_video,
        shuffle=is_train,
        **kwargs,
    )
