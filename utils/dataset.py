import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler


_global_cache: dict | None = None


def _load_all_frames(root: Path) -> dict:
    cache = {'hr': {}, 'lr': {}}
    hr_dir = root / 'HR'

    for video_dir in sorted(hr_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        frames = []
        for p in sorted(video_dir.glob('*.png')):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cache['hr'][video_dir.name] = frames

    for vd in sorted(root.iterdir()):
        if not vd.is_dir() or vd.name == 'HR':
            continue
        cache['lr'][vd.name] = {}
        for video_dir in sorted(vd.iterdir()):
            if not video_dir.is_dir():
                continue
            frames = []
            for p in sorted(video_dir.glob('*.png')):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cache['lr'][vd.name][video_dir.name] = frames

    return cache


def _ensure_cache(root: Path):
    global _global_cache
    if _global_cache is not None:
        return
    _global_cache = _load_all_frames(root)


class AV1CompressedVideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        scales: list[int] = None,
        patch_size: int = 256,
        frames: int = 3,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.scales = scales or [1, 2, 3, 4, 5, 6]
        self.patch_size = patch_size
        self.frames = frames
        self.is_train = is_train
        self.hr_dir = self.root / 'HR'
        self.variant_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and d.name != 'HR'
        ])
        self.samples = self._scan()

    def _get_scale(self, idx: int) -> int:
        n = len(self.samples)
        if self.is_train and n > 0 and idx >= n:
            s = idx // n
            if s in self.scales:
                return s
        return random.choice(self.scales) if self.is_train else 4

    def _scan(self) -> list[dict]:
        samples = []
        for video in sorted(self.hr_dir.iterdir()):
            if not video.is_dir():
                continue
            hr_frames = sorted(video.glob('*.png'))
            n_frames = len(hr_frames)

            variant_names = []
            for vd in self.variant_dirs:
                lr_dir = vd / video.name
                if not lr_dir.exists():
                    continue
                lr_files = sorted(lr_dir.glob('*.png'))
                if len(lr_files) != n_frames:
                    continue
                variant_names.append(vd.name)

            if not variant_names:
                continue

            for i in range(n_frames):
                samples.append({
                    'video_name': video.name,
                    'variant_names': variant_names,
                    'center_idx': i,
                    'n_frames': n_frames,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _safe_crop(self, img: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        crop = img[y:y + size, x:x + size]
        if crop.shape[0] < size or crop.shape[1] < size:
            pad_h = max(0, size - crop.shape[0])
            pad_w = max(0, size - crop.shape[1])
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        return crop

    def _get_crop_params(self, h: int, w: int, scale: int):
        if self.is_train:
            crop_size = (self.patch_size // scale) * scale
        else:
            crop_size = min(h, w) if self.patch_size is None else self.patch_size
            crop_size = (crop_size // scale) * scale
        lr_h = crop_size // scale
        lr_w = crop_size // scale
        if self.is_train:
            y = random.randint(0, max(0, h - crop_size))
            x = random.randint(0, max(0, w - crop_size))
        else:
            y = max(0, (h - crop_size) // 2)
            x = max(0, (w - crop_size) // 2)
        return y, x, crop_size, lr_h, lr_w

    def __getitem__(self, idx: int) -> dict:
        _ensure_cache(self.root)
        cache = _global_cache

        scale = self._get_scale(idx)
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        center = sample['center_idx']
        half = self.frames // 2
        video_name = sample['video_name']
        variant_name = random.choice(sample['variant_names'])

        center_hr = cache['hr'][video_name][center]
        h, w = center_hr.shape[:2]
        crop_y, crop_x, crop_size, lr_h, lr_w = self._get_crop_params(h, w, scale)

        lr_frames = []
        for i in range(center - half, center + half + 1):
            i_clamped = max(0, min(i, sample['n_frames'] - 1))
            lr = cache['lr'][variant_name][video_name][i_clamped]
            lr_crop = self._safe_crop(lr, crop_y, crop_x, crop_size)
            lr = cv2.resize(lr_crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr_frames.append(lr)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {'lr_frames': lr_t, 'hr': hr_t, 'scale': scale}


class CleanVSRDataset(AV1CompressedVideoDataset):
    def _get_scale(self, idx: int) -> int:
        return super()._get_scale(idx)

    def _scan(self) -> list[dict]:
        samples = []
        for video in sorted(self.hr_dir.iterdir()):
            if not video.is_dir():
                continue
            hr_frames = sorted(video.glob('*.png'))
            n_frames = len(hr_frames)
            for i in range(n_frames):
                samples.append({
                    'video_name': video.name,
                    'center_idx': i,
                    'n_frames': n_frames,
                })
        return samples

    def __getitem__(self, idx: int) -> dict:
        _ensure_cache(self.root)
        cache = _global_cache

        scale = self._get_scale(idx)
        idx = idx % len(self.samples)
        sample = self.samples[idx]
        center = sample['center_idx']
        half = self.frames // 2
        video_name = sample['video_name']
        n_frames = sample['n_frames']

        center_hr = cache['hr'][video_name][center]
        h, w = center_hr.shape[:2]
        crop_y, crop_x, crop_size, lr_h, lr_w = self._get_crop_params(h, w, scale)

        lr_frames = []
        for i in range(center - half, center + half + 1):
            i_clamped = max(0, min(i, n_frames - 1))
            hr = cache['hr'][video_name][i_clamped]
            hr_crop = self._safe_crop(hr, crop_y, crop_x, crop_size)
            lr = cv2.resize(hr_crop, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            lr_frames.append(lr)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {'lr_frames': lr_t, 'hr': hr_t, 'scale': scale}


class RandomScaleBatchSampler(BatchSampler):
    """Yields batches where all items share the same scale.
    Encodes scale in high bits of each index so __getitem__ can decode it."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n = len(dataset)

    def __iter__(self):
        indices = list(range(self.n))
        random.shuffle(indices)
        for i in range(0, self.n, self.batch_size):
            scale = random.choice(self.dataset.scales)
            offset = scale * self.n
            batch = [idx + offset for idx in indices[i:i + self.batch_size]]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        return self.n // self.batch_size


def collate_vsr(batch: list[dict]) -> dict[str, Any]:
    scale = batch[0]['scale']
    for item in batch:
        if item['scale'] != scale:
            raise ValueError(f"Mixed scales in batch: {scale} vs {item['scale']}")

    lr_frames = torch.stack([item['lr_frames'] for item in batch], dim=0)
    hr = torch.stack([item['hr'] for item in batch], dim=0)

    return {'lr_frames': lr_frames, 'hr': hr, 'scale': scale}


def create_dataloader(
    root: str,
    batch_size: int = 16,
    scales: list[int] = None,
    patch_size: int = 256,
    frames: int = 3,
    workers: int = 8,
    is_train: bool = True,
    data_type: str = 'compressed',
) -> DataLoader:
    cls = AV1CompressedVideoDataset if data_type == 'compressed' else CleanVSRDataset
    dataset = cls(
        root=root,
        scales=scales,
        patch_size=patch_size,
        frames=frames,
        is_train=is_train,
    )

    kwargs = {
        'num_workers': workers,
        'pin_memory': True,
        'persistent_workers': workers > 0,
        'prefetch_factor': 2 if workers > 0 else None,
    }
    if is_train and batch_size > 1:
        sampler = RandomScaleBatchSampler(dataset, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_vsr,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        drop_last=is_train,
        collate_fn=collate_vsr if batch_size > 1 else None,
        **kwargs,
    )
