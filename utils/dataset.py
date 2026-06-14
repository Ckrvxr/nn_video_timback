import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler

from .video_loader import probe_frame_count
from .frame_cache import FrameCache


class AV1CompressedVideoDataset(Dataset):
    def __init__(
        self,
        datasets: list[str],
        patch_size: int = 256,
        frames: int = 3,
        is_train: bool = True,
        enable_cache: bool = True,
    ):
        self.datasets = [Path(d) for d in datasets]
        self.patch_size = patch_size
        self.frames = frames
        self.is_train = is_train
        self.enable_cache = enable_cache
        self.videos = self._build_inventory()

        self._cache_key = None
        self._hr_cache = []
        self._lr_cache = []

        if not is_train:
            self._val_plan = self._build_val_plan()

    @staticmethod
    def _list_video_names(hr_dir: Path) -> list[str]:
        return sorted(f.stem for f in hr_dir.glob('*.mp4'))

    @staticmethod
    def _get_n_frames(hr_dir: Path, video_name: str) -> int:
        return probe_frame_count(str(hr_dir / f'{video_name}.mp4'))

    @staticmethod
    def _variant_has(variant_dir: Path, video_name: str) -> bool:
        return (variant_dir / f'{video_name}.mp4').exists()

    @staticmethod
    def _variant_n_frames(variant_dir: Path, video_name: str) -> int:
        return probe_frame_count(str(variant_dir / f'{video_name}.mp4'))

    @staticmethod
    def _load_any_format(dir_path: Path, video_name: str) -> list[np.ndarray]:
        from .video_loader import load_video_frames
        return load_video_frames(str(dir_path / f'{video_name}.mp4'))

    def _build_inventory(self) -> list[dict]:
        videos = []
        for ds_root in self.datasets:
            hr_dir = ds_root / 'HR'
            variant_dirs = sorted(d for d in ds_root.iterdir()
                                  if d.is_dir() and d.name != 'HR')
            for name in self._list_video_names(hr_dir):
                n = self._get_n_frames(hr_dir, name)
                if n == 0:
                    continue
                variants = []
                for vd in variant_dirs:
                    if self._variant_has(vd, name) and self._variant_n_frames(vd, name) == n:
                        variants.append(vd.name)
                if variants:
                    videos.append({'name': name, 'n_frames': n,
                                   'variants': variants, 'ds_root': ds_root})
        return videos

    def _build_val_plan(self) -> list[dict]:
        return [{'video': v['name'], 'variant': v['variants'][0],
                 'frame': v['n_frames'] // 2, 'ds_root': v['ds_root']}
                for v in self.videos]

    def load_video(self, video_name: str, variant_name: str, ds_root: Path):
        key = (video_name, variant_name, ds_root)
        if self._cache_key == key:
            return

        if self.enable_cache:
            FrameCache().set_cache_root(ds_root / 'cache')

        hr_path = ds_root / 'HR' / f'{video_name}.mp4'
        lr_path = ds_root / variant_name / f'{video_name}.mp4'
        if not hr_path.exists():
            raise FileNotFoundError(f'HR video not found: {hr_path}')
        if not lr_path.exists():
            raise FileNotFoundError(f'LR video not found: {lr_path}')

        self._hr_cache, self._lr_cache = FrameCache().get_or_load(hr_path, lr_path)
        self._cache_key = key

    def __len__(self) -> int:
        return sum(max(0, v['n_frames'] - self.frames + 1) for v in self.videos)

    def _safe_crop(self, img: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        crop = img[y:y + size, x:x + size]
        if crop.shape[0] < size or crop.shape[1] < size:
            pad_h = max(0, size - crop.shape[0])
            pad_w = max(0, size - crop.shape[1])
            crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        return crop

    def _get_crop_params(self, h: int, w: int):
        if self.is_train:
            crop_size = self.patch_size
        else:
            crop_size = min(h, w)
        if self.is_train:
            y = random.randint(0, max(0, h - crop_size))
            x = random.randint(0, max(0, w - crop_size))
        else:
            y = max(0, (h - crop_size) // 2)
            x = max(0, (w - crop_size) // 2)
        return y, x, crop_size

    def __getitem__(self, item) -> dict:
        if isinstance(item, tuple):
            video_name, variant_name, center_frame, ds_root = item
        else:
            plan = self._val_plan[item % len(self._val_plan)]
            video_name, variant_name, center_frame, ds_root = (
                plan['video'], plan['variant'], plan['frame'], plan['ds_root'])

        self.load_video(video_name, variant_name, ds_root)

        half = self.frames // 2
        center_hr = self._hr_cache[center_frame]
        h, w = center_hr.shape[:2]
        crop_y, crop_x, crop_size = self._get_crop_params(h, w)

        lr_frames = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            lr = self._safe_crop(self._lr_cache[i_clamped], crop_y, crop_x, crop_size)
            lr_frames.append(lr)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {'lr_frames': lr_t, 'hr': hr_t}


class CleanVSRDataset(AV1CompressedVideoDataset):
    def load_video(self, video_name: str, variant_name: str = None, ds_root: Path = None):
        key = (video_name, ds_root)
        if self._cache_key == key:
            return

        if self.enable_cache:
            FrameCache().set_cache_root(ds_root / 'cache')

        hr_path = ds_root / 'HR' / f'{video_name}.mp4'
        self._hr_cache, self._lr_cache = FrameCache().get_or_load(hr_path, hr_path)
        self._cache_key = key

    def _build_inventory(self) -> list[dict]:
        videos = []
        for ds_root in self.datasets:
            hr_dir = ds_root / 'HR'
            for name in self._list_video_names(hr_dir):
                n = self._get_n_frames(hr_dir, name)
                if n > 0:
                    videos.append({'name': name, 'n_frames': n,
                                   'variants': ['clean'], 'ds_root': ds_root})
        return videos

    def __getitem__(self, item) -> dict:
        if isinstance(item, tuple):
            video_name, _, center_frame, ds_root = item
        else:
            plan = self._val_plan[item % len(self._val_plan)]
            video_name, center_frame, ds_root = (
                plan['video'], plan['frame'], plan['ds_root'])

        self.load_video(video_name, ds_root=ds_root)

        half = self.frames // 2
        center_hr = self._hr_cache[center_frame]
        h, w = center_hr.shape[:2]
        crop_y, crop_x, crop_size = self._get_crop_params(h, w)

        lr_frames = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            hr = self._hr_cache[i_clamped]
            hr_crop = self._safe_crop(hr, crop_y, crop_x, crop_size)
            lr_frames.append(hr_crop)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {'lr_frames': lr_t, 'hr': hr_t}


class VideoBatchSampler(BatchSampler):
    """Yields batches where all items share one video + variant.
    This lets __getitem__ load only one video per batch (~1.1 GB peak)."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        videos = list(self.dataset.videos)
        random.shuffle(videos)
        half = self.dataset.frames // 2

        for v in videos:
            variant = random.choice(v['variants'])
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            pool = list(range(half, valid_end))
            random.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                yield [(v['name'], variant, f, v['ds_root']) for f in chunk]

    def __len__(self):
        total = sum(max(0, v['n_frames'] - self.dataset.frames + 1) for v in self.dataset.videos)
        return max(1, total // self.batch_size)


def collate_vsr(batch: list[dict]) -> dict[str, Any]:
    lr_frames = torch.stack([item['lr_frames'] for item in batch], dim=0)
    hr = torch.stack([item['hr'] for item in batch], dim=0)

    return {'lr_frames': lr_frames, 'hr': hr}


def create_dataloader(
    datasets: list[str],
    batch_size: int = 16,
    patch_size: int = 256,
    frames: int = 3,
    workers: int = 8,
    is_train: bool = True,
    data_type: str = 'compressed',
    enable_cache: bool = True,
) -> DataLoader | AV1CompressedVideoDataset:
    cls = AV1CompressedVideoDataset if data_type == 'compressed' else CleanVSRDataset
    dataset = cls(
        datasets=datasets,
        patch_size=patch_size,
        frames=frames,
        is_train=is_train,
        enable_cache=enable_cache,
    )

    kwargs = {
        'num_workers': workers,
        'pin_memory': True,
        'persistent_workers': workers > 0,
        'prefetch_factor': 2 if workers > 0 else None,
    }

    if is_train and batch_size > 1:
        sampler = VideoBatchSampler(dataset, batch_size)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_vsr,
            **kwargs,
        )

    # Validation: return dataset directly (manual loop in validate())
    return dataset
