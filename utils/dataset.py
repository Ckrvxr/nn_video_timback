import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler

from .video_loader import probe_frame_count
from .frame_cache import FrameCache


def _worker_init(worker_id):
    """Worker init: disable PyAV decode — workers read BloscCache only."""
    FrameCache().set_decode_allowed(False)


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

    @staticmethod
    def _inventory_cache_path(ds_root: Path) -> Path:
        return ds_root / '.inventory.json'

    @staticmethod
    def _inventory_fingerprint(ds_root: Path) -> str:
        """Quick fingerprint of the dataset directory contents."""
        hr_dir = ds_root / 'HR'
        if not hr_dir.exists():
            return ''
        files = sorted(hr_dir.glob('*.mp4'))
        variant_dirs = sorted(d for d in ds_root.iterdir()
                              if d.is_dir() and d.name != 'HR')
        parts = [f'{f.name}:{f.stat().st_mtime_ns}' for f in files]
        for vd in variant_dirs:
            v_files = sorted(vd.glob('*.mp4'))
            parts.append(f'vd:{vd.name}')
            parts.extend(f'{f.name}:{f.stat().st_mtime_ns}' for f in v_files)
        return '|'.join(parts)

    def _load_cached_inventory(self, ds_root: Path) -> list[dict] | None:
        cache_path = self._inventory_cache_path(ds_root)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            if cached.get('fingerprint') != self._inventory_fingerprint(ds_root):
                return None
            return [
                {'name': v['name'], 'n_frames': v['n_frames'],
                 'variants': v['variants'], 'ds_root': ds_root}
                for v in cached.get('videos', [])
            ]
        except Exception:
            return None

    def _save_cached_inventory(self, ds_root: Path, videos: list[dict]):
        cache_path = self._inventory_cache_path(ds_root)
        try:
            payload = {
                'fingerprint': self._inventory_fingerprint(ds_root),
                'videos': [
                    {'name': v['name'], 'n_frames': v['n_frames'],
                     'variants': v['variants']}
                    for v in videos
                ],
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _build_inventory(self) -> list[dict]:
        videos = []
        for ds_root in self.datasets:
            cached = self._load_cached_inventory(ds_root)
            if cached is not None:
                videos.extend(cached)
                continue

            ds_videos = []
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
                    ds_videos.append({'name': name, 'n_frames': n,
                                      'variants': variants, 'ds_root': ds_root})
            self._save_cached_inventory(ds_root, ds_videos)
            videos.extend(ds_videos)
        return videos

    def _build_val_plan(self) -> list[dict]:
        """Enumerate every valid centre frame for validation."""
        half = self.frames // 2
        plan = []
        for v in self.videos:
            for f in range(half, v['n_frames'] - half):
                plan.append({'video': v['name'], 'variant': v['variants'][0],
                             'frame': f, 'ds_root': v['ds_root']})
        return plan

    def load_video(self, video_name: str, variant_name: str, ds_root: Path):
        key = (video_name, variant_name, ds_root)
        if self._cache_key == key:
            return

        if self.enable_cache:
            FrameCache().set_cache_root(ds_root / 'cache_yuv')

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
            video_name, variant_name, center_frame, ds_root, *extra = item
            crop_seed = extra[0] if len(extra) > 0 else None
            video_id = extra[1] if len(extra) > 1 else None
        else:
            plan = self._val_plan[item % len(self._val_plan)]
            video_name, variant_name, center_frame, ds_root = (
                plan['video'], plan['variant'], plan['frame'], plan['ds_root'])
            crop_seed = None
            video_id = None

        self.load_video(video_name, variant_name, ds_root)

        half = self.frames // 2
        center_hr = self._hr_cache[center_frame]
        h, w = center_hr.shape[:2]

        if crop_seed is not None:
            rng = random.Random(crop_seed)
            crop_y = rng.randint(0, max(0, h - self.patch_size))
            crop_x = rng.randint(0, max(0, w - self.patch_size))
            crop_size = self.patch_size
        else:
            crop_y, crop_x, crop_size = self._get_crop_params(h, w)

        lr_frames = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            lr = self._safe_crop(self._lr_cache[i_clamped], crop_y, crop_x, crop_size)
            lr_frames.append(lr)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        # hr_prev for temporal loss if needed
        hr_prev_frame = max(0, center_frame - 1)
        hr_prev_patch = self._safe_crop(self._hr_cache[hr_prev_frame], crop_y, crop_x, crop_size)

        # Convert to torch and float32, permute and normalize to YUV range [-1, 1].
        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0
        hr_prev_t = torch.from_numpy(hr_prev_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        out = {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}
        if video_id is not None:
            out['video_id'] = video_id
        return out


class CleanVSRDataset(AV1CompressedVideoDataset):
    def load_video(self, video_name: str, variant_name: str = None, ds_root: Path = None):
        key = (video_name, ds_root)
        if self._cache_key == key:
            return

        if self.enable_cache:
            FrameCache().set_cache_root(ds_root / 'cache_yuv')

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
            video_name, _, center_frame, ds_root, *extra = item
            crop_seed = extra[0] if len(extra) > 0 else None
            video_id = extra[1] if len(extra) > 1 else None
        else:
            plan = self._val_plan[item % len(self._val_plan)]
            video_name, center_frame, ds_root = (
                plan['video'], plan['frame'], plan['ds_root'])
            crop_seed = None
            video_id = None

        self.load_video(video_name, ds_root=ds_root)

        half = self.frames // 2
        center_hr = self._hr_cache[center_frame]
        h, w = center_hr.shape[:2]

        if crop_seed is not None:
            rng = random.Random(crop_seed)
            crop_y = rng.randint(0, max(0, h - self.patch_size))
            crop_x = rng.randint(0, max(0, w - self.patch_size))
            crop_size = self.patch_size
        else:
            crop_y, crop_x, crop_size = self._get_crop_params(h, w)

        lr_frames = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            hr = self._hr_cache[i_clamped]
            hr_crop = self._safe_crop(hr, crop_y, crop_x, crop_size)
            lr_frames.append(hr_crop)

        hr_patch = self._safe_crop(center_hr, crop_y, crop_x, crop_size)

        hr_prev_frame = max(0, center_frame - 1)
        hr_prev_patch = self._safe_crop(self._hr_cache[hr_prev_frame], crop_y, crop_x, crop_size)

        lr_t = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0
        hr_prev_t = torch.from_numpy(hr_prev_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}


class VideoBatchSampler(BatchSampler):
    """Yields batches where all items share one video + variant.
    This lets __getitem__ load only one video per batch (~1.1 GB peak).

    All batches from all videos are collected and shuffled globally so
    that consecutive batches come from different videos.

    With ``clip_repeat > 1`` the shuffled batch list is iterated multiple
    times.  This improves GPU utilization when the model is small (frames
    stay hot in cache) and can also improve convergence by exposing the
    model to repeated views of the same data.

    By default the variant is deterministic (variants[0]) so that the
    per-worker background preload can actually warm the cache. Set
    ``shuffle_variants=True`` to sample a random variant per video, at the
    cost of more cache misses if variants are not all preloaded.
    """
    def __init__(self, dataset, batch_size, shuffle_variants=False, clip_repeat=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants
        self.clip_repeat = clip_repeat

    def __iter__(self):
        half = self.dataset.frames // 2

        all_batches = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            pool = list(range(half, valid_end))
            random.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                all_batches.append([(v['name'], variant, f, v['ds_root']) for f in chunk])

        for _ in range(self.clip_repeat):
            random.shuffle(all_batches)
            yield from all_batches

    def __len__(self):
        total = sum(max(0, v['n_frames'] - self.dataset.frames + 1) for v in self.dataset.videos)
        return max(1, total // self.batch_size) * self.clip_repeat


class ValVideoBatchSampler(BatchSampler):
    """Deterministic batch sampler that covers every valid centre frame."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        half = self.dataset.frames // 2
        videos = list(self.dataset.videos)
        for v in videos:
            variant = v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            pool = list(range(half, valid_end))
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                yield [(v['name'], variant, f, v['ds_root']) for f in chunk]

    def __len__(self):
        return sum(max(0, v['n_frames'] - self.dataset.frames + 1) for v in self.dataset.videos)


class SequentialVideoBatchSampler(BatchSampler):
    """Yields sequential frame batches per video, video-shuffled.

    Each video's frames are walked in temporal order, chunked into batches.
    Videos are shuffled globally so order changes each epoch.
    Crop position is fixed per video (seeded by video name).
    """
    def __init__(self, dataset, batch_size, shuffle_variants=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants

    def __iter__(self):
        half = self.dataset.frames // 2

        video_batches = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            crop_seed = hash(f"{v['name']}_{v['ds_root']}") & 0x7FFFFFFF
            video_id = hash(f"{v['name']}_{v['ds_root']}_{variant}") & 0x7FFFFFFF
            frames = list(range(half, valid_end))
            for i in range(0, len(frames), self.batch_size):
                chunk = frames[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                video_batches.append(
                    [(v['name'], variant, f, v['ds_root'], crop_seed, video_id) for f in chunk]
                )

        random.shuffle(video_batches)
        yield from video_batches

    def __len__(self):
        total = sum(max(0, v['n_frames'] - self.dataset.frames + 1) for v in self.dataset.videos)
        return max(1, total // self.batch_size)


def collate_vsr(batch: list[dict]) -> dict[str, Any]:
    lr_frames = torch.stack([item['lr_frames'] for item in batch], dim=0)
    hr = torch.stack([item['hr'] for item in batch], dim=0)

    out = {'lr_frames': lr_frames, 'hr': hr}
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
    enable_cache: bool = True,
    cache_max_videos: int | None = None,
    prefetch_factor: int | None = None,
    shuffle_variants: bool = False,
    clip_repeat: int = 1,
    sequential: bool = False,
    persistent_workers: bool | None = None,
) -> DataLoader:
    cls = AV1CompressedVideoDataset if data_type == 'compressed' else CleanVSRDataset
    dataset = cls(
        datasets=datasets,
        patch_size=patch_size,
        frames=frames,
        is_train=is_train,
        enable_cache=enable_cache,
    )

    if cache_max_videos is not None:
        import os
        os.environ['FRAME_CACHE_MAX_VIDEOS'] = str(cache_max_videos)
        FrameCache()._max_videos = int(cache_max_videos)

    if prefetch_factor is None:
        prefetch_factor = 2 if workers > 0 else None

    if persistent_workers is None:
        persistent_workers = workers > 0

    kwargs = {
        'num_workers': workers,
        'pin_memory': True,
        'persistent_workers': persistent_workers,
        'prefetch_factor': prefetch_factor if workers > 0 else None,
        'worker_init_fn': _worker_init if workers > 0 else None,
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
            collate_fn=collate_vsr,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_vsr,
        shuffle=is_train,
        **kwargs,
    )
