import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler

from ..memory import estimate_frame_size, safe_max_cached_segments
from .frame_cache import LazyFrameRange


class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_root: str, frames: int = 9):
        self.data_root = Path(data_root)
        self.frames = frames

        self.samples: list[dict] = []
        for seg_dir in sorted(self.data_root.iterdir()):
            hr_path = seg_dir / 'hr.npy'
            lr_path = seg_dir / 'lr.npy'
            meta_path = seg_dir / 'meta.json'
            if not hr_path.exists() or not lr_path.exists() or not meta_path.exists():
                continue
            with open(str(meta_path)) as f:
                meta = json.load(f)
            total = meta['num_frames']
            vid = hash(str(seg_dir)) & 0x7FFFFFFF
            for window_start in range(0, total - self.frames + 1):
                self.samples.append({
                    'seg_dir': str(seg_dir),
                    'lr_path': str(lr_path),
                    'hr_path': str(hr_path),
                    'window_start': window_start,
                    'video_id': vid,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        import functools

        @functools.lru_cache(maxsize=64)
        def _load(p: str) -> np.ndarray:
            return np.load(p)

        lr_all = _load(s['lr_path'])
        hr_all = _load(s['hr_path'])
        window_start = s['window_start']
        window_end = window_start + self.frames
        lr_win = torch.from_numpy(lr_all[window_start:window_end].copy()).float()
        hr_f = torch.from_numpy(hr_all[window_end - 1].copy()).float()
        return {'lr_frames': lr_win, 'hr': hr_f, 'video_id': s['video_id']}


class RawVideoDataset(Dataset):
    def __init__(self, data_root: str, frames: int = 9, max_cached_segments: int = 0):
        self.data_root = Path(data_root)
        self.frames = frames
        if max_cached_segments <= 0:
            lr_bytes = estimate_frame_size(512, 512, frames, np.dtype(np.uint8))
            hr_bytes = estimate_frame_size(512, 512, frames, np.dtype(np.uint16))
            est = lr_bytes + hr_bytes
            if est > 0:
                max_cached_segments = safe_max_cached_segments(est, 0.3)
            if max_cached_segments < 8:
                max_cached_segments = 8
        self._max_cached = max_cached_segments
        self._hr_ranges: OrderedDict[str, LazyFrameRange] = OrderedDict()
        self._lr_ranges: OrderedDict[str, LazyFrameRange] = OrderedDict()

        self.samples: list[dict] = []
        for seg_dir in sorted(self.data_root.iterdir()):
            hr_path = seg_dir / 'HR.mkv'
            lr_path = seg_dir / 'LR.mkv'
            meta_path = seg_dir / 'meta.json'
            if not hr_path.exists() or not lr_path.exists() or not meta_path.exists():
                continue
            with open(str(meta_path)) as f:
                meta = json.load(f)
            total = meta['num_frames']
            vid = hash(str(seg_dir)) & 0x7FFFFFFF
            seg_key = str(seg_dir)
            for window_start in range(0, total - self.frames + 1):
                self.samples.append({
                    'seg_key': seg_key,
                    'hr_path': str(hr_path),
                    'lr_path': str(lr_path),
                    'n_frames': total,
                    'window_start': window_start,
                    'video_id': vid,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        key = s['seg_key']

        if key not in self._hr_ranges:
            if len(self._hr_ranges) >= self._max_cached:
                evict_key, evict_hr = self._hr_ranges.popitem(last=False)
                evict_lr = self._lr_ranges.pop(evict_key)
                evict_hr.clear()
                evict_lr.clear()
            self._hr_ranges[key] = LazyFrameRange(s['hr_path'], s['n_frames'],
                                                  window_size=self.frames)
            self._lr_ranges[key] = LazyFrameRange(s['lr_path'], s['n_frames'],
                                                  window_size=self.frames)
        else:
            self._hr_ranges.move_to_end(key)
            self._lr_ranges.move_to_end(key)

        window_start = s['window_start']
        window_end = window_start + self.frames

        lr_yuvs = np.stack([self._lr_ranges[key][frame_index] for frame_index in range(window_start, window_end)], axis=0)
        hr_yuv = self._hr_ranges[key][window_end - 1]
        return {'lr_frames': lr_yuvs, 'hr': hr_yuv, 'video_id': s['video_id']}


class MirrorDataset(Dataset):
    def __init__(self, wrapped: Dataset):
        self.wrapped = wrapped

    @staticmethod
    def _flip_horizontal(x):
        if isinstance(x, torch.Tensor):
            return torch.flip(x, dims=[-1])
        return np.flip(x, axis=-2)

    def __getitem__(self, idx: int) -> dict:
        mirror = idx < 0
        real_idx = -idx - 1 if mirror else idx
        sample = self.wrapped[real_idx]
        if mirror:
            sample['lr_frames'] = self._flip_horizontal(sample['lr_frames'])
            sample['hr'] = self._flip_horizontal(sample['hr'])
        return sample

    def __len__(self) -> int:
        return len(self.wrapped)


class PreprocessedBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, batch_size: int, segment_repeat: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.segment_repeat = segment_repeat
        self._groups: dict[int, list[int]] | None = None

    def _build_groups(self) -> dict[int, list[int]]:
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        if hasattr(self.dataset, 'datasets'):
            offset = 0
            for ds in self.dataset.datasets:
                seed = hash(str(offset)) & 0x7FFFFFFF
                for i, s in enumerate(ds.samples):
                    global_idx = offset + i
                    unique_vid = (s['video_id'] ^ seed) & 0x7FFFFFFF
                    groups[unique_vid].append(global_idx)
                offset += len(ds)
        else:
            for i, s in enumerate(self.dataset.samples):
                groups[s['video_id']].append(i)
        return groups

    def __iter__(self):
        if self._groups is None:
            self._groups = self._build_groups()
        groups = self._groups
        self._groups = None

        group_batches = []
        for indices in groups.values():
            chunk_batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            for rep in range(self.segment_repeat):
                source = chunk_batches
                if rep % 2:
                    source = [[-i-1 for i in c] for c in reversed(chunk_batches)]
                group_batches.extend(source)

        random.shuffle(group_batches)
        yield from group_batches

    def __len__(self) -> int:
        if self._groups is None:
            self._groups = self._build_groups()
        total_chunks = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self._groups.values()
        )
        return max(1, total_chunks) * self.segment_repeat
