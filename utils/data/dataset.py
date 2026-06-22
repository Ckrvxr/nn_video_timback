import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler

from .video_loader import probe_frame_count, probe_resolution, load_video_frame_range
from .frame_cache import LazyFrameRange


def _yuv_to_ictcp(yuv_batch: np.ndarray) -> torch.Tensor:
    """YUV [B, H, W, 3] uint8/uint16 → ICtCp [B, 3, H, W] float16 on CUDA."""
    import torch
    B, H, W, _ = yuv_batch.shape
    bits = 8
    if yuv_batch.dtype == np.uint16:
        max_val = int(yuv_batch.max())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    t = torch.from_numpy(yuv_batch.astype(np.float32, copy=False)).cuda()
    t = t.permute(0, 3, 1, 2).contiguous()
    t[:, 0:1] = t[:, 0:1] / peak * 255.0
    t[:, 1:] = (t[:, 1:] - center) / peak * 255.0 + 128.0
    t = t / 127.5 - 1.0

    from utils.color_space import yuv_to_ictcp
    with torch.no_grad():
        return yuv_to_ictcp(t).half()


def batch_yuv_to_ictcp(yuv: torch.Tensor, device: torch.device) -> torch.Tensor:
    """CPU YUV tensor → GPU ICtCp float16.

    Input shapes supported:
        [B, H, W, 3]        — single frame per sample
        [B, F, H, W, 3]     — multiple frames per sample

    Returns:
        [B, 3, H, W] or [B, F, 3, H, W] float16 on *device*.
    """
    orig_ndim = yuv.ndim
    if orig_ndim == 5:
        orig_B, F, H, W, C = yuv.shape
        yuv = yuv.flatten(0, 1)

    _, H, W, C = yuv.shape
    bits = 8
    if yuv.dtype == torch.uint16:
        max_val = int(yuv.to(torch.int32).max().item())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    yuv = yuv.to(device=device, dtype=torch.float32, non_blocking=True)
    yuv = yuv.permute(0, 3, 1, 2).contiguous()
    yuv[:, 0:1] = yuv[:, 0:1] / peak * 255.0
    yuv[:, 1:] = (yuv[:, 1:] - center) / peak * 255.0 + 128.0
    yuv = yuv / 127.5 - 1.0

    from utils.color_space import yuv_to_ictcp
    with torch.no_grad():
        ictcp = yuv_to_ictcp(yuv).half()

    if orig_ndim == 5:
        ictcp = ictcp.view(orig_B, F, 3, H, W)
    return ictcp

class CompressedVideoDataset(Dataset):
    def __init__(
        self,
        datasets: list[str],
        patch_size: int = 256,
        frames: int = 3,
        is_train: bool = True,
        patch_buffer: int = 0,
    ):
        self.datasets = [Path(d) for d in datasets]
        self.patch_size = patch_size
        self.frames = frames
        self.is_train = is_train
        self.patch_buffer = patch_buffer
        self.videos = self._build_inventory()

        self._cache_key = None
        self._hr_cache = []
        self._lr_cache = []

    @staticmethod
    def _list_video_names(hr_dir: Path) -> list[str]:
        names = set(f.stem for f in hr_dir.glob('*.mp4'))
        names.update(f.stem for f in hr_dir.glob('*.mkv'))
        return sorted(names)

    @staticmethod
    def _get_n_frames(hr_dir: Path, video_name: str) -> int:
        mp4 = hr_dir / f'{video_name}.mp4'
        return probe_frame_count(str(mp4 if mp4.exists() else hr_dir / f'{video_name}.mkv'))

    @staticmethod
    def _get_resolution(hr_dir: Path, video_name: str) -> tuple[int, int]:
        mp4 = hr_dir / f'{video_name}.mp4'
        return probe_resolution(str(mp4 if mp4.exists() else hr_dir / f'{video_name}.mkv'))

    @staticmethod
    def _hr_path(hr_dir: Path, video_name: str) -> Path:
        mp4 = hr_dir / f'{video_name}.mp4'
        return mp4 if mp4.exists() else hr_dir / f'{video_name}.mkv'

    @staticmethod
    def _variant_has(variant_dir: Path, video_name: str) -> bool:
        return (variant_dir / f'{video_name}.mp4').exists()

    @staticmethod
    def _variant_n_frames(variant_dir: Path, video_name: str) -> int:
        return probe_frame_count(str(variant_dir / f'{video_name}.mp4'))

    @staticmethod
    def _inventory_cache_path(ds_root: Path) -> Path:
        return ds_root / '.inventory.json'

    @staticmethod
    def _inventory_fingerprint(ds_root: Path) -> str:
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
            videos = cached.get('videos', [])
            if videos and ('h' not in videos[0] or 'hr_ext' not in videos[0]):
                return None
            return [
                {'name': v['name'], 'n_frames': v['n_frames'],
                 'variants': v['variants'], 'ds_root': ds_root,
                 'h': v['h'], 'w': v['w'], 'hr_ext': v['hr_ext']}
                for v in videos
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
                     'variants': v['variants'], 'h': v['h'], 'w': v['w'],
                     'hr_ext': v['hr_ext']}
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
                hr_path = self._hr_path(hr_dir, name)
                n = self._get_n_frames(hr_dir, name)
                if n == 0:
                    continue
                h, w = self._get_resolution(hr_dir, name)
                hr_ext = hr_path.suffix
                variants = []
                for vd in variant_dirs:
                    if self._variant_has(vd, name) and self._variant_n_frames(vd, name) == n:
                        variants.append(vd.name)
                if variants:
                    ds_videos.append({'name': name, 'n_frames': n,
                                      'variants': variants, 'ds_root': ds_root,
                                      'h': h, 'w': w, 'hr_ext': hr_ext})
            self._save_cached_inventory(ds_root, ds_videos)
            videos.extend(ds_videos)
        return videos

    def load_video(self, video_name: str, variant_name: str, ds_root: Path):
        key = (video_name, variant_name, ds_root)
        if self._cache_key == key:
            return

        video_info = next(
            (v for v in self.videos
             if v['name'] == video_name and v['ds_root'] == ds_root),
            None)
        if video_info is None:
            raise RuntimeError(f'Video {video_name} not found in inventory')
        n_frames = video_info['n_frames']
        hr_ext = video_info.get('hr_ext', '.mp4')

        hr_path = ds_root / 'HR' / f'{video_name}{hr_ext}'
        lr_path = ds_root / variant_name / f'{video_name}.mp4'
        if not hr_path.exists():
            raise FileNotFoundError(f'HR video not found: {hr_path}')
        if not lr_path.exists():
            raise FileNotFoundError(f'LR video not found: {lr_path}')

        self._hr_cache = LazyFrameRange(str(hr_path), n_frames, window_size=self.frames)
        self._lr_cache = LazyFrameRange(str(lr_path), n_frames, window_size=self.frames)
        self._cache_key = key

    @staticmethod
    def _detect_bits(yuv_batch: np.ndarray) -> int:
        if yuv_batch.dtype == np.uint16:
            return 12 if int(yuv_batch.max()) > 1023 else 10
        return 8

    def __len__(self) -> int:
        total = 0
        ps = self.patch_size
        for v in self.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.frames + 1) * gy * gx
        return total

    def __getitem__(self, item) -> dict:
        video_name, variant_name, center_frame, ds_root, gy, gx, *extra = item
        video_id = extra[0] if len(extra) > 0 else None

        self.load_video(video_name, variant_name, ds_root)

        half = self.frames // 2
        ps = self.patch_size
        b = self.patch_buffer
        crop_y = gy * ps
        crop_x = gx * ps

        def _maybe_pad(arr):
            if b:
                return np.pad(arr, ((b, b), (b, b), (0, 0)), mode='reflect')
            return arr

        lr_yuvs = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            lr_yuvs.append(_maybe_pad(
                self._lr_cache[i_clamped][crop_y:crop_y + ps, crop_x:crop_x + ps]))

        hr_yuv = _maybe_pad(
            self._hr_cache[center_frame][crop_y:crop_y + ps, crop_x:crop_x + ps])
        hr_prev_frame = max(0, center_frame - 1)
        hr_prev_yuv = _maybe_pad(
            self._hr_cache[hr_prev_frame][crop_y:crop_y + ps, crop_x:crop_x + ps])

        all_yuv = np.stack(lr_yuvs + [hr_yuv, hr_prev_yuv], axis=0)
        all_ictcp = _yuv_to_ictcp(all_yuv)

        n_temporal = self.frames
        lr_t = all_ictcp[:n_temporal].float()
        hr_t = all_ictcp[n_temporal].float()
        hr_prev_t = all_ictcp[n_temporal + 1].float()

        out = {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}
        if video_id is not None:
            out['video_id'] = video_id
        return out


class CleanVideoDataset(CompressedVideoDataset):
    def load_video(self, video_name: str, variant_name: str = None, ds_root: Path = None):
        key = (video_name, ds_root)
        if self._cache_key == key:
            return

        video_info = next(
            (v for v in self.videos
             if v['name'] == video_name and v['ds_root'] == ds_root),
            None)
        if video_info is None:
            raise RuntimeError(f'Video {video_name} not found in inventory')
        n_frames = video_info['n_frames']
        hr_ext = video_info.get('hr_ext', '.mp4')

        hr_path = ds_root / 'HR' / f'{video_name}{hr_ext}'

        self._hr_cache = LazyFrameRange(str(hr_path), n_frames, window_size=self.frames)
        self._lr_cache = LazyFrameRange(str(hr_path), n_frames, window_size=self.frames)
        self._cache_key = key

    def _build_inventory(self) -> list[dict]:
        videos = []
        for ds_root in self.datasets:
            hr_dir = ds_root / 'HR'
            for name in self._list_video_names(hr_dir):
                hr_path = self._hr_path(hr_dir, name)
                n = self._get_n_frames(hr_dir, name)
                if n > 0:
                    h, w = self._get_resolution(hr_dir, name)
                    videos.append({'name': name, 'n_frames': n,
                                   'variants': ['clean'], 'ds_root': ds_root,
                                   'h': h, 'w': w, 'hr_ext': hr_path.suffix})
        return videos

    def __getitem__(self, item) -> dict:
        video_name, _, center_frame, ds_root, gy, gx, *extra = item
        video_id = extra[0] if len(extra) > 0 else None

        self.load_video(video_name, ds_root=ds_root)

        half = self.frames // 2
        ps = self.patch_size
        b = self.patch_buffer
        crop_y = gy * ps
        crop_x = gx * ps

        def _maybe_pad(arr):
            if b:
                return np.pad(arr, ((b, b), (b, b), (0, 0)), mode='reflect')
            return arr

        hr_yuvs = []
        for i in range(center_frame - half, center_frame + half + 1):
            i_clamped = max(0, min(i, len(self._hr_cache) - 1))
            hr_yuvs.append(_maybe_pad(
                self._hr_cache[i_clamped][crop_y:crop_y + ps, crop_x:crop_x + ps]))

        hr_prev_frame = max(0, center_frame - 1)
        hr_prev_yuv = _maybe_pad(
            self._hr_cache[hr_prev_frame][crop_y:crop_y + ps, crop_x:crop_x + ps])

        all_yuv = np.stack(hr_yuvs + [hr_prev_yuv], axis=0)
        all_ictcp = _yuv_to_ictcp(all_yuv)

        n_temporal = self.frames
        lr_t = all_ictcp[:n_temporal].float()
        hr_t = all_ictcp[0].float()
        hr_prev_t = all_ictcp[n_temporal].float()

        out = {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}
        if video_id is not None:
            out['video_id'] = video_id
        return out


class PreprocessedVideoDataset(Dataset):
    """Reads pre-trained ICtCp patches from generate_dataset.py preprocessing.

    Directory layout::

        data/{name}/
            HR_seg0.mkv  seg0.mp4          ← original videos
            seg0/                           ← preprocessed patches
                hr.npy     (30, 3, 512, 512) float16
                lr.npy     (30, 3, 512, 512) float16
                meta.json

    Each segment contributes (num_frames - frames + 1) = 22 training samples.
    """
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
    """Real-time CPU-decoded dataset.  Returns raw YUV numpy arrays
    (no GPU ICtCp — that is done by the training loop after collation).

    Uses LazyFrameRange per segment (sliding-window decode cache).

    Directory layout per segment::

        seg_name/
            HR.mkv       (FFV1, yuv444p12le, 512×512)
            LR.mkv       (FFV1, yuv420p*, 512×512)
            meta.json    (contains num_frames)
    """
    def __init__(self, data_root: str, frames: int = 9, max_cached_segments: int = 64):
        self.data_root = Path(data_root)
        self.frames = frames
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
    """Wrap a Dataset, horizontally flipping frames when index < 0.

    Used by PreprocessedBatchSampler with segment_repeat >= 2 to mirror
    alternate passes at zero extra decode cost.

    Handles both torch.Tensor (CHW) and np.ndarray (HWC) formats.
    """
    def __init__(self, wrapped: Dataset):
        self.wrapped = wrapped

    @staticmethod
    def _flip_horizontal(x):
        if isinstance(x, torch.Tensor):
            # [..., 3, H, W] — flip W dim
            return torch.flip(x, dims=[-1])
        # [..., H, W, 3] — flip W dim
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
    """Groups consecutive center-frames from the same segment into batches,
    preserving temporal order for SSM state propagation.

    Yields batches of *batch_size* consecutive center-frame indices from
    one segment, then moves to the next segment.  Segment order is
    shuffled; within a segment the frame order is monotonic.
    """
    def __init__(self, dataset: Dataset, batch_size: int, segment_repeat: int = 1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.segment_repeat = segment_repeat
        self._groups: dict[int, list[int]] | None = None

    def _build_groups(self) -> dict[int, list[int]]:
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        if hasattr(self.dataset, 'datasets'):  # ConcatDataset
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
        self._groups = None  # free groups after building batch list

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


class VideoBatchSampler(BatchSampler):
    """Yields batches where all items share one video + variant.
    Each (frame, gy, gx) is an independent sample; grid cells cover
    the full frame without overlap. Batches from all videos are
    shuffled globally so consecutive batches come from different videos.
    """
    def __init__(self, dataset, batch_size, shuffle_variants=False, clip_repeat=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants
        self.clip_repeat = clip_repeat

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        all_batches = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            pool = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        pool.append((v['name'], variant, f, v['ds_root'], yi, xi))
            random.shuffle(pool)
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)

        for _ in range(self.clip_repeat):
            random.shuffle(all_batches)
            yield from all_batches

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size) * self.clip_repeat


class ValVideoBatchSampler(BatchSampler):
    """Covers every valid (centre_frame, gy, gx) deterministically."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        for v in self.dataset.videos:
            variant = v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            pool = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        pool.append((v['name'], variant, f, v['ds_root'], yi, xi))
            for i in range(0, len(pool), self.batch_size):
                chunk = pool[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                yield chunk

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size)


class SequentialVideoBatchSampler(BatchSampler):
    """Temporally-ordered batches per video, shuffled video order.

    For each video, all grid cells of all valid centre frames are
    enumerated sequentially and chunked into batches.  The Mamba SSM
    state persists across all frames+grid cells of one video.
    """
    def __init__(self, dataset, batch_size, shuffle_variants=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_variants = shuffle_variants

    def __iter__(self):
        half = self.dataset.frames // 2
        ps = self.dataset.patch_size

        video_batch_lists = []
        for v in self.dataset.videos:
            variant = random.choice(v['variants']) if self.shuffle_variants else v['variants'][0]
            valid_end = v['n_frames'] - half
            if valid_end <= half:
                continue
            gy = v['h'] // ps
            gx = v['w'] // ps
            if gy == 0 or gx == 0:
                continue
            video_id = hash(f"{v['name']}_{v['ds_root']}_{variant}") & 0x7FFFFFFF

            items = []
            for f in range(half, valid_end):
                for yi in range(gy):
                    for xi in range(gx):
                        items.append((v['name'], variant, f, v['ds_root'], yi, xi, video_id))

            batches = []
            for i in range(0, len(items), self.batch_size):
                chunk = items[i:i + self.batch_size]
                if len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
            if batches:
                video_batch_lists.append(batches)

        random.shuffle(video_batch_lists)
        for batches in video_batch_lists:
            yield from batches

    def __len__(self):
        total = 0
        ps = self.dataset.patch_size
        for v in self.dataset.videos:
            gy = v['h'] // ps
            gx = v['w'] // ps
            total += max(0, v['n_frames'] - self.dataset.frames + 1) * gy * gx
        return max(1, total // self.batch_size)


def collate_video(batch: list[dict]) -> dict[str, Any]:
    lr_frames = torch.stack([item['lr_frames'] for item in batch], dim=0)
    hr = torch.stack([item['hr'] for item in batch], dim=0)

    out = {'lr_frames': lr_frames, 'hr': hr}
    video_id = batch[0].get('video_id')
    if video_id is not None:
        out['video_id'] = video_id
    return out


def collate_raw(batch: list[dict]) -> dict[str, Any]:
    """Collate for RawVideoDataset: stacks numpy YUV frames → CPU tensors."""
    B = len(batch)
    lr_shape = batch[0]['lr_frames'].shape
    hr_shape = batch[0]['hr'].shape
    lr_dtype = batch[0]['lr_frames'].dtype
    hr_dtype = batch[0]['hr'].dtype

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
    max_cached_segments: int = 64,
) -> DataLoader:
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
