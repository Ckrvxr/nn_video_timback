import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .frame_cache import LazyFrameRange
from .ictcp import yuv_to_ictcp_np_batch
from .video_probe import probe_frame_count, probe_resolution


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
        
        # Memory optimization: limit cache size
        self._max_cache_size = 2  # Only keep 2 videos in memory at once

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

        # Clear previous cache to free memory
        if self._cache_key is not None:
            self._hr_cache = []
            self._lr_cache = []

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
        all_ictcp = yuv_to_ictcp_np_batch(all_yuv)

        n_temporal = self.frames
        lr_t = all_ictcp[:n_temporal].float()
        hr_t = all_ictcp[n_temporal].float()
        hr_prev_t = all_ictcp[n_temporal + 1].float()

        out = {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}
        if video_id is not None:
            out['video_id'] = video_id
        return out



