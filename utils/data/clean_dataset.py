from pathlib import Path

import numpy as np

from .frame_cache import LazyFrameRange
from .compressed_dataset import CompressedVideoDataset
from .ictcp import yuv_to_ictcp_np_batch


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
        all_ictcp = yuv_to_ictcp_np_batch(all_yuv)

        n_temporal = self.frames
        lr_t = all_ictcp[:n_temporal].float()
        hr_t = all_ictcp[0].float()
        hr_prev_t = all_ictcp[n_temporal].float()

        out = {'lr_frames': lr_t, 'hr': hr_t, 'hr_prev': hr_prev_t}
        if video_id is not None:
            out['video_id'] = video_id
        return out
