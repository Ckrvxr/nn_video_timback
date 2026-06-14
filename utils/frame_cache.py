import threading
from pathlib import Path

import numpy as np

from .blosc_cache import BloscCache
from .video_loader import load_video_frames


class FrameCache:
    """Module-level LRU cache for decoded video frames.
    Singleton — persists across dataset instances and epochs.
    Optionally backed by BloscCache for disk-persisted compressed frames."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self, max_videos=2):
        self._cache = {}
        self._order = []
        self._max_videos = max_videos
        self._blosc_cache = None

    def set_cache_root(self, cache_root: str | Path | None):
        self._blosc_cache = BloscCache(cache_root) if cache_root else None

    def _load(self, video_path: str) -> list[np.ndarray]:
        if self._blosc_cache:
            try:
                cached = self._blosc_cache.get(video_path)
                if cached is not None:
                    return [np.require(f, requirements=['OWNDATA']) for f in cached]
            except RuntimeError:
                print(f'[WARN] Cache corrupted, falling back to PyAV: {video_path}', flush=True)
        return load_video_frames(video_path)

    def get_or_load(self, hr_path: Path, lr_path: Path):
        key = (str(hr_path), str(lr_path))
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

        hr_frames = self._load(str(hr_path))
        lr_frames = self._load(str(lr_path))

        self._cache[key] = (hr_frames, lr_frames)
        self._order.append(key)

        if len(self._order) > self._max_videos:
            victim = self._order.pop(0)
            del self._cache[victim]

        return hr_frames, lr_frames

    def clear(self):
        self._cache.clear()
        self._order.clear()
