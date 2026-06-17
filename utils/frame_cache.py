import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .blosc_cache import BloscCache


class LazyFrameRange:
    """Indexable, lazily-loaded frame sequence backed by BloscCache.

    Frames are fetched on demand and cached in a sliding window so that
    memory stays proportional to ``window_size`` frames rather than the
    full video length.

    ``__getitem__`` and ``__len__`` behave like a list, making it a
    drop-in replacement for the old pre-loaded frame lists.
    """

    def __init__(self, blosc_cache: BloscCache | None, video_path: str,
                 n_frames: int, window_size: int = 17):
        self._bc = blosc_cache
        self._path = video_path
        self._n = n_frames
        self._ws = window_size
        self._cache: dict[int, np.ndarray] = {}

    # ── list-like interface ──────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._n))]

        if idx < 0:
            idx += self._n
        if idx < 0 or idx >= self._n:
            raise IndexError(
                f'Frame index {idx} out of range [0, {self._n})')

        if idx in self._cache:
            return self._cache[idx]

        self._evict_outside(idx)
        frame = self._fetch(idx)
        self._cache[idx] = frame
        return frame

    # ── window management ───────────────────────────────────────────

    def _evict_outside(self, new_idx: int):
        half = self._ws // 2
        lo = max(0, new_idx - half)
        hi = min(self._n, lo + self._ws)
        if hi - lo < self._ws:
            lo = max(0, hi - self._ws)

        for k in list(self._cache.keys()):
            if k < lo or k >= hi:
                del self._cache[k]

    def _fetch(self, idx: int) -> np.ndarray:
        if self._bc is not None:
            frame = self._bc.get_frame(self._path, idx)
            if frame is not None:
                return frame
        return self._fetch_live(idx)

    def _fetch_live(self, idx: int) -> np.ndarray:
        """Decode YUV→ICtCp from video file, caching a window around idx."""
        from .video_loader import load_video_frame_range
        from models.components.color_space import yuv_to_ictcp_np

        half = self._ws // 2
        lo = max(0, idx - half)
        hi = min(self._n, lo + self._ws)
        if hi - lo < self._ws:
            lo = max(0, hi - self._ws)

        yuv_frames = load_video_frame_range(self._path, lo, hi)
        batch = np.stack(yuv_frames, axis=0)

        bits = 8
        if batch.dtype == np.uint16:
            max_val = int(batch.max())
            bits = 12 if max_val > 1023 else 10

        ictcp_frames = yuv_to_ictcp_np(batch, bits=bits)

        for j in range(len(ictcp_frames)):
            self._cache[lo + j] = ictcp_frames[j]
        return self._cache[idx]

    def clear(self):
        self._cache.clear()


class FrameCache:
    """Per-process singleton that provides lazy frame access.

    To use:

    >>> fc = FrameCache()
    >>> fc.set_cache_root('data/datasets/foo/cache_yuv')
    >>> hr, lr = fc.get_or_load_frames('/path/to/HR/vid.mp4',
    ...                                '/path/to/LR/vid.mp4',
    ...                                n_frames=120)
    >>> frame = hr[42]       # lazy-loads frame 42
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._blosc_cache = None
        self._decode_allowed = True

    def set_cache_root(self, cache_root: str | Path | None):
        self._blosc_cache = BloscCache(cache_root) if cache_root else None

    def set_decode_allowed(self, allowed: bool):
        self._decode_allowed = allowed

    def get_or_load_frames(self, hr_path: str | Path, lr_path: str | Path,
                           n_frames: int, window_size: int = 17
                           ) -> tuple[LazyFrameRange, LazyFrameRange]:
        """Return lazy frame ranges for HR and LR.

        No frames are loaded until the returned objects are indexed.
        """
        hr = LazyFrameRange(self._blosc_cache, str(hr_path), n_frames, window_size)
        lr = LazyFrameRange(self._blosc_cache, str(lr_path), n_frames, window_size)
        return hr, lr

    def clear(self):
        pass
