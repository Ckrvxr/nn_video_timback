import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .blosc_cache import BloscCache
from .video_loader import load_video_frames_raw


class FrameCache:
    """Per-worker LRU cache for decoded video frames.

    Supports two modes:
      *decode_allowed=True*  — fallback to PyAV when BloscCache misses.
                                Used only during pre-warming in the main process.
      *decode_allowed=False* — BloscCache-only.  If the .blp file does not
                                exist, raises RuntimeError.  This is the
                                worker-mode (training) which must never block
                                on expensive video decoding.
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

    def _init(self, max_videos=None):
        self._cache = {}
        self._order = []
        self._max_videos = int(max_videos) if max_videos is not None else int(
            os.environ.get('FRAME_CACHE_MAX_VIDEOS', 3))
        self._blosc_cache = None
        self._cache_lock = threading.Lock()
        self._decode_allowed = True

    def set_cache_root(self, cache_root: str | Path | None):
        self._blosc_cache = BloscCache(cache_root) if cache_root else None

    def set_decode_allowed(self, allowed: bool):
        """When False, _load() reads BloscCache only — no PyAV fallback."""
        self._decode_allowed = allowed

    def _load(self, video_path: str) -> list[np.ndarray]:
        """Load video frames, preferring BloscCache over PyAV decoding.

        Raises RuntimeError if BloscCache misses and decoding is not allowed
        (worker mode).
        """
        # ── BloscCache hit path (fast) ──
        if self._blosc_cache:
            result = self._blosc_cache.get(video_path)
            if result is not None:
                arr, h, w, n_frames = result
                return self._blp_to_yuv444(arr, h, w, n_frames)

        # ── Decode path (slow — only in pre-warm phase) ──
        if self._decode_allowed:
            raw_frames = load_video_frames_raw(video_path)
            if self._blosc_cache:
                try:
                    self._blosc_cache.put_raw(video_path, raw_frames)
                except Exception:
                    pass
            return self._raw_to_yuv444(raw_frames)

        # ── Worker mode: BloscCache miss + decode prohibited ──
        raise RuntimeError(
            f'BloscCache miss for {video_path} in worker mode. '
            'Run training with warm_cache=True or manually run '
            'scripts/precache_frames.py first.'
        )

    @staticmethod
    def _blp_to_yuv444(arr_1d: np.ndarray, h: int, w: int, n_frames: int) -> list[np.ndarray]:
        """Convert YYYYY-UUUUU-VVVVV 1D array → list of YUV444 frames."""
        n = h * w
        n_uv = (h // 2) * (w // 2)
        y_all = arr_1d[:n_frames * n].reshape(n_frames, h, w)
        u_all = arr_1d[n_frames * n: n_frames * (n + n_uv)].reshape(n_frames, h // 2, w // 2)
        v_all = arr_1d[n_frames * (n + n_uv): n_frames * (n + 2 * n_uv)].reshape(
            n_frames, h // 2, w // 2)
        out = []
        for i in range(n_frames):
            u = cv2.resize(u_all[i], (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v_all[i], (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(np.stack([y_all[i], u, v], axis=-1))
        return out

    @staticmethod
    def _raw_to_yuv444(raw_frames: list) -> list[np.ndarray]:
        """Convert list of (Y,U,V) tuples → list of YUV444 frames."""
        out = []
        for y, u, v in raw_frames:
            h, w = y.shape
            u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
            out.append(np.stack([y, u, v], axis=-1))
        return out

    def get_or_load(self, hr_path: Path, lr_path: Path):
        key = (str(hr_path), str(lr_path))

        with self._cache_lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]

        # Heavy I/O / decompression happens outside the lock.
        hr_frames = self._load(str(hr_path))
        lr_frames = self._load(str(lr_path))

        with self._cache_lock:
            self._cache[key] = (hr_frames, lr_frames)
            self._order.append(key)
            if len(self._order) > self._max_videos:
                victim = self._order.pop(0)
                del self._cache[victim]
            return hr_frames, lr_frames

    def clear(self):
        with self._cache_lock:
            self._cache.clear()
            self._order.clear()
