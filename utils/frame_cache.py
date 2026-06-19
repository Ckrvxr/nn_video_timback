import numpy as np


class LazyFrameRange:
    """Indexable, lazily-loaded frame sequence with sliding-window cache (stores YUV)."""

    def __init__(self, video_path: str, n_frames: int, window_size: int = 9):
        self._path = video_path
        self._n = n_frames
        self._ws = window_size
        self._cache: dict[int, np.ndarray] = {}

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
        return self._fetch_live(idx)

    def _fetch_live(self, idx: int) -> np.ndarray:
        """Decode YUV frames and store raw (no ICtCp conversion)."""
        from .video_loader import load_video_frame_range

        half = self._ws // 2
        lo = max(0, idx - half)
        hi = min(self._n, lo + self._ws)
        if hi - lo < self._ws:
            lo = max(0, hi - self._ws)

        needed = set(range(lo, hi))
        cached = set(self._cache.keys())
        missing = sorted(needed - cached)
        if not missing:
            return self._cache[idx]

        batches = []
        start = missing[0]
        for i in range(1, len(missing)):
            if missing[i] != missing[i - 1] + 1:
                batches.append((start, missing[i - 1] + 1))
                start = missing[i]
        batches.append((start, missing[-1] + 1))

        for batch_lo, batch_hi in batches:
            yuv_frames = load_video_frame_range(self._path, batch_lo, batch_hi)
            for j, f_idx in enumerate(range(batch_lo, batch_hi)):
                self._cache[f_idx] = yuv_frames[j]

        return self._cache[idx]

    def clear(self):
        self._cache.clear()
