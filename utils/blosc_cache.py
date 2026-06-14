from pathlib import Path

import blosc
import cv2
import numpy as np


class BloscCache:
    """Compressed frame cache on disk.
    Each video → single .blp file.
    Format: YUV420 planar (Y+U+V separated) + lz4hc."""

    def __init__(self, cache_root: str | Path, cname: str = 'lz4hc', clevel: int = 5,
                 uv_interp: int = cv2.INTER_CUBIC):
        self.cache_root = Path(cache_root)
        self.cname = cname
        self.clevel = clevel
        self.uv_interp = uv_interp

    def cache_path(self, video_path: str | Path) -> Path:
        p = Path(video_path)
        return self.cache_root / p.parent.name / f'{p.stem}.blp'

    def get(self, video_path: str | Path) -> list[np.ndarray] | None:
        cp = self.cache_path(video_path)
        if not cp.exists():
            return None
        with open(cp, 'rb') as f:
            hdr = f.read(4)
            compressed = f.read()
        h, w = np.frombuffer(hdr, dtype=np.uint16)
        arr = blosc.unpack_array(compressed)
        n = int(h) * int(w)
        n_uv = (int(h) // 2) * (int(w) // 2)
        out = []
        for i in range(arr.shape[0]):
            y = arr[i, :n].reshape(int(h), int(w))
            u = cv2.resize(arr[i, n:n + n_uv].reshape(int(h) // 2, int(w) // 2),
                           (int(w), int(h)), interpolation=self.uv_interp)
            v = cv2.resize(arr[i, n + n_uv:].reshape(int(h) // 2, int(w) // 2),
                           (int(w), int(h)), interpolation=self.uv_interp)
            out.append(np.stack([y, u, v], axis=-1))
        return out

    def put(self, video_path: str | Path, frames: list[np.ndarray]):
        """Store YUV444 frames (H,W,3) → YUV420 planar → .blp."""
        cp = self.cache_path(video_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        n = h * w
        n_uv = (h // 2) * (w // 2)
        rows = []
        for f in frames:
            y = f[:, :, 0].ravel()
            u = cv2.resize(f[:, :, 1], (w // 2, h // 2),
                           interpolation=self.uv_interp).ravel()
            v = cv2.resize(f[:, :, 2], (w // 2, h // 2),
                           interpolation=self.uv_interp).ravel()
            rows.append(np.concatenate([y, u, v]))
        arr = np.stack(rows, axis=0)
        compressed = blosc.pack_array(
            arr, cname=self.cname, clevel=self.clevel, shuffle=blosc.NOSHUFFLE)
        with open(cp, 'wb') as f:
            f.write(np.array([h, w], dtype=np.uint16).tobytes())
            f.write(compressed)

    def put_raw(self, video_path: str | Path, raw_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """Store native YUV420 planes directly — no YUV444 intermediate.

        Each tuple is (Y_plane (H,W), U_plane (H/2,W/2), V_plane (H/2,W/2)).
        Eliminates the up→down redundant cycle for U/V.
        """
        cp = self.cache_path(video_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        h, w = raw_frames[0][0].shape[:2]
        n = h * w
        n_uv = (h // 2) * (w // 2)
        rows = []
        for y_plane, u_plane, v_plane in raw_frames:
            rows.append(np.concatenate([
                y_plane.ravel(), u_plane.ravel(), v_plane.ravel()]))
        arr = np.stack(rows, axis=0)
        compressed = blosc.pack_array(
            arr, cname=self.cname, clevel=self.clevel, shuffle=blosc.NOSHUFFLE)
        with open(cp, 'wb') as f:
            f.write(np.array([h, w], dtype=np.uint16).tobytes())
            f.write(compressed)
