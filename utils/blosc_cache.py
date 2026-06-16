import os
from pathlib import Path

import blosc
import cv2
import numpy as np


class BloscCache:
    """Compressed frame cache on disk.
    Each video → single .blp file.
    Format: YYYYY-UUUUU-VVVVV (all Y planes first, then all U, then all V).
    Header: uint16[H, W, N_frames] (6 bytes).
    Compression: LZ4 + byte SHUFFLE."""

    def __init__(self, cache_root: str | Path, cname: str = 'lz4', clevel: int = 3,
                 uv_interp: int = cv2.INTER_LINEAR):
        self.cache_root = Path(cache_root)
        self.cname = cname
        self.clevel = clevel
        self.uv_interp = uv_interp

    def cache_path(self, video_path: str | Path) -> Path:
        p = Path(video_path)
        return self.cache_root / p.parent.name / f'{p.stem}.blp'

    def get(self, video_path: str | Path) -> tuple[np.ndarray, int, int, int] | None:
        """Read .blp → (arr_1d, h, w, n_frames) without any conversion.

        arr_1d is the raw YYYYY-UUUUU-VVVVV 1D array.
        Caller is responsible for reshaping and YUV420→YUV444 conversion.
        """
        cp = self.cache_path(video_path)
        if not cp.exists():
            return None
        try:
            with open(cp, 'rb') as f:
                hdr = f.read(6)
                compressed = f.read()
            h, w, n_frames = np.frombuffer(hdr, dtype=np.uint16)
            arr = blosc.unpack_array(compressed)  # 1D: [Y_all, U_all, V_all]
        except Exception as e:
            raise RuntimeError(f'Cache decompress failed: {e}  file: {cp}') from e
        return arr, int(h), int(w), int(n_frames)

    def put_raw(self, video_path: str | Path, raw_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]]):
        """Store native YUV420 planes as YYYYY-UUUUU-VVVVV.

        All Y planes flattened and concatenated first, then all U, then all V.
        Header: uint16[H, W, N_frames].
        """
        cp = self.cache_path(video_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        h, w = raw_frames[0][0].shape[:2]
        n_frames = len(raw_frames)

        y_all = np.concatenate([y.ravel() for y, _, _ in raw_frames])
        u_all = np.concatenate([u.ravel() for _, u, _ in raw_frames])
        v_all = np.concatenate([v.ravel() for _, _, v in raw_frames])
        arr = np.concatenate([y_all, u_all, v_all])

        try:
            compressed = blosc.pack_array(
                arr, cname=self.cname, clevel=self.clevel, shuffle=blosc.SHUFFLE)
        except Exception as e:
            raise RuntimeError(f'Cache compress failed: {e}  file: {cp}') from e
        with open(cp, 'wb') as f:
            f.write(np.array([h, w, n_frames], dtype=np.uint16).tobytes())
            f.write(compressed)


