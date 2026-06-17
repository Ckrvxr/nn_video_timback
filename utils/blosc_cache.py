import os
import struct
import tempfile
from pathlib import Path

import blosc
import numpy as np

_CACHE_MAGIC = b'BLPV4'
# sub-format byte: 1 = ICtCp BT.2100 (current), 0 = legacy/HPE (invalidated)
_CACHE_FORMAT = 1


class BloscCache:
    """Compressed ICtCp frame cache on disk.

    Per-frame random access via seek table.
    ``[BLPV4 magic][fmt][H][W][N][frame offset table][frame_0 blosc]…[frame_N blosc]``
    Each frame = independently blosc-compressed ``[I_rav | Ct_rav | Cp_rav]``
    stored as **half-precision float16**, decompressed to float32 ICtCp.
    Format byte differentiates BT.2100 ICtCp (1) from legacy HPE/IPT (0).
    """

    def __init__(self, cache_root: str | Path, cname: str = 'lz4', clevel: int = 3):
        self.cache_root = Path(cache_root)
        self.cname = cname
        self.clevel = clevel

    # ── paths ──────────────────────────────────────────────────────────

    def cache_path(self, video_path: str | Path) -> Path:
        p = Path(video_path)
        return self.cache_root / p.parent.name / f'{p.stem}.blp'

    # ── format detection ──────────────────────────────────────────────

    def is_valid_cache(self, video_path: str | Path) -> bool:
        """Check if cache file exists and has correct magic + format."""
        cp = self.cache_path(video_path)
        if not cp.exists():
            return False
        try:
            with open(cp, 'rb') as f:
                magic = f.read(len(_CACHE_MAGIC))
                if magic != _CACHE_MAGIC:
                    return False
                fmt = f.read(1)
                return fmt == struct.pack('B', _CACHE_FORMAT)
        except Exception:
            return False

    # ── ICtCp float16 caching helpers ─────────────────────────────────

    @staticmethod
    def _quantize_ictcp(ictcp: np.ndarray) -> np.ndarray:
        """ICtCp float32 H×W×3 → float16 ravel [I|Ct|Cp]."""
        i = ictcp[..., 0]
        ct = ictcp[..., 1]
        cp = ictcp[..., 2]
        return np.concatenate([i.ravel(), ct.ravel(), cp.ravel()]).astype(np.float16)

    @staticmethod
    def _dequantize_ictcp(arr: np.ndarray, h: int, w: int) -> np.ndarray:
        """float16 ravel [I|Ct|Cp] → ICtCp float32 H×W×3."""
        n = h * w
        i = arr[:n].astype(np.float32)
        ct = arr[n:2*n].astype(np.float32)
        cp = arr[2*n:].astype(np.float32)
        return np.stack([i.reshape(h, w), ct.reshape(h, w), cp.reshape(h, w)], axis=-1)

    # ── indexed writer ────────────────────────────────────────────────

    def put_frames(self, video_path: str | Path,
                   ictcp_frames: list[np.ndarray]):
        """Write ICtCp BT.2100 frames to cache.

        ``ictcp_frames`` is a list of H×W×3 float32 arrays in ICtCp space.
        Each frame is quantized to float16 and blosc-compressed.
        """
        cp = self.cache_path(video_path)
        cp.parent.mkdir(parents=True, exist_ok=True)

        h, w = ictcp_frames[0].shape[:2]
        n = len(ictcp_frames)

        frame_blocks = []
        for frame in ictcp_frames:
            block = self._quantize_ictcp(frame)
            packed = blosc.pack_array(block, cname=self.cname,
                                      clevel=self.clevel, shuffle=blosc.SHUFFLE)
            frame_blocks.append(packed)

        tmp = cp.with_suffix('.blp.tmp')
        try:
            with open(tmp, 'wb') as f:
                # magic + format byte + H, W, N
                f.write(_CACHE_MAGIC)
                f.write(struct.pack('B', _CACHE_FORMAT))
                f.write(struct.pack('<HH', h, w))
                f.write(struct.pack('<I', n))

                # placeholder for offset table (n+1 uint64)
                offset_pos = f.tell()
                f.write(b'\x00' * (n + 1) * 8)

                # write each frame block & record offsets
                offsets = []
                for blk in frame_blocks:
                    offsets.append(f.tell())
                    f.write(blk)
                offsets.append(f.tell())  # end-of-data sentinel

                # go back and fill real offsets
                f.seek(offset_pos)
                f.write(struct.pack(f'<{n + 1}Q', *offsets))

            os.replace(tmp, cp)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    # ── indexed header reader ─────────────────────────────────────────

    def _read_index(self, path: Path) -> tuple[int, int, int, list[int]]:
        """Read header → (h, w, n_frames, [offset_0, offset_1, …, end])."""
        with open(path, 'rb') as f:
            magic = f.read(len(_CACHE_MAGIC))
            if magic != _CACHE_MAGIC:
                raise ValueError(f'Not a valid cache file: {path}')
            fmt = f.read(1)
            if fmt != struct.pack('B', _CACHE_FORMAT):
                raise ValueError(f'Cache format mismatch (expected {_CACHE_FORMAT})')
            h, w = struct.unpack('<HH', f.read(4))
            n = struct.unpack('<I', f.read(4))[0]
            offsets = struct.unpack(f'<{n + 1}Q', f.read((n + 1) * 8))
        return int(h), int(w), int(n), list(offsets)

    # ── single-frame random access ───────────────────────────────────

    def get_frame(self, video_path: str | Path, frame_idx: int,
                  ) -> np.ndarray | None:
        """Load *one* frame as ICtCp float32 H×W×3."""
        cp = self.cache_path(video_path)
        if not cp.exists():
            return None

        try:
            h, w, n, offsets = self._read_index(cp)
        except Exception as e:
            raise RuntimeError(f'Corrupt cache: {cp}: {e}') from e

        if frame_idx < 0 or frame_idx >= n:
            raise IndexError(
                f'Frame {frame_idx} out of range [0, {n})  file: {cp}')

        with open(cp, 'rb') as f:
            f.seek(offsets[frame_idx])
            compressed = f.read(offsets[frame_idx + 1] - offsets[frame_idx])
        arr = blosc.unpack_array(compressed)
        return self._dequantize_ictcp(arr, h, w)

    def get_frame_range(self, video_path: str | Path, start: int, end: int,
                        ) -> list[np.ndarray] | None:
        """Load a consecutive range of frames as ICtCp float32 H×W×3."""
        cp = self.cache_path(video_path)
        if not cp.exists():
            return None
        return [self.get_frame(video_path, i) for i in range(start, end)]
