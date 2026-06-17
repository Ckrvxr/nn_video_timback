import shutil
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np

from models.components.color_space import yuv_to_ictcp_np


def read_plane(plane, bits: int = 8) -> np.ndarray:
    """Read a YUV plane, handling 8-bit (uint8) and 10/12-bit (uint16)."""
    if bits <= 8:
        dtype = np.uint8
        sample_bytes = 1
    else:
        dtype = np.uint16
        sample_bytes = 2
    buf = np.frombuffer(plane, dtype=dtype).reshape(
        plane.height, plane.line_size // sample_bytes)
    if plane.line_size // sample_bytes == plane.width:
        return buf.copy()
    return buf[:, :plane.width].copy()


def plane_bits(frame, plane_idx: int) -> int:
    """Detect bit depth from frame format."""
    try:
        return frame.format.components[plane_idx].bits
    except Exception:
        return 8


def frame_to_yuv(frame) -> np.ndarray:
    """Decode a single PyAV frame → YUV444 H×W×3 (uint8 or uint16)."""
    bits = plane_bits(frame, 0)
    y = read_plane(frame.planes[0], bits)
    u = read_plane(frame.planes[1], bits)
    v = read_plane(frame.planes[2], bits)
    if u.shape != y.shape:
        u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.stack([y, u, v], axis=-1)


# Backward compat aliases
_read_plane = read_plane
_plane_bits = plane_bits


def load_video_frames_ictcp(video_path: str, batch_size: int = 1) -> list[np.ndarray]:
    """Decode video → YUV444 → PQ ICtCp float32 H×W×3.

    Batch processing reduces Python overhead from multiple frames.
    batch_size=1 behaves like the original per-frame processing.
    """
    import gc
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        batch_yuv = []
        batch_bits = None
        for frame in container.decode(video=0):
            bits = _plane_bits(frame, 0)
            y = _read_plane(frame.planes[0], bits)
            u = _read_plane(frame.planes[1], bits)
            v = _read_plane(frame.planes[2], bits)
            if u.shape != y.shape:
                u = cv2.resize(u, (y.shape[1], y.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (y.shape[1], y.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            batch_yuv.append(np.stack([y, u, v], axis=-1))
            batch_bits = bits

            if len(batch_yuv) >= batch_size:
                batch_arr = np.stack(batch_yuv, axis=0)
                frames.extend(yuv_to_ictcp_np(batch_arr, bits=batch_bits))
                batch_yuv.clear()
                gc.collect()

        if batch_yuv:
            batch_arr = np.stack(batch_yuv, axis=0)
            frames.extend(yuv_to_ictcp_np(batch_arr, bits=batch_bits))
    return frames


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode video into list of YUV444 uint8 arrays (H×W×3)."""
    return load_video_frame_range(video_path, 0, 1<<30)


def load_video_frame_range(video_path: str, start: int, end: int) -> list[np.ndarray]:
    """Decode frames [start, end) from video, return list of YUV uint8/uint16 (H×W×3).

    Uses PyAV seek + PTS-based index, with sequential fallback for
    frames missing PTS (extremely rare for AV1).
    """
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        n_wanted = end - start

        if start > 0:
            fps = float(stream.average_rate or 30)
            seek_pts = int(start / fps / float(stream.time_base))
            container.seek(seek_pts, stream=stream)

        kept = 0
        for frame in container.decode(video=0):
            if kept >= n_wanted:
                break

            if frame.pts is not None:
                fps = float(stream.average_rate or 30)
                frame_idx = int(round(frame.pts * frame.time_base * fps))
                if frame_idx < start:
                    continue
                if frame_idx >= end:
                    break

            bits = _plane_bits(frame, 0)
            y = _read_plane(frame.planes[0], bits)
            u = _read_plane(frame.planes[1], bits)
            v = _read_plane(frame.planes[2], bits)
            if u.shape != y.shape:
                u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
            frames.append(np.stack([y, u, v], axis=-1))
            kept += 1
    return frames


def load_video_frames_raw(video_path: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode video into list of (Y, U, V) tuples.

    If the decoder outputs subsampled (420) planes, U/V are upsampled
    to full resolution so the result is always YUV444.
    """
    out = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        for frame in container.decode(video=0):
            bits = _plane_bits(frame, 0)
            y = _read_plane(frame.planes[0], bits)
            u = _read_plane(frame.planes[1], bits)
            v = _read_plane(frame.planes[2], bits)
            if u.shape != y.shape:
                u = cv2.resize(u, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, (y.shape[1], y.shape[0]), interpolation=cv2.INTER_LINEAR)
            out.append((y, u, v))
    return out


def _probe_frame_count_ffprobe(video_path: str) -> int | None:
    """Fast frame count using ffprobe if available."""
    if shutil.which('ffprobe') is None:
        return None
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out = result.stdout.strip()
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames',
                '-of', 'csv=p=0',
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out = result.stdout.strip()
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass
    return None


def probe_frame_count(video_path: str) -> int:
    """Return number of frames, using fast probes when possible."""
    n = _probe_frame_count_ffprobe(video_path)
    if n is not None and n > 0:
        return n
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        n = stream.frames
    return n
