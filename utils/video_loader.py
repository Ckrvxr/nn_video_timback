import json
import shutil
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np


def _read_plane(plane) -> np.ndarray:
    """Read a YUV plane efficiently using buffer protocol."""
    # np.frombuffer(plane) avoids the heavy bytes(plane) copy
    buf = np.frombuffer(plane, dtype=np.uint8).reshape(plane.height, plane.line_size)
    if plane.line_size == plane.width:
        return buf.copy()
    return buf[:, :plane.width].copy()


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode video into list of YUV444 uint8 arrays (H×W×3)."""
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        # Enable multi-threaded decoding
        stream.thread_type = 'AUTO'
        
        for frame in container.decode(video=0):
            h, w = frame.height, frame.width
            # Use PyAV's built-in fast conversion if possible, 
            # but for YUV420->YUV444 manual might be more precise for your VSR
            y = _read_plane(frame.planes[0])
            u = cv2.resize(_read_plane(frame.planes[1]), (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(_read_plane(frame.planes[2]), (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(np.stack([y, u, v], axis=-1))
    return frames


def load_video_frames_raw_generator(video_path: str):
    """Generator version to save memory: yields (Y, U, V) tuples one by one."""
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        for frame in container.decode(video=0):
            y = _read_plane(frame.planes[0])
            u = _read_plane(frame.planes[1])
            v = _read_plane(frame.planes[2])
            yield y, u, v


def load_video_frames_raw(video_path: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode video into list of (Y, U, V) tuples."""
    out = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        
        for frame in container.decode(video=0):
            y = _read_plane(frame.planes[0])
            u = _read_plane(frame.planes[1])
            v = _read_plane(frame.planes[2])
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
