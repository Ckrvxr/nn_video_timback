import json
import shutil
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np


def _read_plane(plane) -> np.ndarray:
    """Read a YUV plane, stripping line_size alignment padding."""
    buf = np.frombuffer(bytes(plane), dtype=np.uint8).reshape(plane.height, plane.line_size)
    return buf[:, :plane.width].copy()


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode video into list of YUV444 uint8 arrays (H×W×3).

    Y channel is native from decoder (lossless). U/V are cubic
    upsampled from half-resolution YUV420 planes.
    """
    frames = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            h, w = frame.height, frame.width
            y = _read_plane(frame.planes[0])
            u = cv2.resize(
                _read_plane(frame.planes[1]),
                (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(
                _read_plane(frame.planes[2]),
                (w, h), interpolation=cv2.INTER_LINEAR)
            frames.append(np.stack([y, u, v], axis=-1))
    return frames


def load_video_frames_raw(video_path: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Decode video into list of (Y_plane, U_plane, V_plane) tuples.

    Each plane is uint8 with native YUV420 resolution:
      Y: (H, W), U/V: (H/2, W/2).  No upsampling — lossless decode path.
    """
    out = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            h, w = frame.height, frame.width
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
