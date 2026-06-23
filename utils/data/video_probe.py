import shutil
import subprocess

import av


def _probe_frame_count_ffprobe(video_path: str) -> int | None:
    if shutil.which('ffprobe') is None:
        return None

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0',
             video_path],
            capture_output=True, text=True, timeout=30, check=False,
        )
        out = result.stdout.strip().rstrip(',')
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_frames', '-show_entries', 'stream=nb_read_frames',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=60, check=False,
        )
        out = result.stdout.strip().rstrip(',')
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, timeout=30, check=False,
        )
        out = result.stdout.strip().rstrip(',')
        if out and out.isdigit():
            return int(out)
    except Exception:
        pass

    return None


def probe_frame_count(video_path: str) -> int:
    n = _probe_frame_count_ffprobe(video_path)
    if n is not None and n > 0:
        return n
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        n = stream.frames
    return n


def probe_resolution(video_path: str) -> tuple[int, int]:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        return stream.height, stream.width
