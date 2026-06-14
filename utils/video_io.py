import subprocess
import numpy as np
from pathlib import Path
from typing import Optional


def extract_frames(video_path: str, out_dir: str, fmt: str = 'png') -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f'frame_%06d.{fmt}')
    subprocess.run([
        'ffmpeg', '-i', video_path, '-pix_fmt', 'rgb24',
        '-q:v', '2', pattern,
    ], check=True, capture_output=True)
    return sorted([str(p) for p in out_dir.glob(f'*.{fmt}')])


def compress_av1(
    input_pattern: str, output_path: str, fps: int = 30,
    encoder: str = 'svt', crf: int = 40, pix_fmt: str = 'yuv420p',
) -> None:
    if encoder == 'svt':
        subprocess.run([
            'ffmpeg', '-framerate', str(fps), '-i', input_pattern,
            '-c:v', 'libsvtav1', '-crf', str(crf),
            '-pix_fmt', pix_fmt, output_path,
        ], check=True, capture_output=True)
    elif encoder == 'aom':
        subprocess.run([
            'ffmpeg', '-framerate', str(fps), '-i', input_pattern,
            '-c:v', 'libaom-av1', '-crf', str(crf),
            '-cpu-used', '4', '-pix_fmt', pix_fmt, output_path,
        ], check=True, capture_output=True)


def frames_to_video(frame_dir: str, output_path: str, fps: int = 30) -> None:
    subprocess.run([
        'ffmpeg', '-framerate', str(fps), '-i', f'{frame_dir}/frame_%06d.png',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', output_path,
    ], check=True, capture_output=True)
