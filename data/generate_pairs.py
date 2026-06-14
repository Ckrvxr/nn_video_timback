"""
Generate AV1 compressed frame pairs from high-quality video sources.
Supports SVT-AV1 and libaom-av1 encoders at multiple CRF levels.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate AV1 compressed frame pairs')
    parser.add_argument('--input', type=str, required=True, help='Input video directory or single video')
    parser.add_argument('--output', type=str, default='./data/datasets', help='Output dataset directory')
    parser.add_argument('--encoder', type=str, choices=['svt', 'aom', 'all'], default='all')
    parser.add_argument('--crf', type=int, nargs='+', default=[30, 40, 50, 60])
    parser.add_argument('--fps', type=int, default=30, help='Output frame rate')
    return parser.parse_args()


def extract_frames(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / 'frame_%06d.png')
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-pix_fmt', 'rgb24',
        '-q:v', '2', '-y', pattern,
    ], check=True, capture_output=True)
    return sorted(out_dir.glob('*.png'))


def compress_av1(frame_pattern: str, output_path: Path, encoder: str, crf: int, fps: int = 30):
    codec = 'libsvtav1' if encoder == 'svt' else 'libaom-av1'
    extra = ['-svtav1-params', 'tune=0'] if encoder == 'svt' else ['-cpu-used', '4']

    subprocess.run([
        'ffmpeg', '-framerate', str(fps), '-i', frame_pattern,
        '-c:v', codec, '-crf', str(crf),
        '-pix_fmt', 'yuv420p', *extra, '-y', str(output_path),
    ], check=True, capture_output=True)


def decompress_to_frames(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / 'frame_%06d.png')
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-pix_fmt', 'rgb24',
        '-q:v', '2', '-y', pattern,
    ], check=True, capture_output=True)
    return sorted(out_dir.glob('*.png'))


def process_video(video_path: Path, hr_root: Path, lr_root: Path, encoders: list[str], crf_values: list[int], fps: int):
    name = video_path.stem
    print(f'Processing {name}...')

    hr_frames_dir = hr_root / name
    hr_frames = extract_frames(video_path, hr_frames_dir)
    frame_pattern = str(hr_frames_dir / 'frame_%06d.png')
    print(f'  Extracted {len(hr_frames)} frames')

    for enc in encoders:
        for crf in crf_values:
            compressed_video = lr_root / f'{enc}_crf{crf}' / f'{name}.mkv'
            compressed_video.parent.mkdir(parents=True, exist_ok=True)

            print(f'  Encoding {enc} CRF {crf}...')
            compress_av1(frame_pattern, compressed_video, enc, crf, fps)

            compressed_frames_dir = lr_root / f'{enc}_crf{crf}' / name
            decompress_to_frames(compressed_video, compressed_frames_dir)
            print(f'  Decompressed CRF {crf}: {len(list(compressed_frames_dir.glob("*.png")))} frames')


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    hr_root = output_root / 'HR'
    lr_root = output_root

    encoders = ['svt', 'aom'] if args.encoder == 'all' else [args.encoder]

    if input_path.is_file():
        process_video(input_path, hr_root, lr_root, encoders, args.crf, args.fps)
    else:
        for video in sorted(input_path.glob('*.mp4')) + sorted(input_path.glob('*.mkv')) + sorted(input_path.glob('*.mov')):
            process_video(video, hr_root, lr_root, encoders, args.crf, args.fps)

    print('Done!')


if __name__ == '__main__':
    main()
