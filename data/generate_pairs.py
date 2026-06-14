"""
Generate AV1 compressed frame pairs from high-quality video sources.
Generates multiple compression variants (encoder × CRF × preset) so the
dataset can randomly pick a variant at training time.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Generate AV1 compressed frame pairs')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='./data/datasets')
    parser.add_argument('--encoder', type=str, choices=['svt', 'aom', 'all'], default='all')
    parser.add_argument('--crf', type=int, nargs='+', default=[30, 40, 50, 60])
    parser.add_argument('--presets', type=int, nargs='+', default=[6, 8, 10],
                        help='SVT-AV1 presets (0-13, lower=slower/better)')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--gop', type=int, default=160,
                        help='Keyframe interval (smaller=more keyframes)')
    return parser.parse_args()


def extract_frames(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / 'frame_%06d.png')
    subprocess.run([
        'ffmpeg', '-i', str(video_path), '-pix_fmt', 'rgb24',
        '-q:v', '2', '-y', pattern,
    ], check=True, capture_output=True)
    return sorted(out_dir.glob('*.png'))


def variant_dirname(encoder: str, crf: int, preset: int) -> str:
    return f'{encoder}_crf{crf}_p{preset}'


def compress_av1(frame_pattern: str, output_path: Path, encoder: str,
                 crf: int, preset: int, fps: int = 30, gop: int = 160):
    if encoder == 'svt':
        codec = 'libsvtav1'
        extra = ['-svtav1-params', f'tune=0:preset={preset}']
    else:
        codec = 'libaom-av1'
        cpu = max(0, min(6, preset // 2))
        extra = ['-cpu-used', str(cpu)]

    subprocess.run([
        'ffmpeg', '-framerate', str(fps), '-i', frame_pattern,
        '-c:v', codec, '-crf', str(crf),
        '-g', str(gop),
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


def process_video(video_path: Path, hr_root: Path, lr_root: Path,
                  encoders: list[str], crf_values: list[int],
                  presets: list[int], fps: int, gop: int):
    name = video_path.stem
    print(f'Processing {name}...')

    hr_frames_dir = hr_root / name
    hr_frames = extract_frames(video_path, hr_frames_dir)
    frame_pattern = str(hr_frames_dir / 'frame_%06d.png')
    print(f'  Extracted {len(hr_frames)} frames')

    for enc in encoders:
        for crf in crf_values:
            for preset in presets:
                subdir = variant_dirname(enc, crf, preset)
                compressed_video = lr_root / subdir / f'{name}.mp4'
                compressed_video.parent.mkdir(parents=True, exist_ok=True)

                print(f'  Encoding {enc} CRF {crf} preset {preset}...')
                compress_av1(frame_pattern, compressed_video, enc, crf, preset, fps, gop)

                compressed_frames_dir = lr_root / subdir / name
                decompress_to_frames(compressed_video, compressed_frames_dir)
                print(f'    Decompressed: {len(list(compressed_frames_dir.glob("*.png")))} frames')


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output)
    hr_root = output_root / 'HR'
    lr_root = output_root

    encoders = ['svt', 'aom'] if args.encoder == 'all' else [args.encoder]

    if input_path.is_file():
        process_video(input_path, hr_root, lr_root, encoders,
                      args.crf, args.presets, args.fps, args.gop)
    else:
        for video in sorted(input_path.glob('*.mp4')) + sorted(input_path.glob('*.mkv')) + sorted(input_path.glob('*.mov')):
            process_video(video, hr_root, lr_root, encoders,
                          args.crf, args.presets, args.fps, args.gop)

    print('Done!')


if __name__ == '__main__':
    main()
