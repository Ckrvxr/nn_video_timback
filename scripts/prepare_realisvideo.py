"""
Download RealisVideo-4K from HuggingFace and prepare AV1 compressed pairs.
Extracts HR frames at 1080p, encodes AV1 variants, decodes LR frames at 1080p.
"""
import argparse
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.generate_pairs import variant_dirname


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./data/datasets')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Max frames per video (0 = all)')
    parser.add_argument('--crf', type=int, nargs='+', default=[30, 50])
    parser.add_argument('--presets', type=int, nargs='+', default=[6, 10])
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def extract_hr_frames(video_path: Path, out_dir: Path, max_frames: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / 'frame_%06d.png')
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', 'scale=-2:1080,fps=30',
        '-pix_fmt', 'rgb24', '-q:v', '2', '-y',
    ]
    if max_frames > 0:
        cmd += ['-frames:v', str(max_frames)]
    cmd.append(pattern)
    subprocess.run(cmd, check=True, capture_output=True)
    return list(out_dir.glob('*.png'))


def encode_and_decode(hr_dir: Path, lr_root: Path, crf: int, preset: int):
    variant = variant_dirname('svt', crf, preset)
    out_video = lr_root / variant / f'{hr_dir.name}.mp4'
    out_frames = lr_root / variant / hr_dir.name

    if out_video.exists() and len(list(out_frames.glob('*.png'))) >= 3:
        return

    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_frames.mkdir(parents=True, exist_ok=True)

    pattern = str(hr_dir / 'frame_%06d.png')

    enc = subprocess.run([
        'ffmpeg', '-framerate', '30', '-i', pattern,
        '-c:v', 'libsvtav1', '-crf', str(crf),
        '-svtav1-params', f'tune=0:preset={preset}',
        '-pix_fmt', 'yuv420p', '-y', str(out_video),
    ], capture_output=True)
    if enc.returncode != 0:
        raise RuntimeError(
            f'Encode failed for {hr_dir.name} (CRF {crf} p{preset}): '
            + enc.stderr.decode(errors='replace'))

    dec = subprocess.run([
        'ffmpeg', '-i', str(out_video),
        '-vf', 'scale=-2:1080',
        '-pix_fmt', 'rgb24', '-q:v', '2', '-y',
        str(out_frames / 'frame_%06d.png'),
    ], capture_output=True)
    if dec.returncode != 0:
        raise RuntimeError(
            f'Decode failed for {hr_dir.name} (CRF {crf} p{preset}): '
            + dec.stderr.decode(errors='replace'))


def main():
    args = parse_args()
    output_root = Path(args.output)
    hr_root = output_root / 'HR'
    source_root = output_root / 'source_4k'

    if not source_root.exists() or not any(source_root.rglob('*.mp4')):
        print('Downloading RealisVideo-4K from HuggingFace...')
        from huggingface_hub import hf_hub_download
        zip_path = Path(hf_hub_download(
            repo_id="WisonZws/RealisVideo-4K",
            filename="realisvsr_4k_test.zip",
            repo_type="dataset",
        ))
        source_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(source_root)
        print(f'  Extracted to {source_root}')
    else:
        print(f'Source already exists at {source_root}')

    # zip extracts to realisvsr_4k_test/test/origin/
    roots = sorted(source_root.rglob('origin')) + [source_root]
    origin = next((r for r in roots if r.is_dir() and list(r.glob('*.mp4'))), source_root)
    videos = sorted(origin.glob('*.mp4'))
    print(f'Found {len(videos)} video files')

    print(f'Step 1: Extracting HR frames (1080p, max {args.max_frames} frames)...')
    hr_dirs = []
    t0 = time.perf_counter()
    for v in videos:
        name = v.stem
        hr_dir = hr_root / name
        if hr_dir.exists() and len(list(hr_dir.glob('*.png'))) >= 3:
            hr_dirs.append(hr_dir)
            continue
        frames = extract_hr_frames(v, hr_dir, args.max_frames)
        hr_dirs.append(hr_dir)
        print(f'  {name}: {len(frames)} frames')
    print(f'  Done in {time.perf_counter() - t0:.1f}s')

    variants = [(c, p) for c in args.crf for p in args.presets]
    total = len(hr_dirs) * len(variants)
    print(f'Step 2: Encoding {len(variants)} variants × {len(hr_dirs)} seqs = {total} jobs')
    print(f'  Workers: {args.workers}')

    done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(encode_and_decode, hr_dir, output_root, c, p)
            for hr_dir in hr_dirs
            for c, p in variants
        ]
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f'  [{done}/{total}] {rate:.1f} jobs/s, ETA {eta:.0f}s')

    print(f'  Done ({time.perf_counter() - t0:.0f}s)')
    print(f'\nDataset ready at: {output_root.resolve()}')
    print(f'  HR: {len(hr_dirs)} videos, {len(list((output_root/"HR").rglob("*.png")))} frames total')
    print(f'  LR variants: {len(variants)}')


if __name__ == '__main__':
    main()
