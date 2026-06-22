import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ffprobe(args: list[str], video_path: str) -> str:
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-of', 'csv=p=0'] + args + [video_path]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


def probe_metadata(video_path: str) -> tuple[float, int, int]:
    dims = _ffprobe(['-show_entries', 'stream=width,height'], video_path)
    parts = dims.split(',')
    w = int(parts[0]) if parts and parts[0].isdigit() else 1920
    h = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1080

    dur_str = _ffprobe(['-show_entries', 'format=duration'], video_path)
    dur = 0.0
    if dur_str and dur_str != 'N/A':
        try:
            dur = float(dur_str)
        except ValueError:
            pass

    if dur <= 0:
        dur_str2 = _ffprobe(['-show_entries', 'stream=duration'], video_path)
        if dur_str2 and dur_str2 != 'N/A':
            try:
                dur = float(dur_str2)
            except ValueError:
                pass

    if dur <= 0:
        dur = 30.0

    return dur, w, h


def extract_y_frame(video_path: str, seek_sec: float) -> np.ndarray:
    import av
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'

        seek_pts = int(seek_sec / float(stream.time_base))
        container.seek(seek_pts, stream=stream)

        for frame in container.decode(video=0):
            try:
                bits = frame.format.components[0].bits
            except Exception:
                bits = 8
            y_data = frame.planes[0]
            buf = np.frombuffer(y_data, dtype=np.uint16 if bits > 8 else np.uint8)
            h = y_data.height
            stride = y_data.line_size
            src_w = stride // (2 if bits > 8 else 1)
            y = buf.reshape(h, src_w)[:, :frame.width].astype(np.float32)
            peak = float((1 << bits) - 1)
            y = y / peak * 2.0 - 1.0
            return y
    raise RuntimeError('No frame decoded')


def downsample_lanczos(y: np.ndarray, factor: int) -> np.ndarray:
    h, w = y.shape
    out_h = h // factor
    out_w = w // factor
    return cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def compute_fft_2d(y: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(y)
    return np.fft.fftshift(fft)


def magnitude_spectrum(fft: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.log10(np.abs(fft) + eps)


def save_side_by_side(mags: dict[str, np.ndarray], output_path: str,
                      cmap: str = 'inferno', scale: int = 1):
    n = len(mags)
    labels = list(mags.keys())
    dpi = 100
    max_h = max(m.shape[0] for m in mags.values())
    total_w = sum(m.shape[1] for m in mags.values())
    gap = 20
    fig_w = (total_w + gap * (n - 1)) / dpi * scale
    fig_h = (max_h + 80) / dpi * scale

    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h),
                             gridspec_kw={'width_ratios': [m.shape[1] for m in mags.values()]})
    if n == 1:
        axes = [axes]
    for ax, label, mag in zip(axes, labels, mags.values()):
        im = ax.imshow(mag, cmap=cmap, aspect='auto', interpolation='none')
        ax.set_title(label, fontsize=8 * scale)
        ax.axis('off')
    fig.subplots_adjust(wspace=0.02, left=0, right=1, top=0.95, bottom=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Extract one frame → generate high-quality Lanczos downscales 2x/4x → 2D FFT')
    parser.add_argument('input', type=str, help='Input video path')
    parser.add_argument('-f', '--frame', type=float, default=None,
                        help='Time in seconds (default: 1%% into video)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output image path (default: {input_stem}_fft.png)')
    parser.add_argument('--cmap', type=str, default='inferno',
                        help='Matplotlib colormap (default: inferno)')
    parser.add_argument('--scale', type=int, default=1,
                        help='Output resolution multiplier (default: 1)')
    parser.add_argument('--save-npy', action='store_true',
                        help='Also save raw complex spectrum as .npy')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'File not found: {args.input}')
        sys.exit(1)

    print(f'Probing {input_path.name}...')
    dur, _, _ = probe_metadata(str(input_path))
    print(f'  Duration: {dur:.2f}s')

    seek = dur * 0.01 if args.frame is None else args.frame
    print(f'Extracting frame at {seek:.3f}s...')

    y = extract_y_frame(str(input_path), seek)
    h, w = y.shape
    print(f'  Y shape: {h}x{w}')

    versions = {'1x (original)': y}
    for factor in (2, 4):
        if h // factor > 0 and w // factor > 0:
            versions[f'{factor}x down'] = downsample_lanczos(y, factor)

    mags = {}
    for label, arr in versions.items():
        fft = compute_fft_2d(arr)
        mag = magnitude_spectrum(fft)
        mags[label] = mag
        print(f'  {label}: FFT {fft.shape}')

    output = args.output or str(input_path.with_name(f'{input_path.stem}_fft.png'))
    save_side_by_side(mags, output, cmap=args.cmap, scale=args.scale)
    print(f'  Visualization saved: {output}')

    if args.save_npy:
        for label, arr in versions.items():
            fft = compute_fft_2d(arr)
            stem = f'{input_path.stem}_fft_{label.split()[0]}'
            np.save(str(input_path.with_name(f'{stem}.npy')), fft)
        print(f'  Raw spectra saved as .npy')


if __name__ == '__main__':
    main()
