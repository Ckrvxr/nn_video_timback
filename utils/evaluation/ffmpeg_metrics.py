"""Compute PSNR, SSIM, VMAF via ffmpeg subprocess."""

import re
import subprocess


def ffmpeg_metrics(ref_path, dist_path, width=512, height=512,
                   pix_fmt='yuv444p12le', framerate=60, timeout=300):
    """Compare two videos using ffmpeg PSNR/SSIM/libvmaf.

    dist_path: path to the distorted video (or rawvideo file)
    ref_path: path to the reference video (usually HR.mkv)

    Returns dict with psnr, ssim, vmaf.
    """
    is_raw = str(dist_path).endswith('.yuv')

    cmd = ['ffmpeg', '-vsync', '0', '-hide_banner']
    if is_raw:
        cmd += ['-f', 'rawvideo', '-pix_fmt', pix_fmt,
                '-s', f'{width}x{height}', '-r', str(framerate),
                '-i', str(dist_path)]
    else:
        cmd += ['-i', str(dist_path)]
    cmd += ['-i', str(ref_path)]

    cmd += [
        '-filter_complex',
        '[0:v]split=3[dist1][dist2][dist3];'
        '[1:v]split=3[ref1][ref2][ref3];'
        '[dist1][ref1]psnr=stats_file=-;'
        '[dist2][ref2]ssim=stats_file=-;'
        '[dist3][ref3]libvmaf=model=version=vmaf_v0.6.1neg:log_fmt=json',
        '-f', 'null', '-',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    stderr = result.stderr

    if result.returncode != 0:
        # Log a concise chunk of stderr so the user sees what went wrong.
        tail = stderr[-800:] if stderr else '(no stderr)'
        raise RuntimeError(f'ffmpeg failed (code {result.returncode}): {tail}')

    psnr = _parse_psnr(stderr)
    ssim = _parse_ssim(stderr)
    vmaf = _parse_vmaf(stderr)

    return {'psnr': psnr, 'ssim': ssim, 'vmaf': vmaf}


def _parse_psnr(text: str) -> float:
    m = re.search(r'PSNR.*average:([\d.]+)', text)
    return float(m.group(1)) if m else 0.0


def _parse_ssim(text: str) -> float:
    m = re.search(r'SSIM.*All:([\d.]+)\s*\(', text)
    return float(m.group(1)) if m else 0.0


def _parse_vmaf(text: str) -> float:
    m = re.search(r'VMAF score:\s*([\d.]+)', text)
    return float(m.group(1)) if m else 0.0
