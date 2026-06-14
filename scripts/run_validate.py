import argparse
import shutil
import subprocess
import math
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lpips import LPIPS

from models.av1_vsr import AV1VSR
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--crf', type=int, default=40)
    parser.add_argument('--preset', type=int, default=10)
    parser.add_argument('--output', type=str, default='output/validate/validate_output.mp4')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--tile-overlap', type=int, default=32)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--skip-vmaf', action='store_true', help='Skip VMAF computation')
    return parser.parse_args()


def encode_av1(frames_dir: Path, output_path: Path, crf: int, preset: int, fps: int, scale: int):
    pattern = str(frames_dir / 'frame_%06d.png')
    vf = f'scale=iw/{scale}:ih/{scale}:flags=bicubic' if scale > 1 else ''
    cmd = [
        'ffmpeg', '-framerate', str(fps), '-i', pattern,
        '-c:v', 'libsvtav1', '-crf', str(crf),
        '-svtav1-params', f'tune=0:preset={preset}',
        '-pix_fmt', 'yuv420p', '-y', str(output_path),
    ]
    if vf:
        cmd.insert(-1, '-vf')
        cmd.insert(-1, vf)
    subprocess.run(cmd, check=True, capture_output=True)


@torch.inference_mode()
def forward_tiled(model, prev, cur, next_, scale, tile_size, overlap):
    B, C, H, W = cur.shape
    if tile_size >= H and tile_size >= W:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model(prev, cur, next_, scale=scale)

    stride = tile_size - overlap
    h_tiles = max(1, math.ceil((H - overlap) / stride))
    w_tiles = max(1, math.ceil((W - overlap) / stride))

    h_stride = (H - tile_size) / max(1, h_tiles - 1) if h_tiles > 1 else 0
    w_stride = (W - tile_size) / max(1, w_tiles - 1) if w_tiles > 1 else 0

    out_h, out_w = H * scale, W * scale
    out_tile = tile_size * scale
    out_overlap = overlap * scale

    output = torch.zeros(B, C, out_h, out_w, device=cur.device, dtype=torch.float32)
    weight = torch.zeros(B, 1, out_h, out_w, device=cur.device, dtype=torch.float32)

    half_ol = out_overlap // 2
    w1d = torch.ones(out_tile, device=cur.device, dtype=torch.float32)
    if half_ol > 0:
        ramp = torch.linspace(1 / (half_ol + 1), 1, half_ol, device=cur.device, dtype=torch.float32)
        w1d[:half_ol] = ramp
        w1d[out_tile - half_ol:] = ramp.flip(0)
    tw = w1d.view(1, 1, out_tile, 1) * w1d.view(1, 1, 1, out_tile)

    for hi in range(h_tiles):
        hs = int(round(hi * h_stride)) if h_tiles > 1 else 0
        he = hs + tile_size
        if he > H:
            hs = H - tile_size
            he = H

        for wi in range(w_tiles):
            ws = int(round(wi * w_stride)) if w_tiles > 1 else 0
            we = ws + tile_size
            if we > W:
                ws = W - tile_size
                we = W

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tile_out = model(
                    prev[:, :, hs:he, ws:we],
                    cur[:, :, hs:he, ws:we],
                    next_[:, :, hs:he, ws:we],
                    scale=scale,
                )

            ohs, ohe = hs * scale, he * scale
            ows, owe = ws * scale, we * scale
            output[:, :, ohs:ohe, ows:owe] += tile_out.float() * tw
            weight[:, :, ohs:ohe, ows:owe] += tw

    return output / weight


@torch.inference_mode()
def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    lpips_fn = LPIPS(net='alex').to(device).eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(total, args.max_frames) if args.max_frames else total
    cap.release()

    video_name = Path(args.video).stem
    cache_root = Path('output/validate/cache') / f'{video_name}_{orig_w}x{orig_h}'
    variant_dir = cache_root / f'svt_crf{args.crf}_p{args.preset}'
    hr_dir = cache_root / 'hr'
    lr_dir = variant_dir / 'lr_frames'
    lr_video = variant_dir / 'lr.mp4'

    print(f'Original: {orig_w}x{orig_h} @ {fps:.0f}fps, {n_frames} frames')
    print(f'AV1 encode: CRF {args.crf}, preset {args.preset}, scale {args.scale}')
    print(f'Cache: {cache_root}')

    print('Step 1: Extracting HR frames...')
    hr_files = sorted(hr_dir.glob('*.png'))
    if len(hr_files) >= n_frames:
        print(f'  cached: {len(hr_files)} frames')
    else:
        shutil.rmtree(hr_dir, ignore_errors=True)
        hr_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'ffmpeg', '-i', args.video, '-pix_fmt', 'rgb24',
            '-q:v', '2', '-frames:v', str(n_frames),
            '-y', str(hr_dir / 'frame_%06d.png'),
        ], check=True, capture_output=True)
        hr_files = sorted(hr_dir.glob('*.png'))

    print('Step 2: Bicubic down + AV1 encoding...')
    if lr_video.exists():
        print(f'  cached: {lr_video.name}')
        encode_time = 0
    else:
        variant_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        encode_av1(hr_dir, lr_video, args.crf, args.preset, int(fps), args.scale)
        encode_time = time.perf_counter() - t0
        print(f'  encode: {encode_time:.1f}s')

    print('Step 3: Decoding LR frames...')
    lr_files = sorted(lr_dir.glob('*.png'))
    if len(lr_files) >= n_frames:
        print(f'  cached: {len(lr_files)} frames')
    else:
        shutil.rmtree(lr_dir, ignore_errors=True)
        lr_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'ffmpeg', '-i', str(lr_video), '-pix_fmt', 'rgb24',
            '-q:v', '2', '-y', str(lr_dir / 'frame_%06d.png'),
        ], check=True, capture_output=True)
        lr_files = sorted(lr_dir.glob('*.png'))
        print(f'  decoded: {len(lr_files)} frames')

    out_w, out_h = orig_w * args.scale, orig_h * args.scale
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, int(fps), (out_w, out_h))

    frames_lr = []
    frames_hr = []
    times = []
    psnrs, ssims, lpipss = [], [], []
    warmup = args.warmup

    print('Step 4: Model inference...')
    for i in range(min(len(lr_files), len(hr_files))):
        lr = cv2.cvtColor(cv2.imread(str(lr_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        hr_src = cv2.cvtColor(cv2.imread(str(hr_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        lr_t = torch.from_numpy(lr).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        frames_lr.append(lr_t)
        frames_hr.append(torch.from_numpy(hr_src).float().permute(2, 0, 1) / 127.5 - 1.0)

        if len(frames_lr) < 3:
            continue

        stack = torch.stack([frames_lr[-3], frames_lr[-2], frames_lr[-1]], dim=1).squeeze(0).to(device)

        if warmup > 0:
            _ = forward_tiled(model, stack[0:1], stack[1:2], stack[2:3], scale=args.scale, tile_size=args.tile_size, overlap=args.tile_overlap)
            torch.cuda.synchronize()
            warmup -= 1
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = forward_tiled(model, stack[0:1], stack[1:2], stack[2:3], scale=args.scale, tile_size=args.tile_size, overlap=args.tile_overlap)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

        # ── Metrics on GPU ──
        hr_ref = frames_hr[i].unsqueeze(0).to(device)
        if pred.shape[-2:] != hr_ref.shape[-2:]:
            hr_ref = torch.nn.functional.interpolate(hr_ref, size=pred.shape[-2:], mode='bicubic')

        psnr_val = calculate_psnr_batch(pred, hr_ref).item()
        ssim_val = calculate_ssim_batch(pred, hr_ref).item()
        lpips_val = lpips_fn(pred, hr_ref).item()

        psnrs.append(psnr_val)
        ssims.append(ssim_val)
        lpipss.append(lpips_val)

        # ── Save frame ──
        pred = pred.squeeze(0).cpu()
        pred_np = pred.permute(1, 2, 0).numpy()
        pred_np = np.clip((pred_np + 1) * 127.5, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))

    writer.release()

    # ── Step 5: VMAF (if available) ──
    vmaf_score = None
    if not args.skip_vmaf:
        try:
            r = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
            has_libvmaf = 'libvmaf' in r.stdout
        except FileNotFoundError:
            has_libvmaf = False

        if has_libvmaf:
            print('Step 5: Computing VMAF...')
            hr_pattern = str(hr_dir / 'frame_%06d.png')
            vmaf_log = Path(args.output).with_suffix('.vmaf.json')
            vmaf_cmd = [
                'ffmpeg', '-i', str(args.output),
                '-r', str(int(fps)),
                '-start_number', '2',  # skip first frame (warmup fallout)
                '-i', hr_pattern,
                '-filter_complex',
                f'[0:v][1:v]libvmaf=model=version=vmaf_v0.6.1:log_path={vmaf_log}:log_fmt=json:n_threads=4',
                '-f', 'null', '-',
            ]
            try:
                subprocess.run(vmaf_cmd, check=True, capture_output=True)
                import json
                with open(vmaf_log) as f:
                    data = json.load(f)
                scores = [f['metrics']['vmaf'] for f in data['frames']]
                vmaf_score = np.mean(scores)
            except Exception:
                pass

    print()
    print('═══ Validation Report ═══')
    print(f'  Resolution:   {orig_w}x{orig_h} (x{args.scale} via AV1 CRF{args.crf} p{args.preset})')
    print(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'  ---- Timing ----')
    print(f'  Avg latency:  {np.mean(times):.0f} ms/frame')
    print(f'  FPS:          {1000/np.mean(times):.1f}')
    print(f'  ---- Quality ----')
    print(f'  PSNR:         {np.mean(psnrs):.3f}')
    print(f'  SSIM:         {np.mean(ssims):.4f}')
    print(f'  LPIPS:        {np.mean(lpipss):.4f}')
    if vmaf_score is not None:
        print(f'  VMAF:         {vmaf_score:.2f}')
    else:
        print(f'  VMAF:         not available (ffmpeg lacks libvmaf)')
    print(f'  Frames:       {len(times)}')

    print(f'\nOutput video: {args.output}')


if __name__ == '__main__':
    main()
