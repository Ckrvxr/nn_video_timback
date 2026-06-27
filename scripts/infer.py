import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import numpy as np
import torch

from utils.colorspace.color_space import yuv_to_rgb_linear_np


def load_video(path):
    path = Path(path)
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True, timeout=30)
    w, h = map(int, result.stdout.strip().split(','))
    proc = subprocess.run([
        'ffmpeg', '-vsync', '0', '-hide_banner', '-i', str(path),
        '-vf', "setparams=color_primaries=bt2020:color_trc=bt709, zscale=matrix=bt2020nc:transfer=bt709:primaries=bt709:range=full",
        '-f', 'rawvideo', '-pix_fmt', 'yuv444p12le',
        '-s', f'{w}x{h}', 'pipe:1',
    ], capture_output=True, timeout=120)
    raw = np.frombuffer(proc.stdout, dtype=np.uint16)
    nf = raw.size // (3 * h * w)
    planes = raw[:nf * 3 * h * w].reshape(nf, 3, h, w)
    return np.transpose(planes, (0, 2, 3, 1)), w, h


def main():
    lr_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else 'output.mkv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    print(f'Loading checkpoint: {ckpt_path}')
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    state = ckpt.get('model', ckpt.get('params', ckpt))
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    dim = state['head.weight'].shape[0]
    params_per_block = 7
    n1 = sum(1 for k in state if k.startswith('body.enc1.')) // params_per_block
    n2 = sum(1 for k in state if k.startswith('body.enc2.')) // params_per_block
    n3 = sum(1 for k in state if k.startswith('body.dec2.')) // params_per_block
    nmid = sum(1 for k in state if k.startswith('body.mid.')) // params_per_block
    from core import ArtRT
    model = ArtRT(dim=dim, n1=n1, n2=n2, n3=n3, nmid=nmid).to(device)
    model.load_state_dict(state, strict=True)
    model = model.to(device, dtype=torch.float16)
    model.eval()
    print(f'  params: {sum(p.numel() for p in model.parameters()):,}')

    print(f'Loading video: {lr_path}')
    frames, w, h = load_video(lr_path)
    print(f'  {frames.shape[0]} frames, {w}x{h}')

    print('Running inference + encoding...')

    fps = 30
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
        '-s', f'{w}x{h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-vf', "lutrgb=r=gammaval(0.45):g=gammaval(0.45):b=gammaval(0.45)",
        '-c:v', 'libx265', '-preset', 'ultrafast', '-x265-params', 'lossless=1',
        '-pix_fmt', 'yuv444p12le',
        out_path,
    ], stdin=subprocess.PIPE)

    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            rgb = yuv_to_rgb_linear_np(batch, bits=12)
            inp = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device, dtype=dtype)
            pred = model(inp)
            pred_np = pred.float().cpu().permute(0, 2, 3, 1).numpy()
            pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
            raw_out = (pred_np * 65535).clip(0, 65535).astype(np.uint16)
            ff.stdin.write(raw_out.tobytes())
            if (i // batch_size) % 20 == 0:
                print(f'  {i}/{len(frames)}')

    ff.stdin.close()
    ff.wait()
    print(f'  saved: {out_path}')


if __name__ == '__main__':
    main()
