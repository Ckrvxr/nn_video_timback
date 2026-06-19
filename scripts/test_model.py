import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from yaml import safe_load

from models import MambaFixer


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = safe_load(open('configs/mamba.yaml'))
    model = MambaFixer(
        num_features=config['model_architecture'].get('num_features', 16),
        state_dimension=config['model_architecture'].get('state_dimension', 16),
    ).to(device)
    model.eval()

    ckpt = torch.load('runs/mamba/run_004/last.pth', map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    ckpt.pop('_t_state', None)
    model.load_state_dict(ckpt, strict=False)

    # Load a training sample
    lr = np.load('data/hoppers/69m17s_69m18s_h264_crf38_pslow_gop96_yuv420p10le/lr.npy')
    hr = np.load('data/hoppers/69m17s_69m18s_h264_crf38_pslow_gop96_yuv420p10le/hr.npy')
    print(f'lr: {lr.shape} {lr.dtype}  range=[{lr.min():.4f}, {lr.max():.4f}]')
    print(f'hr: {hr.shape} {hr.dtype}  range=[{hr.min():.4f}, {hr.max():.4f}]')

    frames = config['dataset']['num_frames']
    half = frames // 2
    center = 15
    lo, hi = center - half, center + half + 1

    lr_win = torch.from_numpy(lr[lo:hi]).float().unsqueeze(0).to(device)
    hr_t = torch.from_numpy(hr[center]).float().unsqueeze(0).to(device)
    center_idx = frames // 2

    model.reset_state(1, device)
    pred = model(lr_win[:, center_idx].to(memory_format=torch.channels_last))

    diff = pred - hr_t
    print(f'\npred: [{pred.min():.4f}, {pred.max():.4f}]')
    print(f'target: [{hr_t.min():.4f}, {hr_t.max():.4f}]')
    print(f'diff mean={diff.mean().item():.6f}  max_abs={diff.abs().max().item():.6f}  rmse={diff.pow(2).mean().sqrt().item():.6f}')

    for ch, name in enumerate(['I', 'Ct', 'Cp']):
        ch_d = diff[0, ch]
        print(f'  {name}: mean={ch_d.mean().item():.6f}  max_abs={ch_d.abs().max().item():.6f}')
        print(f'    pred range=[{pred[0,ch].min():.4f}, {pred[0,ch].max():.4f}]  hr=[{hr_t[0,ch].min():.4f}, {hr_t[0,ch].max():.4f}]')

    if diff.abs().max().item() < 0.05:
        print('\n\u2713 Model predictions match training targets')
    elif diff.abs().max().item() < 0.2:
        print('\n~ Reasonable but some error')
    else:
        print(f'\n\u2717 Large errors (max_diff={diff.abs().max().item():.4f})')


if __name__ == '__main__':
    main()
