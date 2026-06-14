import argparse
import sys
from pathlib import Path

import torch
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.av1_vsr import AV1VSR
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Test AV1-VSR')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--metrics', type=str, nargs='+', default=['psnr', 'ssim'])
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


@torch.no_grad()
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt.get("epoch", "?")}')

    test_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=1,
        scales=[1, 2, 3, 4],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=2,
        is_train=False,
        data_type='compressed',
    )

    results = {}
    for batch in test_loader:
        lr_frames = batch['lr_frames'].to(device)
        hr = batch['hr'].to(device)
        scale_val = batch['scale']

        pred = model(lr_frames[:, 0], lr_frames[:, 1], lr_frames[:, 2], scale=scale_val)

        key = f'scale_{scale_val}'
        if key not in results:
            results[key] = {'psnr': [], 'ssim': []}
        if 'psnr' in args.metrics:
            results[key]['psnr'].append(calculate_psnr(pred, hr))
        if 'ssim' in args.metrics:
            results[key]['ssim'].append(calculate_ssim(pred, hr))

    for scale_label, vals in sorted(results.items()):
        line = f'{scale_label}: '
        if 'psnr' in args.metrics and vals['psnr']:
            line += f'PSNR={np.mean(vals["psnr"]):.2f} '
        if 'ssim' in args.metrics and vals['ssim']:
            line += f'SSIM={np.mean(vals["ssim"]):.4f}'
        print(line)


if __name__ == '__main__':
    main()
