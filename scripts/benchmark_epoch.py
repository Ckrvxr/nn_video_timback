"""Time one train epoch and one validation pass."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from yaml import safe_load

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch


def main():
    config = safe_load(open('configs/prod.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) else None

    train_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['training']['batch_size'],
        scales=config['data'].get('scales', config['model']['scales']),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('workers', 4),
        is_train=True,
        data_type=config['data'].get('data_type', 'compressed'),
    )
    val_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['data'].get('val_batch_size', 16),
        scales=[4],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=0,
        is_train=False,
        data_type=config['data'].get('data_type', 'compressed'),
    )

    # Warmup one batch
    for batch in train_loader:
        break

    model.train()
    t0 = time.time()
    for batch_idx, batch in enumerate(train_loader):
        lr = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        scale = batch['scale']
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
                loss_dict = criterion(pred, hr)
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
            loss_dict = criterion(pred, hr)
            loss_dict['total'].backward()
            optimizer.step()
    train_time = time.time() - t0
    print(f'Train epoch ({len(train_loader)} batches): {train_time:.3f}s')

    model.eval()
    t0 = time.time()
    total_psnr = total_ssim = n = 0
    max_samples = 100
    with torch.no_grad():
        for batch in val_loader:
            lr = batch['lr_frames'].to(device)
            hr = batch['hr'].to(device)
            scale = batch['scale']
            pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
            b = min(pred.size(0), max_samples - n)
            if b < pred.size(0):
                pred = pred[:b]
                hr = hr[:b]
            total_psnr += calculate_psnr_batch(pred, hr).sum().item()
            total_ssim += calculate_ssim_batch(pred, hr).sum().item()
            n += b
            if n >= max_samples:
                break
    val_time = time.time() - t0
    print(f'Validation ({n} samples): {val_time:.3f}s')
    print(f'val psnr={total_psnr/max(1,n):.2f} ssim={total_ssim/max(1,n):.4f}')


if __name__ == '__main__':
    main()
