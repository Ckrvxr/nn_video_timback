import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr, calculate_ssim


def parse_args():
    parser = argparse.ArgumentParser(description='Train AV1-VSR')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, phase, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for batch in loader:
        lr_frames = batch['lr_frames'].to(device)
        hr = batch['hr'].to(device)
        scales = batch['scale'].tolist()

        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(lr_frames.size(0)):
            scale_val = int(scales[i]) if isinstance(scales[i], (int, float)) else int(scales[i][0])
            f_prev = lr_frames[i, 0:1]
            f_cur = lr_frames[i, 1:2]
            f_next = lr_frames[i, 2:3]
            hr_i = hr[i:i+1]

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    pred = model(f_prev, f_cur, f_next, scale=scale_val)
                    loss_dict = criterion(pred, hr_i)
                scaler.scale(loss_dict['total']).backward()
            else:
                pred = model(f_prev, f_cur, f_next, scale=scale_val)
                loss_dict = criterion(pred, hr_i)
                loss_dict['total'].backward()

            batch_loss += loss_dict['total'].item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
            optimizer.step()

        total_loss += batch_loss / lr_frames.size(0)

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device, max_samples=100):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for batch in loader:
        lr_frames = batch['lr_frames'].to(device)
        hr = batch['hr'].to(device)
        scales = batch['scale']

        for i in range(min(lr_frames.size(0), max_samples - n)):
            scale_val = int(scales[i].item())
            f_prev = lr_frames[i, 0:1]
            f_cur = lr_frames[i, 1:2]
            f_next = lr_frames[i, 2:3]
            hr_i = hr[i:i+1]

            pred = model(f_prev, f_cur, f_next, scale=scale_val)
            total_psnr += calculate_psnr(pred, hr_i)
            total_ssim += calculate_ssim(pred, hr_i)
            n += 1
            if n >= max_samples:
                break
        if n >= max_samples:
            break

    return total_psnr / max(1, n), total_ssim / max(1, n)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_cfg = config['training']
    phase_key = f'phase{args.phase}'
    phase_cfg = train_cfg['phases'][phase_key]

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_rcab=config['model']['n_rcab'],
        scales=config['model']['scales'],
    ).to(device)

    criterion = CompositeLoss(config['loss'])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=phase_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=phase_cfg['epochs'],
        eta_min=train_cfg['lr_min'],
    )

    scaler = torch.amp.GradScaler('cuda') if train_cfg['amp'] and device.type == 'cuda' else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') and ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    data_type = 'clean' if phase_cfg['data'] == 'clean' else 'compressed'
    train_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=train_cfg['batch_size'],
        scales=config['data']['scales'],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data']['workers'],
        is_train=True,
        data_type=data_type,
    )

    val_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=1,
        scales=[4],
        patch_size=None,
        frames=config['data']['frames'],
        workers=2,
        is_train=False,
        data_type=data_type,
    )

    print(f'Phase {args.phase}: {phase_cfg["desc"]}')
    print(f'Epochs: {phase_cfg["epochs"]}, LR: {phase_cfg["lr"]}')
    print(f'Train samples: {len(train_loader.dataset)}')

    best_psnr = 0.0
    output_dir = Path(config['output_dir']) / phase_key
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, phase_cfg['epochs']):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler,
        )

        if scheduler is not None:
            scheduler.step()

        if epoch % config['logging']['val_interval'] == 0:
            val_psnr, val_ssim = validate(model, val_loader, device)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    model, optimizer, scheduler, epoch, args.phase,
                    str(output_dir / 'best.pth'),
                )
                print(f'  New best PSNR: {val_psnr:.2f}')

            print(
                f'Epoch {epoch:3d}/{phase_cfg["epochs"]} | '
                f'Loss: {train_loss:.4f} | '
                f'PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}'
            )

        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, args.phase,
                str(output_dir / f'epoch_{epoch:04d}.pth'),
            )

    print(f'Phase {args.phase} complete! Best PSNR: {best_psnr:.2f}')


if __name__ == '__main__':
    main()
