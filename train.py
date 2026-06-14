import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from yaml import safe_load

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, path: str):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, path)


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    log_interval = config['logging'].get('log_interval', 50)

    has_temp = config['loss'].get('temporal', 0) > 0

    for batch_idx, batch in enumerate(loader):
        lr_frames = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)
        scale = batch['scale']

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            pred_cur = model(lr_frames[:, 1], lr_frames[:, 2], lr_frames[:, 3], scale=scale)
            pred_next = model(lr_frames[:, 2], lr_frames[:, 3], lr_frames[:, 4], scale=scale) if has_temp else None
            loss_dict = criterion(pred_cur, hr, pred_next=pred_next)

        if scaler is not None:
            scaler.scale(loss_dict['total']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad'])
            optimizer.step()

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss

        if batch_idx % log_interval == 0:
            print(f'  [{batch_idx}/{n_batches}] loss: {batch_loss:.4f} | '
                  f'char: {loss_dict.get("char", 0):.4f} | ssim: {loss_dict.get("ssim", 0):.4f}')

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
        scale = batch['scale']

        pred = model(lr_frames[:, 1], lr_frames[:, 2], lr_frames[:, 3], scale=scale)

        # Truncate the last batch if we have already collected enough samples.
        b = min(pred.size(0), max_samples - n)
        if b < pred.size(0):
            pred = pred[:b]
            hr = hr[:b]

        total_psnr += calculate_psnr_batch(pred, hr).sum().item()
        total_ssim += calculate_ssim_batch(pred, hr).sum().item()
        n += b

        if n >= max_samples:
            break

    return total_psnr / max(1, n), total_ssim / max(1, n)


def main():
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('seed')
    if seed:
        set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_cfg = config['training']

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)

    criterion = CompositeLoss(config['loss'], device=device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
    )

    n_epochs = train_cfg['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=train_cfg['lr_min'])

    scaler = torch.amp.GradScaler('cuda') if train_cfg.get('amp', False) and device.type == 'cuda' else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    data_type = config['data'].get('data_type', 'compressed')
    train_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=train_cfg['batch_size'],
        scales=config['data'].get('scales', config['model']['scales']),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('workers', 4),
        is_train=True,
        data_type=data_type,
    )

    val_loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['data'].get('val_batch_size', 16),
        scales=[4],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=2,
        is_train=False,
        data_type=data_type,
    )

    print(f'Training: {data_type} data, {n_epochs} epochs')
    print(f'LR: {train_cfg["lr"]}, batch_size: {train_cfg["batch_size"]}')
    print(f'Train samples: {len(train_loader.dataset)}')

    best_psnr = -float('inf')
    prev_loss = None
    prev_psnr = None
    prev_ssim = None
    up_streak = 0
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler)
        scheduler.step()

        if epoch % config['logging'].get('val_interval', 1) == 0:
            torch.cuda.empty_cache()
            val_psnr, val_ssim = validate(model, val_loader, device)

            def arrow(curr, prev):
                if prev is None:
                    return '➡️'
                return '⬆️' if curr > prev else '⬇️'

            l_emoji = arrow(train_loss, prev_loss)
            p_emoji = arrow(val_psnr, prev_psnr)
            s_emoji = arrow(val_ssim, prev_ssim)

            flags = []
            if epoch == 0:
                flags.append('🌱')
            is_new_best = val_psnr > best_psnr
            if is_new_best:
                best_psnr = val_psnr
                flags.append('🎯')
                save_checkpoint(model, optimizer, scheduler, epoch, str(output_dir / 'best.pth'))
            if prev_psnr is not None:
                diff = val_psnr - prev_psnr
                if diff < -2.0:
                    flags.append('💀')
                elif diff < -0.5:
                    flags.append('⚠️')
            if val_psnr >= 20:
                flags.append('🏆')
            elif val_psnr >= 15 and is_new_best:
                flags.append('🔥')
            if val_psnr > (prev_psnr if prev_psnr is not None else -1):
                up_streak += 1
            else:
                up_streak = 0
            if up_streak >= 3:
                flags.append('🚀')
            if (prev_psnr is not None and prev_loss is not None
                    and train_loss > prev_loss and val_psnr < prev_psnr):
                flags.append('🤖')

            flag_str = '  ' + ' '.join(flags) if flags else ''
            print(f'Epoch {epoch:3d}:  {l_emoji} loss={train_loss:.4f}  │  {p_emoji} psnr={val_psnr:5.2f}  │  {s_emoji} ssim={val_ssim:.4f}    {flag_str}')

            prev_loss = train_loss
            prev_psnr = val_psnr
            prev_ssim = val_ssim

        save_checkpoint(model, optimizer, scheduler, epoch, str(output_dir / 'last.pth'))

    print('Training complete.')


if __name__ == '__main__':
    main()
