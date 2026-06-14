import argparse
import json
import random
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from yaml import safe_load

from models.av1_vsr import AV1VSR
from models import HyperFixer
from losses.composite import CompositeLoss
from utils.console import console
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch
from rich.progress import Progress, TimeElapsedColumn, TextColumn, BarColumn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only first N videos (0 = all)')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, run_dir: Path, output_dir: Path, is_best: bool = False):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(ckpt, str(run_dir / 'last.pth'))
    if is_best:
        shutil.copy2(str(run_dir / 'last.pth'), str(run_dir / 'best.pth'))
    shutil.copy2(str(run_dir / 'last.pth'), str(output_dir / 'last.pth'))
    if is_best:
        shutil.copy2(str(run_dir / 'best.pth'), str(output_dir / 'best.pth'))


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    has_temp = config['loss'].get('temporal', 0) > 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=n_batches)

        for batch_idx, batch in enumerate(loader):
            lr = batch['lr_frames'].to(device, non_blocking=True)
            hr = batch['hr'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                pred_cur = model(lr[:, 1], lr[:, 2], lr[:, 3])
                pred_next = model(lr[:, 2], lr[:, 3], lr[:, 4]) if has_temp else None
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
            progress.update(task, advance=1, description=f"loss: {batch_loss:.4f}")

    return total_loss / n_batches


def _yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) normalized YUV [-1,1] → (B,3,H,W) normalized RGB [-1,1]."""
    y = (x[:, 0:1] + 1) * 127.5
    u_centered = (x[:, 1:2] + 1) * 127.5 - 128.0
    v_centered = (x[:, 2:3] + 1) * 127.5 - 128.0
    r = y + 1.402 * v_centered
    g = y - 0.344 * u_centered - 0.714 * v_centered
    b = y + 1.772 * u_centered
    return torch.cat([r, g, b], dim=1) / 127.5 - 1.0


@torch.no_grad()
def validate(model, dataset, device, max_samples=100):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    half = dataset.frames // 2

    for v in dataset.videos:
        variant = v['variants'][0]
        dataset.load_video(v['name'], variant)
        hr_cache = dataset._hr_cache
        lr_cache = dataset._lr_cache
        n_frames = v['n_frames']

        for center in range(half, n_frames - half):
            if n >= max_samples:
                break

            center_hr = hr_cache[center]
            h, w = center_hr.shape[:2]
            crop_y, crop_x, crop_size = dataset._get_crop_params(h, w)

            lr_patches = []
            for i in range(center - half, center + half + 1):
                i_c = max(0, min(i, n_frames - 1))
                lr_crop = dataset._safe_crop(lr_cache[i_c], crop_y, crop_x, crop_size)
                lr_patches.append(lr_crop)

            hr_patch = dataset._safe_crop(center_hr, crop_y, crop_x, crop_size)

            lr_t = torch.from_numpy(np.stack(lr_patches, 0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
            hr_t = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

            lr_t = lr_t.unsqueeze(0).to(device)

            pred = model(
                lr_t[:, 1],
                lr_t[:, 2],
                lr_t[:, 3],
            )

            pred_rgb = _yuv_to_rgb(pred)
            hr_rgb = _yuv_to_rgb(hr_t[None].to(device))
            total_psnr += calculate_psnr_batch(pred_rgb, hr_rgb).sum().item()
            total_ssim += calculate_ssim_batch(pred_rgb, hr_rgb).sum().item()
            n += 1

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
    console.print(f'Using device: {device}')

    train_cfg = config['training']

    model_name = config['model'].get('name', 'av1_vsr')
    console.print(f'Model: {model_name}')

    if model_name == 'hyper_fixer':
        model = HyperFixer(
            n_features=config['model']['n_features'],
            n_blocks=config['model']['n_blocks'],
            latent=config['model'].get('latent', 512),
        ).to(device)
    else:
        model = AV1VSR(
            in_channels=3,
            n_features=config['model']['n_features'],
            n_blocks=config['model']['n_blocks'],
            scales=config['model'].get('scales', [1, 2, 4]),
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
        console.print(f'Resumed from epoch {start_epoch}')

    data_type = config['data'].get('data_type', 'compressed')

    train_loader = create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=train_cfg['batch_size'],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('workers', 4),
        is_train=True,
        data_type=data_type,
    )

    limit = args.limit or config['data'].get('limit', 0)
    if limit > 0:
        train_loader.dataset.videos = train_loader.dataset.videos[:limit]
        console.print(f'[yellow]Limited to {len(train_loader.dataset.videos)} videos[/]')

    val_dataset = create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=config['data'].get('val_batch_size', 16),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=0,
        is_train=False,
        data_type=data_type,
    )

    console.print(f'Training: {data_type} data, {n_epochs} epochs')
    console.print(f'LR: {train_cfg["lr"]}, batch_size: {train_cfg["batch_size"]}')
    console.print(f'Train samples: {len(train_loader.dataset)}')

    best_psnr = -float('inf')
    prev_loss = None
    prev_psnr = None
    prev_ssim = None
    up_streak = 0
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob('run_*'))
    next_id = max(int(d.name.split('_')[1]) for d in existing) + 1 if existing else 1
    run_dir = output_dir / f'run_{next_id:03d}'
    run_dir.mkdir()
    console.print(f'Run: {run_dir}')

    metrics_path = run_dir / 'metrics.json'
    metrics = []
    save_interval = config['logging'].get('save_interval', 5)

    for epoch in range(start_epoch, n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler)
        scheduler.step()

        val_psnr = None
        val_ssim = None
        is_new_best = False

        if epoch % config['logging'].get('val_interval', 1) == 0:
            torch.cuda.empty_cache()
            val_psnr, val_ssim = validate(model, val_dataset, device)

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
            console.print(f'Epoch {epoch:3d}:  {l_emoji} loss={train_loss:.4f}  │  {p_emoji} psnr={val_psnr:5.2f}  │  {s_emoji} ssim={val_ssim:.4f}    {flag_str}')

            prev_loss = train_loss
            prev_psnr = val_psnr
            prev_ssim = val_ssim

        save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir, is_best=is_new_best if val_psnr is not None else False)

        if save_interval > 0 and epoch % save_interval == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, str(run_dir / f'epoch_{epoch:03d}.pth'))

        metrics.append({"epoch": epoch, "loss": round(train_loss, 4),
                        "psnr": round(val_psnr, 4) if val_psnr is not None else None,
                        "ssim": round(val_ssim, 4) if val_ssim is not None else None})
        metrics_path.write_text(json.dumps(metrics, indent=2))

    console.print('Training complete.')


if __name__ == '__main__':
    main()
