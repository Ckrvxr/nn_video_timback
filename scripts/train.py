import argparse
import json
import queue
import random
import shutil
import sys
import threading
import time
import warnings
from datetime import datetime
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
from tqdm import tqdm

from models.av1_vsr import AV1VSR
from models import HyperFixer
from losses.composite import CompositeLoss
from utils.console import console
from utils.dataset import create_dataloader
from utils.frame_cache import FrameCache
from utils.blosc_cache import BloscCache
from utils.video_loader import load_video_frames_raw
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch


# Global background writer for checkpoints / metrics so that disk IO does not
# block the training loop.
_IO_QUEUE = queue.Queue()
_IO_THREAD = None


def _io_worker():
    while True:
        item = _IO_QUEUE.get()
        if item is None:
            break
        fn, args, kwargs = item
        try:
            fn(*args, **kwargs)
        except Exception as e:
            console.error(f'Background IO failed: {e}')


def start_io_worker():
    global _IO_THREAD
    if _IO_THREAD is None:
        _IO_THREAD = threading.Thread(target=_io_worker, daemon=True)
        _IO_THREAD.start()


def stop_io_worker():
    global _IO_THREAD
    if _IO_THREAD is not None:
        _IO_QUEUE.put(None)
        _IO_THREAD.join(timeout=30)
        _IO_THREAD = None


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
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                                 for k, v in optimizer.state_dict().items()},
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    run_last = run_dir / 'last.pth'
    run_best = run_dir / 'best.pth'
    out_last = output_dir / 'last.pth'
    out_best = output_dir / 'best.pth'

    def _write():
        torch.save(ckpt, str(run_last))
        if is_best:
            shutil.copy2(str(run_last), str(run_best))
        shutil.copy2(str(run_last), str(out_last))
        if is_best:
            shutil.copy2(str(run_best), str(out_best))

    _IO_QUEUE.put((_write, (), {}))


def _cache_one_video(args: tuple) -> str:
    """Decode one video and write to BloscCache. Returns status string."""
    video_path, cache_root = args
    cache = BloscCache(cache_root)
    cp = cache.cache_path(video_path)
    if cp.exists():
        return f'skip  {Path(video_path).name}'
    raw_frames = load_video_frames_raw(video_path)
    cache.put_raw(video_path, raw_frames)
    return f'cached {Path(video_path).name} ({len(raw_frames)} frames)'


def warm_blosc_cache(config):
    """Pre-decode videos to BloscCache before training starts.

    Respects ``data.limit`` — only warms the first N videos.
    After warming, workers read BloscCache only (fast, ~50ms per video).

    Set ``data.warm_cache: false`` to skip entirely.
    """
    if not config['data'].get('warm_cache', True):
        console.info('Skipping cache warming (warm_cache: false)')
        return

    datasets = config['data']['datasets']
    data_type = config['data'].get('data_type', 'compressed')
    limit = config['data'].get('limit', 0)

    todos = []
    for ds_root_str in datasets:
        ds_root = Path(ds_root_str)
        cache_root = ds_root / 'cache'
        hr_dir = ds_root / 'HR'

        # Build sorted list of video names (same order as dataset inventory).
        video_names = sorted(f.stem for f in hr_dir.glob('*.mp4'))
        if limit > 0:
            video_names = video_names[:limit]

        variant_dirs = sorted(
            d for d in ds_root.iterdir()
            if d.is_dir() and d.name not in ('HR', 'cache')
        ) if data_type == 'compressed' else [ds_root / 'HR']

        # Only warm HR + each variant for the limited set of names.
        targets = [hr_dir] + variant_dirs
        for vd in targets:
            for name in video_names:
                mp4 = vd / f'{name}.mp4'
                if mp4.exists():
                    blosc = BloscCache(cache_root)
                    if not blosc.cache_path(mp4).exists():
                        todos.append((str(mp4), str(cache_root)))

    if not todos:
        console.success('BloscCache already warm — no videos need decoding')
        return

    console.info(f'Warming BloscCache: {len(todos)} files to decode...')
    t0 = time.perf_counter()
    done = 0
    for todo in todos:
        done += 1
        status = _cache_one_video(todo)
        elapsed = time.perf_counter() - t0
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (len(todos) - done) / rate if rate > 0 else 0
        console.info(f'  [{done}/{len(todos)}] {status}  '
                      f'{elapsed:.0f}s elapsed  {remaining:.0f}s remaining')

    elapsed = time.perf_counter() - t0
    console.success(f'Cache warming complete ({elapsed:.1f}s)')


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None, batch_times=None):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    has_temp = config['loss'].get('temporal', 0) > 0
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    clip_grad = config['training']['clip_grad']
    bs = config['training']['batch_size']

    pbar = tqdm(total=n_batches, desc="Training", unit="batch", leave=False,
                miniters=10)
    t_batch_start = time.perf_counter()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        data_end = time.perf_counter()
        lr = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            pred_cur = model(lr[:, 1], lr[:, 2], lr[:, 3])
            pred_next = model(lr[:, 2], lr[:, 3], lr[:, 4]) if has_temp else None
            loss_dict = criterion(pred_cur, hr, pred_next=pred_next)
            loss = loss_dict['total'] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_accum = ((batch_idx + 1) % grad_accum == 0) or (batch_idx == n_batches - 1)
        if is_last_accum:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad()

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0
        pbar.set_postfix(loss=f'{batch_loss:.4f}', samples=f'{samples_sec:.0f}/s')
        pbar.update(1)

    pbar.close()
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
def validate(model, val_loader, device, max_samples=100):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for batch in val_loader:
        if n >= max_samples:
            break

        lr = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)

        pred = model(lr[:, 1], lr[:, 2], lr[:, 3])

        pred_rgb = _yuv_to_rgb(pred)
        hr_rgb = _yuv_to_rgb(hr)

        batch_size = pred.size(0)
        take = min(batch_size, max_samples - n)
        total_psnr += calculate_psnr_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        total_ssim += calculate_ssim_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        n += take

    return total_psnr / max(1, n), total_ssim / max(1, n)


def main():
    start_io_worker()
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('seed')
    if seed:
        set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    console.info(f'Using device: {device}')

    # ── GPU speed tuning ──
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        console.info('cudnn.benchmark + TF32 enabled')

    train_cfg = config['training']

    model_name = config['model'].get('name', 'av1_vsr')
    console.info(f'Model: {model_name}')

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

    # torch.compile speeds up forward/backward significantly on GPU.
    if train_cfg.get('compile', False):
        console.info('Using torch.compile')
        model = torch.compile(model, mode='max-autotune')

    criterion = CompositeLoss(config['loss'], device=device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
        betas=(train_cfg['beta1'], train_cfg['beta2']),
        fused=device.type == 'cuda',
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
        console.info(f'Resumed from epoch {start_epoch}')

    data_type = config['data'].get('data_type', 'compressed')

    # ── Pre-decode all videos to BloscCache so workers never hit PyAV ──
    warm_blosc_cache(config)

    train_loader = create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=train_cfg['batch_size'],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('workers', 4),
        is_train=True,
        data_type=data_type,
        cache_max_videos=config['data'].get('cache_max_videos'),
        prefetch_factor=config['data'].get('prefetch_factor'),
        shuffle_variants=config['data'].get('shuffle_variants', False),
        clip_repeat=config['data'].get('clip_repeat', 1),
    )

    limit = args.limit or config['data'].get('limit', 0)
    if limit > 0:
        train_loader.dataset.videos = train_loader.dataset.videos[:limit]
        console.info(f'Limited to {len(train_loader.dataset.videos)} videos')

    val_loader = create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=config['data'].get('val_batch_size', 16),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('val_workers', config['data'].get('workers', 4)),
        is_train=False,
        data_type=data_type,
        cache_max_videos=config['data'].get('cache_max_videos'),
        prefetch_factor=config['data'].get('prefetch_factor'),
    )

    console.info(f'Training: {data_type} data, {n_epochs} epochs')
    console.info(f'LR: {train_cfg["lr"]}, batch_size: {train_cfg["batch_size"]}')
    console.info(f'Train samples: {len(train_loader.dataset)}')

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
    console.info(f'Run: {run_dir}')

    metrics_path = run_dir / 'metrics.json'
    metrics = []
    save_interval = config['logging'].get('save_interval', 5)
    timing_path = run_dir / 'batch_timing.json'
    all_batch_times = []

    try:
        for epoch in range(start_epoch, n_epochs):
            batch_times = []
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler, batch_times)
            all_batch_times.append(batch_times)

            def _write_timing(path, data):
                path.write_text(json.dumps(data, indent=2))
            _IO_QUEUE.put((_write_timing, (timing_path, all_batch_times), {}))

            scheduler.step()

            val_psnr = None
            val_ssim = None
            is_new_best = False

            if epoch % config['logging'].get('val_interval', 1) == 0:
                # Avoid the heavy CUDA cache flush that causes periodic stalls.
                if config['logging'].get('empty_cuda_cache', False):
                    torch.cuda.empty_cache()
                val_psnr, val_ssim = validate(model, val_loader, device)

                def arrow(curr, prev):
                    if prev is None:
                        return '->'
                    return '^' if curr > prev else 'v'

                l_arrow = arrow(train_loss, prev_loss)
                p_arrow = arrow(val_psnr, prev_psnr)
                s_arrow = arrow(val_ssim, prev_ssim)

                flags = []
                if epoch == 0:
                    flags.append('first')
                is_new_best = val_psnr > best_psnr
                if is_new_best:
                    best_psnr = val_psnr
                    flags.append('best')
                if prev_psnr is not None:
                    diff = val_psnr - prev_psnr
                    if diff < -2.0:
                        flags.append('CRASH')
                    elif diff < -0.5:
                        flags.append('WARN')
                if val_psnr >= 20:
                    flags.append('GOOD')
                if up_streak >= 3:
                    flags.append('STREAK')

                flag_str = ' | ' + ' '.join(flags) if flags else ''
                console.info(f'Epoch {epoch:3d}:  {l_arrow} loss={train_loss:.4f}  |  '
                             f'{p_arrow} psnr={val_psnr:5.2f}  |  '
                             f'{s_arrow} ssim={val_ssim:.4f}{flag_str}')

                prev_loss = train_loss
                prev_psnr = val_psnr
                prev_ssim = val_ssim

            save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir,
                            is_best=is_new_best if val_psnr is not None else False)

            if save_interval > 0 and epoch % save_interval == 0 and epoch > 0:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                epoch_path = run_dir / f'epoch_{epoch:03d}_{ts}.pth'
                state = {
                    'epoch': epoch,
                    'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'optimizer_state_dict': {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                                             for k, v in optimizer.state_dict().items()},
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                }
                _IO_QUEUE.put((torch.save, (state, str(epoch_path)), {}))

            metrics.append({"epoch": epoch, "loss": round(train_loss, 4),
                            "psnr": round(val_psnr, 4) if val_psnr is not None else None,
                            "ssim": round(val_ssim, 4) if val_ssim is not None else None})

            def _write_metrics(path, data):
                path.write_text(json.dumps(data, indent=2))
            _IO_QUEUE.put((_write_metrics, (metrics_path, metrics), {}))
    finally:
        stop_io_worker()

    console.success('Training complete.')


if __name__ == '__main__':
    main()
