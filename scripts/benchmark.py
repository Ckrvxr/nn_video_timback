#!/usr/bin/env python3
"""Training speed benchmarks: DataLoader throughput, step breakdown, epoch timing."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from yaml import safe_load
from rich.table import Table

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch
from utils.console import console


def setup(config_path: str, device: str = 'auto'):
    config = safe_load(open(config_path))
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None
    return config, device, model, criterion, optimizer, scaler


def make_loader(config, batch_size=None, workers=None, is_train=True, scales=None):
    return create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=batch_size or config['training']['batch_size'],
        scales=scales or [1],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=workers if workers is not None else config['data'].get('workers', 4) if is_train else 0,
        is_train=is_train,
        data_type=config['data'].get('data_type', 'compressed'),
    )


def train_step(model, criterion, optimizer, batch, scaler=None):
    lr = batch['lr_frames']
    hr = batch['hr']
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
    return loss_dict


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ─── Subcommands ────────────────────────────────────────────

def cmd_speed(args):
    config, device, model, criterion, optimizer, scaler = setup(args.config, args.device)
    console.print(f'Config: {args.config}')
    console.print(f'Device: {device}')
    console.print(f"batch_size={config['training']['batch_size']}, workers={config['data'].get('workers', 4)}")

    # DataLoader
    console.print('\n[1/2] DataLoader...')
    loader = make_loader(config, is_train=True)
    lt = []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        lt.append(time.perf_counter() - t0)
        t0 = time.perf_counter()

    t = Table()
    t.add_column('Metric', style='cyan')
    t.add_column('Value', style='green')
    t.add_row('Dataset samples', str(len(loader.dataset)))
    t.add_row('First batch', f'{lt[0]:.3f}s')
    if len(lt) > 1:
        a = avg(lt[1:])
        t.add_row(f'Avg batch (n={len(lt)-1})', f'{a:.3f}s')
        t.add_row('it/s', f'{1.0/a:.2f}')
    console.print(t)

    # Train step
    console.print('\n[2/2] Train step (workers=0)...')
    loader0 = make_loader(config, workers=0, is_train=True)
    st = []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader0):
        if i >= args.n_steps:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        train_step(model, criterion, optimizer, batch, scaler)
        st.append(time.perf_counter() - t0)
        t0 = time.perf_counter()

    t2 = Table()
    t2.add_column('Metric', style='cyan')
    t2.add_column('Value', style='green')
    t2.add_row('First step', f'{st[0]:.3f}s')
    if len(st) > 1:
        a = avg(st[1:])
        t2.add_row(f'Avg step (n={len(st)-1})', f'{a:.3f}s')
        t2.add_row('it/s', f'{1.0/a:.2f}')
    console.print(t2)


def cmd_breakdown(args):
    config, device, model, criterion, optimizer, scaler = setup(args.config, args.device)
    console.print(f'Config: {args.config}')
    console.print(f'Device: {device}')

    loader = make_loader(config, workers=0, is_train=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    console.print(f"lr={list(batch['lr_frames'].shape)}, hr={list(batch['hr'].shape)}, scale={batch['scale']}")

    for _ in range(args.n_warmup):
        train_step(model, criterion, optimizer, batch, scaler)

    fwd_t, loss_t, bwd_t, total_t = [], [], [], []
    lr, hr, scale = batch['lr_frames'], batch['hr'], batch['scale']

    for _ in range(args.n_iter):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss_dict = criterion(pred, hr)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.zero_grad()
        loss_dict['total'].backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        fwd_t.append(t1 - t0)
        loss_t.append(t2 - t1)
        bwd_t.append(t3 - t2)
        total_t.append(t4 - t0)

    total_avg = avg(total_t)
    t = Table()
    t.add_column('Phase', style='cyan')
    t.add_column('Avg', style='green')
    t.add_column('%', style='yellow')
    t.add_row('Forward', f'{avg(fwd_t)*1000:.2f}ms', f'{avg(fwd_t)/total_avg*100:.0f}%')
    t.add_row('Loss', f'{avg(loss_t)*1000:.2f}ms', f'{avg(loss_t)/total_avg*100:.0f}%')
    t.add_row('Backward', f'{avg(bwd_t)*1000:.2f}ms', f'{avg(bwd_t)/total_avg*100:.0f}%')
    t.add_row('Optimizer', f'{avg(opt_t)*1000:.2f}ms' if 'opt_t' in dir() else '', '')
    t.add_row('Total', f'{total_avg*1000:.2f}ms', '100%')
    console.print(t)


def cmd_epoch(args):
    config, device, model, criterion, optimizer, scaler = setup(args.config, args.device)
    console.print(f'Config: {args.config}')
    console.print(f'Device: {device}')

    train_loader = make_loader(config, is_train=True)
    val_dataset = make_loader(config, batch_size=config['data'].get('val_batch_size', 16), workers=0, is_train=False)

    for batch in train_loader:
        break

    model.train()
    t0 = time.perf_counter()
    n_batches = 0
    for batch in train_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        train_step(model, criterion, optimizer, batch, scaler)
        n_batches += 1
    train_time = time.perf_counter() - t0

    model.eval()
    t0 = time.perf_counter()
    total_psnr = total_ssim = n = 0
    with torch.no_grad():
        for item in val_dataset:
            lr = item['lr_frames'].to(device)
            hr = item['hr'].to(device)
            scale = item['scale']
            mid = lr.size(0) // 2
            pred = model(lr[mid-1:mid], lr[mid:mid+1], lr[mid+1:mid+2], scale=scale)
            total_psnr += calculate_psnr_batch(pred, hr.unsqueeze(0)).sum().item()
            total_ssim += calculate_ssim_batch(pred, hr.unsqueeze(0)).sum().item()
            n += 1
            if n >= args.max_val:
                break
    val_time = time.perf_counter() - t0

    t = Table()
    t.add_column('Phase', style='cyan')
    t.add_column('Value', style='green')
    t.add_row('Train epoch', f'{train_time:.1f}s ({n_batches} batches)')
    t.add_row('Validation', f'{val_time:.1f}s ({n} samples)')
    t.add_row('Val PSNR', f'{total_psnr/max(1,n):.2f}')
    t.add_row('Val SSIM', f'{total_ssim/max(1,n):.4f}')
    console.print(t)


def cmd_infer_latency(args):
    config = safe_load(open(args.config))
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device).eval()

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model = model.half()
    console.print(f'Config: {args.config}')
    console.print(f'Device: {device}')
    console.print(f'n_features={config["model"]["n_features"]}, n_blocks={config["model"]["n_blocks"]}')
    console.print(f'FP16: {args.fp16}\n')

    scenarios = [
        ('540p → 4K (×4)',   (960, 540), 4),
        ('1080p → 4K (×2)',  (1920, 1080), 2),
        ('4K → 4K (×1)',     (3840, 2160), 1),
    ]

    t = Table(title='Inference Latency')
    t.add_column('Scenario', style='cyan')
    t.add_column('Median', style='green')
    t.add_column('Mean', style='green')
    t.add_column('p95', style='yellow')
    t.add_column('Min', style='white')
    t.add_column('FPS', style='magenta')

    for label, (w, h), scale in scenarios:
        inp_w, inp_h = w // scale, h // scale
        frame_prev = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)
        frame_cur  = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)
        frame_next = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)

        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(frame_prev, frame_cur, frame_next, scale=scale)

        times = []
        with torch.no_grad():
            for _ in range(args.n_iter):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(frame_prev, frame_cur, frame_next, scale=scale)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        sorted_t = sorted(times)
        median = sorted_t[len(sorted_t) // 2]
        mean = sum(times) / len(times)
        p95 = sorted_t[int(len(sorted_t) * 0.95)]
        min_t = sorted_t[0]
        fps = 1000.0 / median

        t.add_row(label, f'{median:.1f}ms', f'{mean:.1f}ms',
                  f'{p95:.1f}ms', f'{min_t:.1f}ms', f'{fps:.1f}')

    console.print(t)


# ─── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Training speed benchmarks')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='auto')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_speed = sub.add_parser('speed', help='DataLoader + train step throughput')
    p_speed.set_defaults(func=cmd_speed)
    p_speed.add_argument('--n-batches', type=int, default=20)
    p_speed.add_argument('--n-steps', type=int, default=20)

    p_bd = sub.add_parser('breakdown', help='Forward/loss/backward step breakdown')
    p_bd.set_defaults(func=cmd_breakdown)
    p_bd.add_argument('--n-warmup', type=int, default=3)
    p_bd.add_argument('--n-iter', type=int, default=10)

    p_ep = sub.add_parser('epoch', help='Full epoch + validation timing')
    p_ep.set_defaults(func=cmd_epoch)
    p_ep.add_argument('--max-val', type=int, default=100)

    p_il = sub.add_parser('infer-latency', help='Inference latency at various resolutions')
    p_il.set_defaults(func=cmd_infer_latency)
    p_il.add_argument('--warmup', type=int, default=5)
    p_il.add_argument('--n-iter', type=int, default=20)
    p_il.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
