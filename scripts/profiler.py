#!/usr/bin/env python3
"""Comprehensive training profiler. Measures every stage to identify bottlenecks."""
import argparse
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from yaml import safe_load
from rich.table import Table
from rich.console import Console

console = Console()


# ─── helpers ──────────────────────────────────────────────

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def median(lst):
    s = sorted(lst)
    return s[len(s) // 2]


def pct(lst, p):
    s = sorted(lst)
    return s[int(len(s) * p)]


def fmt_ms(sec):
    return f'{sec * 1000:.2f}ms'


def fmt_s(sec):
    if sec < 60:
        return f'{sec:.2f}s'
    return f'{sec / 60:.1f}m'


def build_model(config, device):
    from models.hyper_fixer import HyperFixer
    from models.av1_vsr import AV1VSR
    name = config['model'].get('name', 'av1_vsr')
    if name == 'hyper_fixer':
        m = HyperFixer(
            n_features=config['model']['n_features'],
            n_blocks=config['model']['n_blocks'],
            latent=config['model'].get('latent', 512),
        )
    else:
        m = AV1VSR(
            in_channels=3,
            n_features=config['model']['n_features'],
            n_blocks=config['model']['n_blocks'],
        )
    return m.to(device)


def make_loader(config, batch_size=None, workers=None, is_train=True):
    from utils.dataset import create_dataloader
    return create_dataloader(
        datasets=config['data']['datasets'],
        batch_size=batch_size or config['training']['batch_size'],
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=workers if workers is not None else 0,
        is_train=is_train,
        data_type=config['data'].get('data_type', 'compressed'),
    )


def synced_timer(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def fetch_one_batch(loader, device):
    """Pull a single batch, move to device, return it."""
    for batch in loader:
        break
    for k in ('lr_frames', 'hr'):
        batch[k] = batch[k].to(device, non_blocking=True)
    return batch


# ─── 1. DataLoader throughput ────────────────────────────

def bench_dataloader(config, n_batches=20, device=None):
    console.rule('[bold]1. DataLoader Throughput')
    rows = []
    for w in [0, 2, 4, 8]:
        gc.collect()
        loader = make_loader(config, workers=w, is_train=True)
        times = []
        t0 = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            now = time.perf_counter()
            times.append((batch['lr_frames'].shape[0], now - t0))
            t0 = now

        batch_sizes = [b[0] for b in times]
        batch_times = [b[1] for b in times]
        n = len(batch_times) - 1  # skip first (cold start)
        if n > 0:
            avg_bt = avg(batch_times[1:])
            its = 1.0 / avg_bt
            fps = its * avg(batch_sizes[1:])
        else:
            avg_bt = float('nan')
            its = float('nan')
            fps = float('nan')
        cold_start = batch_times[0] if batch_times else float('nan')
        rows.append((w, cold_start, avg_bt, its, fps))

    t = Table(title='DataLoader Throughput')
    t.add_column('Workers', style='cyan')
    t.add_column('Cold Start', style='yellow')
    t.add_column('Steady Batch', style='green')
    t.add_column('Batches/s', style='magenta')
    t.add_column('Frames/s', style='magenta')
    for w, cold, steady, its, fps in rows:
        t.add_row(f'{w}', fmt_s(cold), fmt_ms(steady), f'{its:.2f}', f'{fps:.0f}')
    console.print(t)
    return rows


# ─── 2. CPU→GPU transfer + interpolation ─────────────────

def bench_transfer(config, device, n_warmup=5, n_iter=30):
    console.rule('[bold]2. CPU→GPU Transfer + Interpolation')
    loader = make_loader(config, workers=0, is_train=True)

    # get one CPU batch
    for batch in loader:
        break
    lr_cpu = batch['lr_frames']  # (B, T, 3, H, W) on CPU
    scale = batch['scale']
    B, T, C, H, W = lr_cpu.shape

    # warmup
    for _ in range(n_warmup):
        _ = lr_cpu.to(device, non_blocking=True)
        sync(device)

    times = []
    for _ in range(n_iter):
        t0 = synced_timer(device)
        lr_gpu = lr_cpu.to(device, non_blocking=True)
        sync(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    transfer_times = times

    if scale > 1:
        interpolation_times = []
        lr_gpu = lr_cpu.to(device, non_blocking=True)
        for _ in range(n_iter):
            t0 = synced_timer(device)
            _ = nn.functional.interpolate(
                lr_gpu.view(B * T, C, H, W),
                size=(H // scale, W // scale),
                mode='bilinear', align_corners=False,
            )
            sync(device)
            t1 = time.perf_counter()
            interpolation_times.append(t1 - t0)
    else:
        interpolation_times = [0.0] * n_iter

    t = Table(title='Transfer + Preprocess')
    t.add_column('Phase', style='cyan')
    t.add_column('Median', style='green')
    t.add_column('Mean', style='green')
    t.add_column('p95', style='yellow')
    t.add_row('CPU→GPU transfer', fmt_ms(median(transfer_times)), fmt_ms(avg(transfer_times)), fmt_ms(pct(transfer_times, 0.95)))
    if scale > 1:
        t.add_row('Interpolation (scale>1)', fmt_ms(median(interpolation_times)), fmt_ms(avg(interpolation_times)), fmt_ms(pct(interpolation_times, 0.95)))
    t.add_row('Total preprocess', fmt_ms(median([a + b for a, b in zip(transfer_times, interpolation_times)])), '', '')
    console.print(t)
    return transfer_times, interpolation_times


# ─── 3. GPU pipeline breakdown ───────────────────────────

def bench_gpu_pipeline(config, device, n_warmup=5, n_iter=30):
    console.rule('[bold]3. GPU Pipeline Breakdown')
    model = build_model(config, device).train()
    from losses.composite import CompositeLoss
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None

    amp_enabled = scaler is not None
    batch = fetch_one_batch(make_loader(config, workers=0, is_train=True), device)
    lr, hr, scale = batch['lr_frames'], batch['hr'], batch['scale']

    # warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        if amp_enabled:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
                loss_dict = criterion(pred, hr)
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
            loss_dict = criterion(pred, hr)
            loss_dict['total'].backward()
            optimizer.step()
        sync(device)

    # measure
    fwd_t, loss_t, bwd_t, optim_t, total_t = [], [], [], [], []
    for _ in range(n_iter):
        optimizer.zero_grad()

        t0 = synced_timer(device)
        if amp_enabled:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
        else:
            pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
        sync(device)
        t1 = time.perf_counter()
        fwd_t.append(t1 - t0)

        if amp_enabled:
            loss_dict = criterion(pred, hr)
        else:
            loss_dict = criterion(pred, hr)
        sync(device)
        t2 = time.perf_counter()
        loss_t.append(t2 - t1)

        if amp_enabled:
            scaler.scale(loss_dict['total']).backward()
        else:
            loss_dict['total'].backward()
        sync(device)
        t3 = time.perf_counter()
        bwd_t.append(t3 - t2)

        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        sync(device)
        t4 = time.perf_counter()
        optim_t.append(t4 - t3)

        total_t.append(t4 - t0)

    total_avg = avg(total_t)
    t = Table(title='GPU Pipeline Breakdown')
    t.add_column('Phase', style='cyan')
    t.add_column('Median', style='green')
    t.add_column('Mean', style='green')
    t.add_column('%', style='yellow')
    phases = [
        ('Forward', fwd_t),
        ('Loss', loss_t),
        ('Backward', bwd_t),
        ('Optimizer', optim_t),
        ('Total', total_t),
    ]
    for name, ts in phases:
        t.add_row(name, fmt_ms(median(ts)), fmt_ms(avg(ts)), f'{avg(ts) / total_avg * 100:.0f}%')
    console.print(t)
    return fwd_t, loss_t, bwd_t, optim_t, total_t


# ─── 4. Submodule latency ────────────────────────────────

class HookTiming:
    def __init__(self):
        self.times = OrderedDict()
        self._hooks = []

    def _make_hook(self, name):
        def hook(_, __, output):
            if torch.is_grad_enabled():
                return
            t = time.perf_counter()
            self.times.setdefault(name, []).append(t)
        return hook

    def attach(self, module, name):
        h = module.register_forward_hook(self._make_hook(name))
        self._hooks.append(h)

    def detach_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def bench_submodules(config, device, n_warmup=5, n_iter=50):
    console.rule('[bold]4. Submodule Latency')
    model = build_model(config, device).eval()
    batch = fetch_one_batch(make_loader(config, workers=0, is_train=True), device)
    lr, scale = batch['lr_frames'], batch['scale']

    # attach hooks to key submodules
    ht = HookTiming()
    targets = {
        'pixel_unshuffle': model.pixel_unshuffle,
        'frame_feature': model.frame_feature,
        'temporal_fusion': model.temporal_fusion,
        'hyper': model.hyper,
        'hfbs': model.hfbs,
        'sca': model.sca,
        'global_residual': model.global_residual,
        'recon': model.recon,
    }
    for name, mod in targets.items():
        if hasattr(model, name):
            ht.attach(mod, name)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
        sync(device)

    # measure — hooks record end timestamps; diff of consecutive = module time
    ht.times.clear()
    with torch.no_grad():
        for _ in range(n_iter):
            ht.times.clear()
            t_start = synced_timer(device)
            _ = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
            sync(device)
            t_end = time.perf_counter()
            total = t_end - t_start
            # record total
            ht.times['_total'] = [total]

    # parse: each hook records when that module finishes
    # We measure by running separately with single hooks for precise timing
    # Better approach: re-run with per-module independent measurement
    ht.detach_all()

    # Actually do per-module timing properly
    mod_times = {}
    for name, mod in targets.items():
        if not hasattr(model, name):
            continue
        ts = []
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
            sync(device)
            for _ in range(n_iter):
                t0 = synced_timer(device)
                _ = mod(lr[:, 1], lr[:, 2], lr[:, 3]) if name == 'frame_feature' else _call_submod(model, name, lr, scale)
                sync(device)
                t1 = time.perf_counter()
                ts.append(t1 - t0)
        mod_times[name] = ts

    # Also time full model
    full_ts = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
        sync(device)
        for _ in range(n_iter):
            t0 = synced_timer(device)
            _ = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
            sync(device)
            t1 = time.perf_counter()
            full_ts.append(t1 - t0)

    t = Table(title='Submodule Latency')
    t.add_column('Module', style='cyan')
    t.add_column('Median', style='green')
    t.add_column('% of total', style='yellow')
    total_m = median(full_ts)
    t.add_row('Full model', fmt_ms(total_m), '100%')
    for name, ts in mod_times.items():
        m = median(ts)
        t.add_row(f'  {name}', fmt_ms(m), f'{m / total_m * 100:.0f}%')
    console.print(t)
    return mod_times, full_ts


def _call_submod(model, name, lr, scale):
    """Call a submodule with the right inputs — handles special cases."""
    if name == 'pixel_unshuffle':
        return model.pixel_unshuffle(lr[:, 2])
    elif name in ('frame_feature',):
        pu = model.pixel_unshuffle(lr[:, 2])
        return model.frame_feature(pu)
    elif name == 'temporal_fusion':
        f = model.frame_feature(model.pixel_unshuffle(lr[:, 2]))
        return model.temporal_fusion(torch.cat([f, f, f], dim=1))
    elif name == 'hyper':
        f = model.frame_feature(model.pixel_unshuffle(lr[:, 2]))
        fused = model.temporal_fusion(torch.cat([f, f, f], dim=1))
        return model.hyper(fused)
    elif name == 'hfbs':
        f = model.frame_feature(model.pixel_unshuffle(lr[:, 2]))
        fused = model.temporal_fusion(torch.cat([f, f, f], dim=1))
        x = model.hyper(fused)
        return model.hfbs(x)
    elif name == 'sca':
        f = model.frame_feature(model.pixel_unshuffle(lr[:, 2]))
        fused = model.temporal_fusion(torch.cat([f, f, f], dim=1))
        x = model.hyper(fused)
        x = model.hfbs(x)
        return model.sca(x)
    elif name == 'global_residual':
        pu = model.pixel_unshuffle(lr[:, 2])
        return model.global_residual(pu)
    elif name == 'recon':
        f = model.frame_feature(model.pixel_unshuffle(lr[:, 2]))
        fused = model.temporal_fusion(torch.cat([f, f, f], dim=1))
        x = model.hyper(fused)
        x = model.hfbs(x)
        x = model.sca(x)
        x = x + model.global_residual(model.pixel_unshuffle(lr[:, 2]))
        return model.recon(x)
    return None


# ─── 5. Memory ───────────────────────────────────────────

def bench_memory(config, device):
    console.rule('[bold]5. Memory Footprint')
    model = build_model(config, device).train()
    from losses.composite import CompositeLoss
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None

    model_params = sum(p.numel() for p in model.parameters()) * 4  # fp32 bytes
    if next(model.parameters()).dtype == torch.float16:
        model_params = model_params // 2

    torch.cuda.reset_peak_memory_stats(device)
    loader = make_loader(config, workers=0, is_train=True)
    batch = fetch_one_batch(loader, device)
    lr, hr, scale = batch['lr_frames'], batch['hr'], batch['scale']

    # full train step
    optimizer.zero_grad()
    if scaler:
        with torch.amp.autocast(device_type='cuda'):
            pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
            loss_dict = criterion(pred, hr)
        scaler.scale(loss_dict['total']).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred = model(lr[:, 1], lr[:, 2], lr[:, 3], scale=scale)
        loss_dict = criterion(pred, hr)
        loss_dict['total'].backward()
        optimizer.step()
    sync(device)

    peak = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    batch_bytes = (lr.numel() + hr.numel()) * lr.element_size()
    free_before = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    t = Table(title='Memory Footprint')
    t.add_column('Metric', style='cyan')
    t.add_column('Value', style='green')
    t.add_row('Model params (est.)', f'{model_params / 1024**2:.1f} MB')
    t.add_row('Batch size', f'{lr.shape[0]}×{lr.shape[2]}p')
    t.add_row('Batch data', f'{batch_bytes / 1024**2:.1f} MB')
    t.add_row('Peak allocated VRAM', f'{peak / 1024**2:.1f} MB')
    t.add_row('Peak reserved VRAM', f'{peak_reserved / 1024**2:.1f} MB')
    t.add_row('Total GPU VRAM', f'{torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MB')
    t.add_row('Free after step', f'{free_before / 1024**2:.1f} MB')
    console.print(t)
    return peak


# ─── 6. Bottleneck summary ──────────────────────────────

def print_summary(dl_rows, total_t):
    console.rule('[bold]6. Bottleneck Summary')
    best_dl = min((r for r in dl_rows if r[2] != float('nan')), key=lambda r: r[2]) if dl_rows else None
    gpu_step = median(total_t) if total_t else 0

    t = Table(title='Bottleneck Analysis')
    t.add_column('Metric', style='cyan')
    t.add_column('Value', style='green')
    t.add_column('Status', style='bold')

    if best_dl:
        dl_time = best_dl[2]
        dl_its = best_dl[3]
        t.add_row('Best DataLoader (workers)', f'{best_dl[0]}', '')
        t.add_row('Steady batch time', fmt_ms(dl_time), '')
        t.add_row('GPU step time', fmt_ms(gpu_step), '')

        ratio = dl_time / gpu_step if gpu_step > 0 else float('inf')
        if ratio > 1.2:
            status = '⚠️ DataLoader bound (waiting for data)'
        elif ratio < 0.8:
            status = '✅ GPU bound (DataLoader faster than GPU)'
        else:
            status = '~ Balanced'
        t.add_row('DL / GPU ratio', f'{ratio:.2f}x', status)

        max_its = 1.0 / max(dl_time, gpu_step)
        t.add_row('Max achievable iter/s', f'{max_its:.2f}', '')
        t.add_row('Samples/s', f'{max_its * 16:.0f}', '')  # batch_size=16

    console.print(t)


# ─── CLI ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Training pipeline profiler')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--n-batches', type=int, default=20, help='batches for DataLoader bench')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=30)
    parser.add_argument('--only', type=str, default=None,
                        choices=['dataloader', 'transfer', 'gpu', 'submodules', 'memory'])
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    config = safe_load(open(args.config))
    model_name = config['model'].get('name', 'av1_vsr')
    console.print(f'[bold]Config:[/bold] {args.config}')
    console.print(f'[bold]Model:[/bold] {model_name}')
    console.print(f'[bold]Device:[/bold] {device}\n')

    dl_rows = []
    total_t = []

    if args.only is None or args.only == 'dataloader':
        dl_rows = bench_dataloader(config, n_batches=args.n_batches, device=device)
        console.print()

    if args.only is None or args.only == 'transfer':
        bench_transfer(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter)
        console.print()

    if args.only is None or args.only == 'gpu':
        *_, total_t = bench_gpu_pipeline(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter)
        console.print()

    if args.only is None or args.only == 'submodules':
        bench_submodules(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter * 2)
        console.print()

    if args.only is None or args.only == 'memory':
        bench_memory(config, device)
        console.print()

    if args.only is None:
        print_summary(dl_rows, total_t)


if __name__ == '__main__':
    main()
