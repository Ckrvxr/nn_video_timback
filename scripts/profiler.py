#!/usr/bin/env python3
"""Comprehensive training profiler. Measures every stage to identify bottlenecks."""
import argparse
import gc
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from yaml import safe_load

from utils.console import console


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

def bench_dataloader(config, n_batches=20, device=None, save_path=None):
    console.info('--- DataLoader Throughput ---')
    all_workers_data = {}
    rows = []
    for w in [0, 2, 4]:
        gc.collect()
        loader = make_loader(config, workers=w, is_train=True)
        batch_times = []
        t0 = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            now = time.perf_counter()
            dt = now - t0
            batch_times.append(dt)
            t0 = now

        cold_start = batch_times[0] if batch_times else 0.0
        if len(batch_times) > 1:
            tail = batch_times[1:]
            tail_ms = [t * 1000 for t in tail]
            tail_ms.sort()
            p50 = tail_ms[len(tail_ms) // 2]
            p95 = tail_ms[int(len(tail_ms) * 0.95)]
            p99 = tail_ms[int(len(tail_ms) * 0.99)]
            p99_9 = tail_ms[int(len(tail_ms) * 0.999)]
            mx = tail_ms[-1]
            avg_bt = avg(tail)
            its = 1.0 / avg_bt
        else:
            p50 = p95 = p99 = p99_9 = mx = 0.0
            avg_bt = float('nan')
            its = float('nan')

        rows.append((w, cold_start, avg_bt, its, p50, p95, p99, p99_9, mx))
        all_workers_data[w] = {'cold_start_sec': cold_start, 'batch_times_sec': batch_times}

    header = f"{'workers':>8} | {'ColdStart':>9} | {'P50':>8} | {'P95':>8} | {'P99':>8} | {'P99.9':>8} | {'Max':>8} | {'Max/P50':>8} | {'Avg':>8}"
    console.info(header)
    console.info('-' * len(header))
    for w, cold, avg_bt, its, p50, p95, p99, p99_9, mx in rows:
        ratio = f'{mx / p50:.1f}x' if p50 > 0 else 'N/A'
        line = (f'workers={w:>2} | {fmt_s(cold):>9} | {fmt_ms(p50/1000):>8} | {fmt_ms(p95/1000):>8} | '
                f'{fmt_ms(p99/1000):>8} | {fmt_ms(p99_9/1000):>8} | {fmt_ms(mx/1000):>8} | {ratio:>8} | {fmt_ms(avg_bt):>8}')
        console.info(line)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(all_workers_data, indent=2))
        console.info(f'Saved raw timing to {save_path}')

    return rows


# ─── 2. CPU→GPU transfer + interpolation ─────────────────

def bench_transfer(config, device, n_warmup=5, n_iter=30):
    console.info('--- CPU->GPU Transfer + Interpolation ---')
    loader = make_loader(config, workers=0, is_train=True)

    # get one CPU batch
    for batch in loader:
        break
    lr_cpu = batch['lr_frames']  # (B, T, 3, H, W) on CPU
    scale = 1  # training always uses scale=1
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

    console.info(f'  {"Phase":<25} {"Median":>10} {"Mean":>10} {"p95":>10}')
    console.info(f'  {"-"*55}')
    console.info(f'  {"CPU->GPU transfer":<25} {fmt_ms(median(transfer_times)):>10} {fmt_ms(avg(transfer_times)):>10} {fmt_ms(pct(transfer_times, 0.95)):>10}')
    if scale > 1:
        console.info(f'  {"Interpolation (scale>1)":<25} {fmt_ms(median(interpolation_times)):>10} {fmt_ms(avg(interpolation_times)):>10} {fmt_ms(pct(interpolation_times, 0.95)):>10}')
    total_median = median([a + b for a, b in zip(transfer_times, interpolation_times)])
    console.info(f'  {"Total preprocess":<25} {fmt_ms(total_median):>10}')
    return transfer_times, interpolation_times


# ─── 3. GPU pipeline breakdown ───────────────────────────

def bench_gpu_pipeline(config, device, n_warmup=5, n_iter=30):
    console.info('--- GPU Pipeline Breakdown ---')
    model = build_model(config, device).train()
    from losses.composite import CompositeLoss
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None

    amp_enabled = scaler is not None
    batch = fetch_one_batch(make_loader(config, workers=0, is_train=True), device)
    lr, hr = batch['lr_frames'], batch['hr']
    scale = 1

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
    console.info(f'  {"Phase":<15} {"Median":>10} {"Mean":>10} {"%":>8}')
    console.info(f'  {"-"*43}')
    phases = [
        ('Forward', fwd_t),
        ('Loss', loss_t),
        ('Backward', bwd_t),
        ('Optimizer', optim_t),
        ('Total', total_t),
    ]
    for name, ts in phases:
        console.info(f'  {name:<15} {fmt_ms(median(ts)):>10} {fmt_ms(avg(ts)):>10} {avg(ts) / total_avg * 100:.0f}%')
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
    console.info('--- Submodule Latency ---')
    model = build_model(config, device).eval()
    batch = fetch_one_batch(make_loader(config, workers=0, is_train=True), device)
    lr = batch['lr_frames']
    scale = 1

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

    total_m = median(full_ts)
    console.info(f'  {"Module":<20} {"Median":>10} {"% of total":>12}')
    console.info(f'  {"-"*42}')
    console.info(f'  {"Full model":<20} {fmt_ms(total_m):>10} {"100%":>12}')
    for name, ts in mod_times.items():
        m = median(ts)
        console.info(f'  {name:<20} {fmt_ms(m):>10} {m / total_m * 100:.0f}%')
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
    console.info('--- Memory Footprint ---')
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
    lr, hr = batch['lr_frames'], batch['hr']

    # full train step
    optimizer.zero_grad()
    if scaler:
        with torch.amp.autocast(device_type='cuda'):
            pred = model(lr[:, 0], lr[:, 1], lr[:, 2])
            loss_dict = criterion(pred, hr)
        scaler.scale(loss_dict['total']).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred = model(lr[:, 0], lr[:, 1], lr[:, 2])
        loss_dict = criterion(pred, hr)
        loss_dict['total'].backward()
        optimizer.step()
    sync(device)

    peak = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    batch_bytes = (lr.numel() + hr.numel()) * lr.element_size()
    free_before = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)

    console.info(f'  {"Metric":<25} {"Value":>12}')
    console.info(f'  {"-"*37}')
    console.info(f'  {"Model params (est.)":<25} {model_params / 1024**2:.1f} MB')
    console.info(f'  {"Batch size":<25} {lr.shape[0]}.{lr.shape[2]}p')
    console.info(f'  {"Batch data":<25} {batch_bytes / 1024**2:.1f} MB')
    console.info(f'  {"Peak allocated VRAM":<25} {peak / 1024**2:.1f} MB')
    console.info(f'  {"Peak reserved VRAM":<25} {peak_reserved / 1024**2:.1f} MB')
    console.info(f'  {"Total GPU VRAM":<25} {torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MB')
    console.info(f'  {"Free after step":<25} {free_before / 1024**2:.1f} MB')
    return peak


# ─── 6. Bottleneck summary ──────────────────────────────

def print_summary(dl_rows, total_t):
    console.info('--- Bottleneck Summary ---')
    best_dl = min((r for r in dl_rows if r[2] != float('nan')), key=lambda r: r[2]) if dl_rows else None
    gpu_step = median(total_t) if total_t else 0

    if best_dl:
        dl_time = best_dl[2]
        dl_its = best_dl[3]
        console.info(f'  Best DataLoader workers: {best_dl[0]}')
        console.info(f'  Steady batch time:       {fmt_ms(dl_time)}')
        console.info(f'  GPU step time:           {fmt_ms(gpu_step)}')

        ratio = dl_time / gpu_step if gpu_step > 0 else float('inf')
        if ratio > 1.2:
            status = 'DataLoader bound (waiting for data)'
        elif ratio < 0.8:
            status = 'GPU bound (DataLoader faster than GPU)'
        else:
            status = 'Balanced'
        console.info(f'  DL / GPU ratio:          {ratio:.2f}x  ({status})')

        max_its = 1.0 / max(dl_time, gpu_step)
        console.info(f'  Max achievable iter/s:   {max_its:.2f}')
        console.info(f'  Samples/s (bs=16):       {max_its * 16:.0f}')


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
    parser.add_argument('--save', type=str, default=None, help='save raw batch timing JSON')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    config = safe_load(open(args.config))
    model_name = config['model'].get('name', 'av1_vsr')
    console.info(f'Config: {args.config}')
    console.info(f'Model: {model_name}')
    console.info(f'Device: {device}')

    dl_rows = []
    total_t = []

    if args.only is None or args.only == 'dataloader':
        dl_rows = bench_dataloader(config, n_batches=args.n_batches, device=device, save_path=args.save)

    if args.only is None or args.only == 'transfer':
        bench_transfer(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter)

    if args.only is None or args.only == 'gpu':
        *_, total_t = bench_gpu_pipeline(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter)

    if args.only is None or args.only == 'submodules':
        bench_submodules(config, device, n_warmup=args.n_warmup, n_iter=args.n_iter * 2)

    if args.only is None or args.only == 'memory':
        bench_memory(config, device)

    if args.only is None:
        print_summary(dl_rows, total_t)


if __name__ == '__main__':
    main()