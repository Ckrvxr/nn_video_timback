#!/usr/bin/env python3
"""Benchmark Dual-Speed AV1-Fixer architecture."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from rich.table import Table
from rich.console import Console

console = Console()


class FrameFeature(nn.Module):
    """Shared across frames: 48→32ch @ 540p"""
    def __init__(self, in_ch=48, out_ch=32):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.dw(x))
        return self.silu(self.pw(x))


class HFB(nn.Module):
    """Hardware-Friendly Block: DW→SiLU→PW + skip"""
    def __init__(self, ch=64):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return x + self.pw(self.silu(self.dw(x)))


class DualSpeedAV1Fixer(nn.Module):
    """Takes 3 raw frames, outputs 1 restored frame."""
    def __init__(self, n_features=64, n_blocks=4):
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)

        self.frame_feature = FrameFeature(in_ch=48, out_ch=32)

        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(96, n_features, 1, bias=False),
            nn.SiLU(inplace=True),
        )

        self.hfbs = nn.Sequential(*[HFB(n_features) for _ in range(n_blocks)])

        self.global_residual = nn.Conv2d(48, n_features, 1, bias=False)

        self.recon = nn.Sequential(
            nn.Conv2d(n_features, 48, 3, padding=1, bias=False),
            nn.PixelShuffle(4),
        )

    def forward(self, f_prev, f_cur, f_next):
        # Each frame goes through PixelUnshuffle + FrameFeature
        pu_prev = self.pixel_unshuffle(f_prev)
        pu_cur  = self.pixel_unshuffle(f_cur)
        pu_next = self.pixel_unshuffle(f_next)

        f_prev = self.frame_feature(pu_prev)
        f_cur  = self.frame_feature(pu_cur)
        f_next = self.frame_feature(pu_next)

        # Temporal fusion: cat 3 frames
        fused = self.temporal_fusion(torch.cat([f_prev, f_cur, f_next], dim=1))

        # Backbone
        out = self.hfbs(fused)

        # Global residual: t_cur frame's PixelUnshuffle feature (48ch → 64ch)
        skip = self.global_residual(pu_cur)

        return self.recon(out + skip)


class DualSpeedAV1FixerCached(nn.Module):
    """Takes 3 pre-computed features (cache hit), no PixelUnshuffle/FrameFeature."""
    def __init__(self, n_features=64, n_blocks=4):
        super().__init__()
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(96, n_features, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.hfbs = nn.Sequential(*[HFB(n_features) for _ in range(n_blocks)])
        self.global_residual = nn.Conv2d(48, n_features, 1, bias=False)
        self.recon = nn.Sequential(
            nn.Conv2d(n_features, 48, 3, padding=1, bias=False),
            nn.PixelShuffle(4),
        )

    def forward(self, f_prev, f_cur, f_next, skip_raw):
        fused = self.temporal_fusion(torch.cat([f_prev, f_cur, f_next], dim=1))
        out = self.hfbs(fused)
        skip = self.global_residual(skip_raw)
        return self.recon(out + skip)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_model(name, model, n_warmup, n_iter, dtype, device, uses_cache=False):
    model = model.to(device).eval()
    if dtype == torch.float16:
        model = model.half()

    # Input shapes: 3 raw 4K frames
    prev = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)
    cur  = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)
    next_ = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)

    # For cached variant: pre-compute features (once)
    if uses_cache:
        feat_model = FrameFeature(in_ch=48, out_ch=32).to(device).eval()
        if dtype == torch.float16:
            feat_model = feat_model.half()
        pu = nn.PixelUnshuffle(4).to(device)
        with torch.no_grad():
            f_prev = feat_model(pu(prev))
            f_cur  = feat_model(pu(cur))
            f_next = feat_model(pu(next_))
            skip_raw = pu(cur)
        # Time only the cached model
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(f_prev, f_cur, f_next, skip_raw)
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(f_prev, f_cur, f_next, skip_raw)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        # Measure FrameFeature for 1 new frame (the cached path per frame)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = feat_model(pu(cur))
            torch.cuda.synchronize()

        feat_times = []
        with torch.no_grad():
            for _ in range(n_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = feat_model(pu(cur))
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                feat_times.append((t1 - t0) * 1000)

        # Total per-frame cached: FrameFeature(1 new frame) + Backbone
        feat_med = sorted(feat_times)[len(feat_times)//2]
        bb_med = sorted(times)[len(times)//2]
        total_med = feat_med + bb_med

        console.print(f'\n[bold]{name}[/bold] ({count_params(model):,} params)')
        t = Table()
        t.add_column('Metric', style='cyan')
        t.add_column('Value', style='green')
        t.add_row('Backbone + Recon (cached)', f'{bb_med:.1f}ms')
        t.add_row('FrameFeature (1 new frame)', f'{feat_med:.1f}ms')
        t.add_row('Total per-frame', f'{total_med:.1f}ms')
        t.add_row('FPS', f'{1000/total_med:.1f}')
        console.print(t)
    else:
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(prev, cur, next_)
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(prev, cur, next_)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        sorted_t = sorted(times)
        median = sorted_t[len(sorted_t)//2]
        mean = sum(times)/len(times)
        p95 = sorted_t[int(len(sorted_t)*0.95)]
        fps = 1000.0/median

        console.print(f'\n[bold]{name}[/bold] ({count_params(model):,} params)')
        t = Table()
        t.add_column('Metric', style='cyan')
        t.add_column('Value', style='green')
        t.add_row('Median', f'{median:.1f}ms')
        t.add_row('Mean', f'{mean:.1f}ms')
        t.add_row('p95', f'{p95:.1f}ms')
        t.add_row('FPS', f'{fps:.1f}')
        console.print(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if args.fp16 else torch.float32

    console.print(f'Device: {device}, FP16: {args.fp16}')

    # Scenario 1: Full pipeline (no cache) — 3 raw frames in
    console.print('[bold]\n═══ Full Pipeline (3 raw frames) ═══[/bold]')
    for n_f, n_b in [(64, 4), (48, 4), (32, 4)]:
        m = DualSpeedAV1Fixer(n_features=n_f, n_blocks=n_b)
        benchmark_model(f'n_f={n_f}, n_b={n_b}', m, args.warmup, args.n_iter, dtype, device, uses_cache=False)

    # Scenario 2: Cached pipeline
    console.print('[bold]\n═══ Cached Pipeline (75Hz sweep) ═══[/bold]')

    def cached_table(name, model):
        model = model.to(device).eval()
        if dtype == torch.float16:
            model = model.half()
        feat_model = FrameFeature(in_ch=48, out_ch=32).to(device).eval()
        if dtype == torch.float16:
            feat_model = feat_model.half()
        pu = nn.PixelUnshuffle(4).to(device)
        cur = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)
        with torch.no_grad():
            f_prev = feat_model(pu(torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)))
            f_cur  = feat_model(pu(cur))
            f_next = feat_model(pu(torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)))
            skip_raw = pu(cur)
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(f_prev, f_cur, f_next, skip_raw)
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(args.n_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(f_prev, f_cur, f_next, skip_raw)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        with torch.no_grad():
            for _ in range(args.warmup):
                _ = feat_model(pu(cur))
            torch.cuda.synchronize()
        ft = []
        with torch.no_grad():
            for _ in range(args.n_iter):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = feat_model(pu(cur))
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                ft.append((t1 - t0) * 1000)

        bb_m = sorted(times)[len(times)//2]
        ff_m = sorted(ft)[len(ft)//2]
        total = bb_m + ff_m
        fps = 1000.0/total
        return total, fps, sum(p.numel() for p in model.parameters()), bb_m, ff_m

    configs = [
        (64, 4), (64, 3), (64, 2),
        (48, 4), (48, 3), (48, 2),
        (40, 4), (40, 3), (40, 2),
        (32, 4), (32, 3), (32, 2),
    ]
    rows = []
    for n_f, n_b in configs:
        m = DualSpeedAV1FixerCached(n_features=n_f, n_blocks=n_b)
        try:
            total, fps, params, bb, ff = cached_table(f'n_f={n_f}, n_b={n_b}', m)
        except Exception as e:
            continue
        rows.append((n_f, n_b, params, bb, ff, total, fps))

    t = Table(title='Cached Pipeline @ 75Hz Target')
    t.add_column('n_f', style='cyan')
    t.add_column('n_b', style='cyan')
    t.add_column('Params', style='green')
    t.add_column('Backbone', style='green')
    t.add_column('FrameFeat', style='green')
    t.add_column('Total', style='yellow')
    t.add_column('FPS', style='magenta')
    t.add_column('75Hz?', style='bold')

    for n_f, n_b, params, bb, ff, total, fps in rows:
        ok = '✅' if fps >= 75 else '❌'
        style = 'green' if fps >= 75 else 'red'
        t.add_row(f'{n_f}', f'{n_b}', f'{params:,}',
                  f'{bb:.1f}ms', f'{ff:.1f}ms',
                  f'{total:.1f}ms', f'{fps:.1f}', ok)
    console.print(t)


if __name__ == '__main__':
    main()
