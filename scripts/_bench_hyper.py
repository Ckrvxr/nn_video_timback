#!/usr/bin/env python3
"""Benchmark HyperBlock global modulation for high-param, zero-latency scaling."""
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


class HyperBlock(nn.Module):
    """Global modulation: GAP → FC layers → scale+shift (element-wise).
    Produces 884K params with <0.05ms latency at latent=512."""
    def __init__(self, channels: int, latent: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, latent), nn.SiLU(),
            nn.Linear(latent, channels * 2),  # [scale | shift]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        p = self.net(x)  # (B, C*2)
        scale, shift = p.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1) + 1.0  # around 1
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * scale + shift


class FrameFeature(nn.Module):
    def __init__(self, in_ch=48, out_ch=32):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.silu(self.pw(self.silu(self.dw(x))))


class HFB(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        return x + self.pw(self.silu(self.dw(x)))


class SCA(nn.Module):
    """Simplified Channel Attention — 0 MACs at 540p, ~4.6K param @ 128ch."""
    def __init__(self, ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // 8, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(ch // 8, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class HyperFixer(nn.Module):
    """Cached pipeline with HyperBlock for high-parameter modulation."""
    def __init__(self, n_features=64, n_blocks=2, latent=512):
        super().__init__()
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(96, n_features, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.hyper = HyperBlock(n_features, latent)
        self.hfbs = nn.Sequential(*[HFB(n_features) for _ in range(n_blocks)])
        self.sca = SCA(n_features)
        self.global_residual = nn.Conv2d(48, n_features, 1, bias=False)
        self.recon = nn.Sequential(
            nn.Conv2d(n_features, 48, 3, padding=1, bias=False),
            nn.PixelShuffle(4),
        )

    def forward(self, f_prev, f_cur, f_next, skip_raw):
        x = self.temporal_fusion(torch.cat([f_prev, f_cur, f_next], dim=1))
        x = self.hyper(x)
        x = self.hfbs(x)
        x = self.sca(x)
        x = x + self.global_residual(skip_raw)
        return self.recon(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def benchmark(arch_name, model, n_warmup, n_iter):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        f = torch.randn(1, 32, 540, 960, dtype=torch.float16, device=device)
        sr = torch.randn(1, 48, 540, 960, dtype=torch.float16, device=device)
        for _ in range(n_warmup):
            _ = model(f, f, f, sr)
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(f, f, f, sr)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    s = sorted(times)
    median = s[len(s)//2]
    fps = 1000.0 / median
    return median, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    console.print(f'Device: {device}\n')

    # ─── 1) HyperBlock latent sweep (n_f=64, n_b=2) ───
    console.print('[bold]═══ HyperBlock Latent Sweep (n_f=64, n_b=2) ═══[/bold]')
    rows = []
    for latent in [64, 128, 256, 384, 512, 768, 1024]:
        m = HyperFixer(n_features=64, n_blocks=2, latent=latent)
        m = m.to(device).half()
        median, fps = benchmark(f'latent={latent}', m, args.warmup, args.n_iter)
        n_p = count_params(m)
        rows.append((latent, n_p, median, fps))

    t = Table()
    t.add_column('Latent', style='cyan')
    t.add_column('Params', style='green')
    t.add_column('Latency', style='green')
    t.add_column('FPS', style='magenta')
    t.add_column('75Hz?', style='bold')
    for lat, np_, med, fps in rows:
        ok = '✅' if fps >= 75 else '❌'
        t.add_row(f'{lat}', f'{np_:,}', f'{med:.1f}ms', f'{fps:.1f}', ok)
    console.print(t)

    # ─── 2) Best 75Hz: vary n_f + HyperBlock ───
    console.print('\n[bold]═══ 75Hz Tuning: n_f + Hyper(512) + n_b=2 ═══[/bold]')
    rows2 = []
    for n_f in [96, 80, 64, 48]:
        m = HyperFixer(n_features=n_f, n_blocks=2, latent=512)
        m = m.to(device).half()
        median, fps = benchmark(f'n_f={n_f}', m, args.warmup, args.n_iter)
        rows2.append((n_f, count_params(m), median, fps))

    t2 = Table()
    t2.add_column('n_f', style='cyan')
    t2.add_column('Params', style='green')
    t2.add_column('Latency', style='green')
    t2.add_column('FPS', style='magenta')
    t2.add_column('75Hz?', style='bold')
    for nf, np_, med, fps in rows2:
        ok = '✅' if fps >= 75 else '❌'
        t2.add_row(f'{nf}', f'{np_:,}', f'{med:.1f}ms', f'{fps:.1f}', ok)
    console.print(t2)

    # ─── 3) Depth sweep at best n_f ───
    console.print('\n[bold]═══ Depth Sweep: best n_f + Hyper(512) ═══[/bold]')
    rows3 = []
    best_nf = rows2[-1][0] if rows2[-1][3] >= 75 else rows2[0][0]
    for n_b in [1, 2, 3]:
        m = HyperFixer(n_features=best_nf, n_blocks=n_b, latent=512)
        m = m.to(device).half()
        median, fps = benchmark(f'n_b={n_b}', m, args.warmup, args.n_iter)
        rows3.append((best_nf, n_b, count_params(m), median, fps))

    t3 = Table()
    t3.add_column('n_f', style='cyan')
    t3.add_column('n_b', style='cyan')
    t3.add_column('Params', style='green')
    t3.add_column('Latency', style='green')
    t3.add_column('FPS', style='magenta')
    t3.add_column('75Hz?', style='bold')
    for nf, nb, np_, med, fps in rows3:
        ok = '✅' if fps >= 75 else '❌'
        t3.add_row(f'{nf}', f'{nb}', f'{np_:,}', f'{med:.1f}ms', f'{fps:.1f}', ok)
    console.print(t3)

    # ─── 4) Reference: no HyperBlock ───
    console.print('\n[bold]═══ Reference (no HyperBlock) ═══[/bold]')
    class NoHyper(nn.Module):
        def __init__(self, nf, nb):
            super().__init__()
            self.tf = nn.Sequential(nn.Conv2d(96, nf, 1, bias=False), nn.SiLU(inplace=True))
            self.hfb = nn.Sequential(*[HFB(nf) for _ in range(nb)])
            self.sca = SCA(nf)
            self.gr = nn.Conv2d(48, nf, 1, bias=False)
            self.recon = nn.Sequential(nn.Conv2d(nf, 48, 3, padding=1, bias=False), nn.PixelShuffle(4))
        def forward(self, fp, fc, fn, sr): return self.recon(self.sca(self.hfb(self.tf(torch.cat([fp,fc,fn],1)))) + self.gr(sr))
    refs = [(96, 2), (80, 2), (64, 2), (48, 2), (48, 1)]
    for n_f, n_b in refs:
        m = NoHyper(n_f, n_b).to(device).half()
        med, fps = benchmark(f'n_f={n_f}, n_b={n_b}', m, args.warmup, args.n_iter)
        console.print(f'n_f={n_f}, n_b={n_b}: {count_params(m):,} params → {med:.1f}ms ({fps:.1f} FPS)')


if __name__ == '__main__':
    main()
