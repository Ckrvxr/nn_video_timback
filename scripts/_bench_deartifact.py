#!/usr/bin/env python3
"""Benchmark target de-artifact architecture (no scale, 4K→4K only)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from rich.table import Table

from models.components.alignment import SimpleWarpAlign
from models.components.fusion import TSAFusion
from models.components.resblock import ResBlock
from utils.console import console


class DeArtifactNet(nn.Module):
    def __init__(self, n_features: int = 16, n_blocks: int = 4):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(3, n_features, 3, stride=4, padding=1),
            nn.PReLU(n_features),
        )
        self.alignment = SimpleWarpAlign(n_features)
        self.fusion = TSAFusion(n_features)
        dilations = [1, 2, 2, 1]
        backbone = []
        for i in range(n_blocks):
            backbone.append(ResBlock(n_features, dilation=dilations[i % len(dilations)]))
        self.backbone = nn.Sequential(*backbone)
        self.conv_last = nn.Conv2d(n_features, n_features, 3, padding=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(n_features, 3, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        )

    def forward(self, prev, cur, next_):
        f_prev = self.conv_first(prev)
        f_cur = self.conv_first(cur)
        f_next = self.conv_first(next_)

        a_prev, a_next = self.alignment(f_prev, f_cur, f_next)
        fused = self.fusion(a_prev, f_cur, a_next)

        deep = self.backbone(fused)
        deep = self.conv_last(deep) + fused

        return torch.tanh(self.out(deep))


def benchmark(configs, device, dtype, warmup, n_iter):
    for label, model in configs:
        model = model.to(device).eval()
        if dtype == torch.float16:
            model = model.half()

        prev = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)
        cur  = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)
        next_ = torch.randn(1, 3, 2160, 3840, dtype=dtype, device=device)

        with torch.no_grad():
            for _ in range(warmup):
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
        median = sorted_t[len(sorted_t) // 2]
        mean = sum(times) / len(times)
        p95 = sorted_t[int(len(sorted_t) * 0.95)]
        min_t = sorted_t[0]
        fps = 1000.0 / median

        t = Table(title=label)
        t.add_column('Metric', style='cyan')
        t.add_column('Value', style='green')
        t.add_row('Median', f'{median:.1f}ms')
        t.add_row('Mean', f'{mean:.1f}ms')
        t.add_row('p95', f'{p95:.1f}ms')
        t.add_row('Min', f'{min_t:.1f}ms')
        t.add_row('FPS', f'{fps:.1f}')
        t.add_row('VRAM (est)', f'{torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
        console.print(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dtype = torch.float16 if args.fp16 else torch.float32

    console.print(f'Device: {device}, FP16: {args.fp16}\n')

    configs = []
    for n_f, n_b in [(8,4), (12,4), (14,4), (16,4), (18,4), (20,4), (22,4), (24,4)]:
        m = DeArtifactNet(n_features=n_f, n_blocks=n_b)
        n_p = sum(p.numel() for p in m.parameters())
        configs.append((f'n_f={n_f}, n_b={n_b}  |  {n_p/1000:.0f}K', m))

    benchmark(configs, device, dtype, args.warmup, args.n_iter)


if __name__ == '__main__':
    main()
