#!/usr/bin/env python3
"""Benchmark strided-conv variant: downsample input 2× at entry, process at 1/2 res, upsample at exit."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from yaml import safe_load
from rich.table import Table

from models.av1_vsr import AV1VSR
from utils.console import console


class StridedAV1VSR(nn.Module):
    def __init__(self, base: AV1VSR, stride: int = 2, n_down: int = 1):
        super().__init__()
        self.stride = stride
        self.n_down = n_down
        down = []
        for _ in range(n_down):
            down.append(nn.Conv2d(3, 3, 3, stride=stride, padding=1))
        self.down = nn.Sequential(*down)
        self.up = nn.Upsample(scale_factor=stride ** n_down, mode='bilinear', align_corners=False)
        self.base = base

    def forward(self, prev, cur, next_, scale=1):
        out = self.base(self.down(prev), self.down(cur), self.down(next_), scale=scale)
        return self.up(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/test.yaml')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    config = safe_load(open(args.config))
    dtype = torch.float16 if args.fp16 else torch.float32

    model_kwargs = dict(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],

    )

    configs_to_test = [
        ('n_f=32 stride=2 (4K→1080p)',  StridedAV1VSR(AV1VSR(**model_kwargs).to(device).eval(), stride=2, n_down=1)),
        ('n_f=32 stride=4 (4K→540p)',   StridedAV1VSR(AV1VSR(**model_kwargs).to(device).eval(), stride=4, n_down=1)),
        ('n_f=16 stride=4 (4K→540p)',   StridedAV1VSR(AV1VSR(**{**model_kwargs, 'n_features': 16}).to(device).eval(), stride=4, n_down=1)),
        ('n_f=32 stride=2+2 (4K→540p)', StridedAV1VSR(AV1VSR(**model_kwargs).to(device).eval(), stride=2, n_down=2)),
    ]

    for i in range(len(configs_to_test)):
        name, model = configs_to_test[i]
        model = model.to(device).eval()
        if args.fp16:
            model = model.half()
        configs_to_test[i] = (name, model)

    console.print(f'Device: {device}, n_features={model_kwargs["n_features"]}, n_blocks={model_kwargs["n_blocks"]}, FP16: {args.fp16}\n')

    scenarios = [
        ('4K → 4K (×1)',  (3840, 2160), 1),
    ]

    for arch_name, model in configs_to_test:
        t = Table(title=f'{arch_name}')
        t.add_column('Scenario', style='cyan')
        t.add_column('Median', style='green')
        t.add_column('Mean', style='green')
        t.add_column('p95', style='yellow')
        t.add_column('Min', style='white')
        t.add_column('FPS', style='magenta')

        for label, (w, h), scale in scenarios:
            inp_w, inp_h = w // scale, h // scale
            prev = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)
            cur  = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)
            next_ = torch.randn(1, 3, inp_h, inp_w, dtype=dtype, device=device)

            with torch.no_grad():
                for _ in range(args.warmup):
                    _ = model(prev, cur, next_, scale=scale)
                torch.cuda.synchronize()

            times = []
            with torch.no_grad():
                for _ in range(args.n_iter):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model(prev, cur, next_, scale=scale)
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


if __name__ == '__main__':
    main()
