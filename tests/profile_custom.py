import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from components import Timback


class ProfilingHook:
    def __init__(self):
        self.times = []
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def pre_hook(self, module, args):
        self.start.record()

    def post_hook(self, module, args, output):
        self.end.record()
        torch.cuda.synchronize()
        self.times.append(self.start.elapsed_time(self.end))


def profile(num_experts: int, n_active: int, nf: int,
            H: int, W: int, device: torch.device,
            dilation_rates: list[int] | None = None,
            n_warmup: int = 10, n_measure: int = 50):
    if dilation_rates is None:
        dilation_rates = [1, 2, 4, 8]
    model = Timback(
        num_features=64,
        state_dimension=32,
        num_features_stream=nf,
        num_experts=num_experts,
        n_active=n_active,
        dilation_rates=dilation_rates,
        routing_threshold=1.0,
    ).to(device)
    model.eval()
    model = model.half()
    for p in model.parameters():
        if p.dtype == torch.float32 and p.is_floating_point():
            p.data = p.data.half()

    param_count = sum(p.numel() for p in model.parameters())
    expert_only = sum(p.numel() for n, p in model.named_parameters()
                      if 'experts' in n)
    non_expert = param_count - expert_only

    target = ['downsample', 't_ssm', 'fusion', 'router', 'experts']
    profilers = {}

    handles = []
    for name, mod in model.named_modules():
        if name in target:
            p = ProfilingHook()
            profilers[name] = p
            handles.append(mod.register_forward_pre_hook(p.pre_hook))
            handles.append(mod.register_forward_hook(p.post_hook))

    x = torch.randn(1, 3, H, W, dtype=torch.float16, device=device)

    model.reset_state(1, device)
    for _ in range(n_warmup):
        model(x)
    for h in handles:
        h.remove()

    profilers = {}
    handles = []
    for name, mod in model.named_modules():
        if name in target:
            p = ProfilingHook()
            profilers[name] = p
            handles.append(mod.register_forward_pre_hook(p.pre_hook))
            handles.append(mod.register_forward_hook(p.post_hook))

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    total_start.record()
    model.reset_state(1, device)
    for _ in range(n_measure):
        model(x)
    total_end.record()
    torch.cuda.synchronize()

    total_ms = total_start.elapsed_time(total_end) / n_measure
    fps = 1000.0 / total_ms
    peak_mem_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1024**2

    for h in handles:
        h.remove()

    module_times = {}
    for key in target:
        p = profilers[key]
        avg_ms = sum(p.times) / len(p.times)
        module_times[key] = avg_ms

    experts_time = sum(module_times[k] for k in ['experts'])
    non_expert_time = sum(module_times[k] for k in target if k not in ['experts'])
    measured_sum = sum(module_times.values())
    overhead = total_ms - measured_sum

    print(f'  专家={num_experts} 激活={n_active} nf={nf} | {W}×{H}')
    print(f'  参数: total={param_count:,} (专家={expert_only:,} 非专家={non_expert:,})')
    print(f'  Total:  {total_ms:.3f} ms  ({fps:.2f} fps)')
    print(f'  VRAM:   {peak_mem_mb:.0f} MB')
    print()
    print(f'  Module breakdown (ms):')

    expert_keys = ['experts']
    non_expert_keys = ['downsample', 't_ssm', 'fusion', 'router']

    for key in non_expert_keys + expert_keys:
        t = module_times.get(key, 0)
        pct = t / total_ms * 100
        marker = '  ◆ ' if key in expert_keys else '    '
        print(f'  {marker}{key:<14s} {t:>8.3f} ({pct:>4.1f}%)')

    print(f'  {"":->28s}')
    print(f'  {"Sum measured":<22s} {measured_sum:>8.3f} ({measured_sum/total_ms*100:.1f}%)')
    print(f'  {"Overhead(glue)":<22s} {overhead:>8.3f} ({overhead/total_ms*100:.1f}%)')
    print()

    return total_ms, experts_time, fps, peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experts', type=int, default=42)
    parser.add_argument('--active', type=int, default=1)
    parser.add_argument('--nf', type=int, default=8)
    parser.add_argument('--dilations', type=str, default=None,
                        help='Comma-separated dilations, e.g. "1,2,4,8"')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.dilations is None:
        dilations = [1, 2, 4, 8]
    elif args.dilations.strip().lower() == 'none':
        dilations = []
    else:
        dilations = [int(x) for x in args.dilations.split(',')]
        if not dilations:
            dilations = [1, 2, 4, 8]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('CUDA not available')
        return

    H, W = 2160, 3840
    profile(args.experts, args.active, args.nf, H, W, device, dilation_rates=dilations)


if __name__ == '__main__':
    main()
