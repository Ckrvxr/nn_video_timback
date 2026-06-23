import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from yaml import safe_load

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


def profile(num_features_stream: int, H: int, W: int, device: torch.device,
            n_warmup: int = 10, n_measure: int = 50):
    model = Timback(
        num_features=64,
        state_dimension=32,
        num_features_stream=num_features_stream,
        num_experts=100,
        n_active=4,
        dilation_rates=[1, 2, 4, 8],
        routing_threshold=0.90,
    ).to(device)
    model.eval()
    model = model.half()
    for p in model.parameters():
        if p.dtype == torch.float32 and p.is_floating_point():
            p.data = p.data.half()

    param_count = sum(p.numel() for p in model.parameters())

    target = ['experts']
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

    expert_ms = 0.0
    for key in target:
        p = profilers[key]
        avg_ms = sum(p.times) / len(p.times)
        expert_ms += avg_ms

    overhead = total_ms - expert_ms

    print(f'  nf={num_features_stream} | {W}×{H} | params={param_count:,}')
    print(f'  Total:  {total_ms:.3f} ms   ({fps:.1f} fps)')
    print(f'  Expert: {expert_ms:.3f} ms  ({expert_ms/total_ms*100:.1f}%)')
    print(f'  Overhd: {overhead:.3f} ms')
    print(f'  VRAM:   {peak_mem_mb:.0f} MB')
    print()

    return total_ms, expert_ms, fps, peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nf', type=int, default=2, choices=[2, 4])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('CUDA not available')
        return

    H, W = 2160, 3840
    profile(args.nf, H, W, device)


if __name__ == '__main__':
    main()
