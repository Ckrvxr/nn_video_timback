import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from yaml import safe_load

from models import MambaFixer


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


def profile(config_path: str, H: int, W: int, device: torch.device,
            n_warmup: int = 50, n_measure: int = 100):
    config = safe_load(open(config_path))
    arch_cfg = config['model_architecture']

    model = MambaFixer(
        num_features=arch_cfg['num_features'],
        state_dimension=arch_cfg['state_dimension'],
        num_features_stream=arch_cfg['num_features_stream'],
        num_experts=arch_cfg['num_experts'],
        n_active=arch_cfg['n_active'],
        dilation_rates=arch_cfg['dilation_rates'],
        routing_threshold=arch_cfg.get('routing_threshold', 0.9),
    ).to(device)
    model.eval()
    model = model.half()
    for p in model.parameters():
        if p.dtype == torch.float32 and p.is_floating_point():
            p.data = p.data.half()

    target = {'downsample', 'ssm_fwd', 'ssm_bwd', 'ssm_proj', 't_ssm',
              'spatial_stats', 'router', 'experts'}
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

    # re-register with fresh events for measured pass
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
    peak_mem_mb = (torch.cuda.max_memory_allocated() - mem_before) / 1024**2

    for h in handles:
        h.remove()

    display_name = {
        'downsample': 'DownsampleChain',
        'ssm_fwd': 'SSM fwd',
        'ssm_bwd': 'SSM bwd',
        'ssm_proj': 'SSM proj',
        't_ssm': 't_SSM',
        'spatial_stats': 'SpatialStats',
        'router': 'Router',
        'experts': 'Experts(3ch)',
    }

    print(f'\n## {device} | {W}×{H}\n')
    header = f'{"Module":<20s} {"Time(ms)":>10s} {"%total":>8s}'
    print(header)
    print('─' * len(header))

    acc = 0.0
    for key in target:
        p = profilers[key]
        avg_ms = sum(p.times) / len(p.times)
        acc += avg_ms
    for key in target:
        p = profilers[key]
        avg_ms = sum(p.times) / len(p.times)
        pct = avg_ms / acc * 100
        print(f'{display_name[key]:<20s} {avg_ms:>8.3f}  {pct:>6.1f}%')

    print('─' * len(header))
    print(f'{"Sum (hooks)":<20s} {acc:>8.3f}  {100.0:>6.1f}%')
    print(f'{"Total (measured)":<20s} {total_ms:>8.3f}')
    print(f'{"Peak VRAM":<20s} {peak_mem_mb:>8.1f} MB')
    print()

    # diff between hook sum and total = overhead of data movement / concat / etc
    overhead = total_ms - acc
    if overhead > 0:
        print(f'(非模块开销: data prep + glue = {overhead:.3f}ms / {overhead/total_ms*100:.1f}%)')


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='configs/mamba.yaml')
    args.add_argument('--device', default='cuda')
    args = args.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sizes = [(720, 1280), (1080, 1920)]
    for H, W in sizes:
        profile(args.config, H, W, device)


if __name__ == '__main__':
    main()
