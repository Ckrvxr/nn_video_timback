import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from core import ArtRT


def count_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.inference_mode()
def benchmark_infer(model, x, label, warmup=5, runs=30):
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(runs):
        y = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    avg_ms = (end - start) / runs * 1000
    mem_used = (peak - mem_before) / 1024 ** 2
    print(f"  {label:28s}  {avg_ms:>8.2f} ms  {mem_used:>8.2f} MB  {list(y.shape)}")


@torch.inference_mode()
def benchmark_compiled(model, x, label, warmup=5, runs=30):
    model.eval()
    compiled = torch.compile(model, mode='default')
    for _ in range(warmup):
        _ = compiled(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(runs):
        y = compiled(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    avg_ms = (end - start) / runs * 1000
    mem_used = (peak - mem_before) / 1024 ** 2
    print(f"  {label:28s}  {avg_ms:>8.2f} ms  {mem_used:>8.2f} MB  {list(y.shape)}")


def benchmark_train(model, x, label, warmup=3, runs=15):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for _ in range(warmup):
        y = model(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(runs):
        y = model(x)
        loss = y.mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    avg_ms = (end - start) / runs * 1000
    mem_used = (peak - mem_before) / 1024 ** 2
    print(f"  {label:28s}  {avg_ms:>8.2f} ms  {mem_used:>8.2f} MB  {list(y.shape)}")


def main():
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    dtype = torch.float16
    sizes = [(1, 3, 512, 512), (1, 3, 1024, 1024), (4, 3, 512, 512),
             (1, 3, 1920, 1080), (1, 3, 3840, 2160)]

    m = ArtRT().to(device, dtype=dtype)
    print(f'ArtRT — params: {count_params(m):,}\n')

    for mode_name, fn, w, r in [
        ('Inference (eval, no grad)', benchmark_infer, 5, 30),
        ('Compiled (torch.compile)', benchmark_compiled, 5, 30),
        ('Train (forward+backward)', benchmark_train, 3, 15),
    ]:
        print(f'  === {mode_name} ===')
        h = f"  {'Input':28s}  {'Time':>8s}  {'Memory':>8s}"
        print(h)
        print('  ' + '-' * (len(h) - 2))
        for size in sizes:
            B, C, H, W = size
            x = torch.randn(B, C, H, W, device=device, dtype=dtype)
            fn(m, x, f'{B}x{C}x{H}x{W}')
        print()


if __name__ == '__main__':
    main()
