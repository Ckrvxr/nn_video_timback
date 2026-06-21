import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import time

from models.mamba_fixer import MambaFixer

CONFIGS = [
    ("Lite",     32, 16),
    ("Balanced", 64, 32),
    ("Wide",     32, 64),
    ("High-96_48", 96, 48),
    ("High-96_64", 96, 64),
    ("High-128_64", 128, 64),
]

N_ITERS = 100
WARMUP  = 20

results = []
for name, nf, sd in CONFIGS:
    print(f"\n--- {name} (nf={nf}, sd={sd}) ---")
    m = MambaFixer(
        num_features=nf, state_dimension=sd,
        num_features_stream=2, num_experts=100, n_active=2,
        dilation_rates=[1, 2, 4, 8], routing_threshold=0.9,
    )
    params = sum(p.numel() for p in m.parameters())
    print(f"  Params: {params:,}")

    m = m.cuda()
    for p in m.parameters():
        p.data = p.data.to(dtype=torch.float16, device="cuda")
    for b in m.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype=torch.float16, device="cuda")
        else:
            b.data = b.data.to(device="cuda")

    def bench(H, W):
        x = torch.randn(1, 3, H, W, dtype=torch.float16, device="cuda")
        m.reset_state(1, "cuda")
        for _ in range(WARMUP):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            m(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N_ITERS
        return dt * 1000, 1.0 / dt

    ms_720, fps_720 = bench(720, 1280)
    print(f"  720p: {ms_720:.2f}ms {fps_720:.0f}fps")

    ms_1080, fps_1080 = bench(1080, 1920)
    print(f"  1080p: {ms_1080:.2f}ms {fps_1080:.0f}fps")

    results.append((name, nf, sd, params, ms_720, fps_720, ms_1080, fps_1080))

print("\n\n=== Summary ===")
print(f"{'Name':<10} {'nf':>4} {'sd':>4} {'Params':>8} {'720p ms':>8} {'720p fps':>8} {'1080p ms':>9} {'1080p fps':>9}")
print("-" * 70)
for r in results:
    print(f"{r[0]:<10} {r[1]:>4} {r[2]:>4} {r[3]:>8,} {r[4]:>8.2f} {r[5]:>8.0f} {r[6]:>9.2f} {r[7]:>9.0f}")
