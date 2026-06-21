"""Profile MambaFixer inference pipeline: stage decomposition, 4K tiled inference, expert distribution."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from yaml import safe_load
from models import MambaFixer
from models.components import yuv_to_ictcp, ictcp_to_yuv
from utils.inference.tiling import tiled_mamba_inference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_profile_infer_forward_smoke():
    config = safe_load(open('configs/mamba.yaml'))
    arch_cfg = config['model_architecture']
    model = MambaFixer(
        num_features=arch_cfg.get('num_features', 64),
        state_dimension=arch_cfg.get('state_dimension', 32),
        num_features_stream=arch_cfg.get('num_features_stream', 2),
        num_experts=arch_cfg.get('num_experts', 100),
        n_active=arch_cfg.get('n_active', 2),
        dilation_rates=arch_cfg.get('dilation_rates', [1, 2, 4, 32]),
        routing_threshold=arch_cfg.get('routing_threshold', 0.90),
    ).to('cuda').half()
    model.eval()
    x = torch.randn(1, 3, 128, 128, device='cuda', dtype=torch.float16)
    model.reset_state(1, 'cuda')
    y = tiled_mamba_inference(model, x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


@torch.no_grad()
def run_profiler():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type != 'cuda':
        print("ERROR: CUDA is required for profiling GPU kernels.")
        return

    config = safe_load(open('configs/mamba.yaml'))
    arch_cfg = config['model_architecture']

    model = MambaFixer(
        num_features=arch_cfg.get('num_features', 64),
        state_dimension=arch_cfg.get('state_dimension', 32),
        num_features_stream=arch_cfg.get('num_features_stream', 2),
        num_experts=arch_cfg.get('num_experts', 100),
        n_active=arch_cfg.get('n_active', 2),
        dilation_rates=arch_cfg.get('dilation_rates', [1, 2, 4, 32]),
        routing_threshold=arch_cfg.get('routing_threshold', 0.90),
    ).to(device).half()
    model.eval()

    x = torch.randn(1, 3, 2160, 3840, device=device, dtype=torch.float16)

    print("Warming up...")
    model.reset_state(1, device)
    _ = tiled_mamba_inference(model, x)
    torch.cuda.synchronize()

    def time_op(fn, label, iters=10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:<30s}: {t:8.2f} ms")
        return t

    # ─── 1. Component breakdown ────────────────────────────────
    print("\n--- 4K Detailed Profiling Breakdown ---")

    t_yuv2ictcp = time_op(lambda: yuv_to_ictcp(x), "YUV -> ICtCp JIT")
    ictcp = yuv_to_ictcp(x)

    t_ssm = time_op(lambda: model.forward_ssm_ictcp(ictcp), "SSM Path (Mamba)")
    z_t = model.forward_ssm_ictcp(ictcp)

    t_spatial = time_op(
        lambda: model.spatial_stats(ictcp[:, 0:1]),
        "Spatial Stats (Conv Tower)",
    )
    z_spatial = model.spatial_stats(ictcp[:, 0:1])

    router_in = torch.cat([z_t, z_spatial], dim=-1)
    t_router = time_op(
        lambda: model.router(router_in, ictcp, k=model.n_active, threshold=model.routing_threshold),
        "MoE Router (MLP)",
    )
    idx, weights, _ = model.router(router_in, ictcp, k=model.n_active, threshold=model.routing_threshold)

    tile = ictcp[:, :, 0:1024, 0:1024]
    t_experts_tile = time_op(
        lambda: model.apply_experts(tile, idx, weights),
        "Apply Experts (Single 1024 Tile)",
    )

    t_experts_full = time_op(
        lambda: model.apply_experts(ictcp, idx, weights),
        "Apply Experts (Full 4K Frame)",
    )

    t_ictcp2yuv = time_op(lambda: ictcp_to_yuv(ictcp), "ICtCp -> YUV JIT")

    t_full = time_op(
        lambda: tiled_mamba_inference(model, x),
        "Total Tiled Inference (4K)",
        iters=5,
    )
    print(f"\n  Summary: FPS = {1000.0 / t_full:.2f}")

    # ─── 2. Memory ─────────────────────────────────────────────
    print(f"\n--- Memory (peak during forward) ---")
    print(f"  Peak alloc:  {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    free, total_mem = torch.cuda.mem_get_info(0)
    print(f"  GPU memory:  {free/1024**2:.0f}/{total_mem/1024**2:.0f} MB free")

    # ─── 3. Expert distribution ────────────────────────────────
    print(f"\n--- MoE Expert Distribution ---")
    counts = torch.zeros(arch_cfg.get('num_experts', 100))
    z_t = model.forward_ssm_ictcp(ictcp)
    for _ in range(1000):
        x_batch = torch.randn(1, 3, 2160, 3840, device=device, dtype=torch.float16)
        ictcp_batch = yuv_to_ictcp(x_batch)
        z_t = model.forward_ssm_ictcp(ictcp_batch)
        idx, _ = model.router(z_t)
        counts[idx[0].item()] += 1
    counts = counts / counts.sum()
    active = (counts > 0).sum().item()
    uniform_kl = (counts * (counts + 1e-10).log()).sum().item()
    print(f"  Active experts: {active}/{arch_cfg.get('num_experts', 100)}")
    print(f"  Max usage: {counts.max().item()*100:.1f}%  Min: {counts.min().item()*100:.1f}%")

    # ─── 4. torch.compile ──────────────────────────────────────
    print("\nCompiling model with torch.compile...")
    try:
        compiled_model = torch.compile(model, mode='reduce-overhead')
        import models.components.color_space as cs
        cs.yuv_to_ictcp = torch.compile(cs.yuv_to_ictcp)
        cs.ictcp_to_yuv = torch.compile(cs.ictcp_to_yuv)

        print("  Running compiled warmup...")
        compiled_model.reset_state(1, device)
        _ = tiled_mamba_inference(compiled_model, x)
        torch.cuda.synchronize()

        print("\n--- 4K Compiled Inference Performance ---")
        t_compiled = time_op(
            lambda: tiled_mamba_inference(compiled_model, x),
            "Compiled Tiled Inference (4K)",
            iters=5,
        )
        print(f"  Summary Compiled: FPS = {1000.0 / t_compiled:.2f}")
    except Exception as e:
        print(f"Compilation failed: {e}")


if __name__ == '__main__':
    run_profiler()
