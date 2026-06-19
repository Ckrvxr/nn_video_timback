import sys
import time
from pathlib import Path
from yaml import safe_load

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models import MambaFixer
from models.components import yuv_to_ictcp, ictcp_to_yuv
from scripts.tiled_infer import tiled_mamba_inference

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
        routing_threshold=arch_cfg.get('routing_threshold', 0.90)
    ).to(device).half() # Benchmark in FP16 since it is standard for inference
    model.eval()
    
    # Create input frame of 4K resolution
    x = torch.randn(1, 3, 2160, 3840, device=device, dtype=torch.float16)
    
    # Warmup
    print("Warming up...")
    model.reset_state(1, device)
    _ = tiled_mamba_inference(model, x)
    torch.cuda.synchronize()
    
    # Timing function helper
    def time_operation(fn, label, iters=10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters * 1000  # ms
        print(f"  {label:<30s}: {t:8.2f} ms")
        return t

    print("\n--- 4K Detailed Profiling Breakdown ---")
    
    # 1. Color space conversion
    t_yuv2ictcp = time_operation(lambda: yuv_to_ictcp(x), "YUV -> ICtCp JIT")
    ictcp = yuv_to_ictcp(x)
    
    # 2. forward_ssm_ictcp
    t_ssm = time_operation(lambda: model.forward_ssm_ictcp(ictcp), "SSM Path (Mamba)")
    z_t = model.forward_ssm_ictcp(ictcp)
    
    # 3. Spatial stats
    t_spatial = time_operation(lambda: model.spatial_stats(ictcp[:, 0:1]), "Spatial Stats (Conv Tower)")
    z_spatial = model.spatial_stats(ictcp[:, 0:1])
    
    # 4. MoE Router
    router_in = torch.cat([z_t, z_spatial], dim=-1)
    t_router = time_operation(lambda: model.router(router_in, ictcp, k=model.n_active, threshold=model.routing_threshold), "MoE Router (MLP)")
    idx, weights, _ = model.router(router_in, ictcp, k=model.n_active, threshold=model.routing_threshold)
    
    # 5. Apply experts (for a single 1024x1024 tile to measure local overhead)
    tile = ictcp[:, :, 0:1024, 0:1024]
    t_experts_tile = time_operation(lambda: model.apply_experts(tile, idx, weights), "Apply Experts (Single 1024 Tile)")
    
    # 6. Apply experts (for full 4K frame to see what happens without tiling)
    t_experts_full = time_operation(lambda: model.apply_experts(ictcp, idx, weights), "Apply Experts (Full 4K Frame)")
    
    # 7. Color space conversion back
    t_ictcp2yuv = time_operation(lambda: ictcp_to_yuv(ictcp), "ICtCp -> YUV JIT")
    
    # 8. Full tiled inference
    t_full = time_operation(lambda: tiled_mamba_inference(model, x), "Total Tiled Inference (4K)", iters=5)
    print(f"\n  Summary: FPS = {1000.0 / t_full:.2f}")
    
    # 9. Compiled benchmarking
    print("\nCompiling model and color space conversions with torch.compile...")
    try:
        compiled_model = torch.compile(model, mode='reduce-overhead')
        
        # Compile color space functions
        import models.components.color_space as cs
        cs.yuv_to_ictcp = torch.compile(cs.yuv_to_ictcp)
        cs.ictcp_to_yuv = torch.compile(cs.ictcp_to_yuv)
        
        # Warmup compiled run (compilation happens here)
        print("  Running compiled warmup (this takes a minute)...")
        compiled_model.reset_state(1, device)
        _ = tiled_mamba_inference(compiled_model, x)
        torch.cuda.synchronize()
        
        print("\n--- 4K Compiled Inference Performance ---")
        t_compiled = time_operation(lambda: tiled_mamba_inference(compiled_model, x), "Compiled Tiled Inference (4K)", iters=5)
        print(f"  Summary Compiled: FPS = {1000.0 / t_compiled:.2f}")
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == '__main__':
    run_profiler()
