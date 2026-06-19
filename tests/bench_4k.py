import sys
import time
from pathlib import Path
from yaml import safe_load

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models import MambaFixer
from scripts.tiled_infer import tiled_mamba_inference

@torch.no_grad()
def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type != 'cuda':
        print("WARNING: CUDA is not available. Running on CPU will be extremely slow.")
        
    # Read production configuration from configs/mamba.yaml
    config = safe_load(open('configs/mamba.yaml'))
    arch_cfg = config['model_architecture']
    
    num_features = arch_cfg.get('num_features', 64)
    state_dimension = arch_cfg.get('state_dimension', 32)
    num_features_stream = arch_cfg.get('num_features_stream', 2)
    num_experts = arch_cfg.get('num_experts', 100)
    n_active = arch_cfg.get('n_active', 2)
    dilation_rates = arch_cfg.get('dilation_rates', [1, 2, 4, 32])
    routing_threshold = arch_cfg.get('routing_threshold', 0.90)
    
    print("\n--- Production Model Parameters ---")
    print(f"  num_features: {num_features}")
    print(f"  state_dimension: {state_dimension}")
    print(f"  num_features_stream: {num_features_stream}")
    print(f"  num_experts: {num_experts}")
    print(f"  n_active: {n_active}")
    print(f"  dilation_rates: {dilation_rates}")
    print(f"  routing_threshold: {routing_threshold}")
    
    # Test for both FP32 and FP16 (if CUDA)
    precisions = ['FP16', 'FP32'] if device.type == 'cuda' else ['FP32']
    
    for prec in precisions:
        print(f"\nBenchmarking in {prec}...")
        
        # Initialize model
        model = MambaFixer(
            num_features=num_features,
            state_dimension=state_dimension,
            num_features_stream=num_features_stream,
            num_experts=num_experts,
            n_active=n_active,
            dilation_rates=dilation_rates,
            routing_threshold=routing_threshold
        ).to(device)
        model.eval()
        
        dtype = torch.float16 if prec == 'FP16' else torch.float32
        if prec == 'FP16':
            model = model.half()
            
        # Create simulated 4K input tensor [B, C, H, W] = [1, 3, 2160, 3840]
        # In YUV format (we will simulate values between -1 and 1)
        x = torch.randn(1, 3, 2160, 3840, device=device, dtype=dtype)
        
        # Reset state
        model.reset_state(1, device)
        
        # Warmup runs
        print("  Warming up (3 runs)...")
        for _ in range(3):
            _ = tiled_mamba_inference(model, x, tile_size=1024, overlap=64)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Benchmark runs
        runs = 5
        print(f"  Measuring latency over {runs} runs...")
        t_start = time.perf_counter()
        for _ in range(runs):
            model.reset_state(1, device)
            _ = tiled_mamba_inference(model, x, tile_size=1024, overlap=64)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        
        avg_latency = (t_end - t_start) / runs * 1000  # ms
        fps = 1000.0 / avg_latency
        
        print(f"  => Average Latency: {avg_latency:.2f} ms")
        print(f"  => FPS: {fps:.2f} frames/sec")

if __name__ == '__main__':
    run_benchmark()
