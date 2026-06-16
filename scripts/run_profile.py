"""MambaFixer 完整 profiler — 无需数据集/OpenCV，纯合成数据"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.mamba_fixer import MambaFixer
from models.components import yuv_to_ictcp, ictcp_to_yuv


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", default="4K", choices=["4K", "2K", "1080p", "256"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--nf", type=int, default=2)
    parser.add_argument("--d_state", type=int, default=32)
    parser.add_argument("--n_experts", type=int, default=42)
    parser.add_argument("--dilations", nargs="+", type=int, default=[1, 2, 4, 32])
    parser.add_argument("--channels_last", action="store_true", default=True)
    args = parser.parse_args()

    res_map = {"4K": (2160, 3840), "2K": (1440, 2560), "1080p": (1080, 1920), "256": (256, 256)}
    H, W = res_map[args.res]
    device = torch.device("cuda")
    assert torch.cuda.is_available()
    dtype = torch.float16
    torch.backends.cudnn.benchmark = True

    model = MambaFixer(
        n_features=16, d_state=args.d_state,
        nf_stream=args.nf, n_experts=args.n_experts,
        dilations=args.dilations,
    ).to(device, dtype=dtype)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{'='*60}")
    print(f"  MambaFixer Profiler")
    print(f"  Resolution: {args.res} ({H}×{W})")
    print(f"  Params: {total_params:,}  (activated: ~{total_params // args.n_experts + 6000:,})")
    print(f"  nf={args.nf}, d_state={args.d_state}, dilations={args.dilations}")
    rf_1080p = 1 + 2 * sum(args.dilations)
    print(f"  RF: {rf_1080p}@1080p → {rf_1080p * 2}@4K")
    print(f"{'='*60}")

    x = torch.randn(1, 3, H, W, device=device, dtype=dtype)
    if args.channels_last:
        x = x.to(memory_format=torch.channels_last)

    model.reset_state(1, device)

    # ─── 1. Component breakdown ───
    print(f"\n{'─'*50}")
    print(f"  Component breakdown (avg {args.iters} iters)")
    print(f"{'─'*50}")

    with torch.no_grad():
        # warmup
        for _ in range(5):
            ictcp = yuv_to_ictcp(x)
            z_t = model.forward_ssm_ictcp(ictcp)
            idx, logits = model.router(z_t)
            e = idx[0].item()
            di = model.experts_i[e](ictcp[:, 0:1])
            dct = model.experts_ct[e](ictcp[:, 1:2])
            dcp = model.experts_cp[e](ictcp[:, 2:3])
            cleaned = ictcp + torch.cat([di, dct, dcp], dim=1)
            _ = ictcp_to_yuv(cleaned)
        torch.cuda.synchronize()

        def time_fn(fn, label):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                fn()
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / args.iters * 1000
            return t

        ictcp = yuv_to_ictcp(x)
        z_t = model.forward_ssm_ictcp(ictcp)
        idx, _ = model.router(z_t)
        e = idx[0].item()

        t_color_fwd = time_fn(lambda: yuv_to_ictcp(x), "")
        t_ssm_path = time_fn(lambda: model.forward_ssm_ictcp(ictcp), "")
        t_router_dummy = time_fn(lambda: model.router(z_t), "")
        t_stream_i = time_fn(lambda: model.experts_i[e](ictcp[:, 0:1]), "")
        t_stream_ct = time_fn(lambda: model.experts_ct[e](ictcp[:, 1:2]), "")
        t_stream_cp = time_fn(lambda: model.experts_cp[e](ictcp[:, 2:3]), "")
        t_color_bwd = time_fn(lambda: ictcp_to_yuv(ictcp), "")

        labels = ["yuv→ictcp (JIT)", "SSM path", "MoE Router", "Stream_I",
                  "Stream_Ct", "Stream_Cp", "ictcp→yuv (JIT)"]
        times = [t_color_fwd, t_ssm_path, t_router_dummy, t_stream_i,
                 t_stream_ct, t_stream_cp, t_color_bwd]

        total = sum(times)
        for label, t in zip(labels, times):
            bars = "█" * int(t / total * 50) if total > 0 else ""
            print(f"  {label:20s}  {t:6.2f}ms  {t/total*100:4.0f}%  {bars}")

        # Full forward time
        print(f"\n  {'─'*45}")
        model.reset_state(1, device)
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters // 5):
            model(x)
        torch.cuda.synchronize()
        t_full = (time.perf_counter() - t0) / (args.iters // 5) * 1000

        print(f"  {'Full forward':20s}  {t_full:6.2f}ms  → {1000/t_full:.0f}fps")
        print(f"  {'3050 estimate (50%)':20s}  {t_full*2:6.2f}ms  → {1000/(t_full*2):.0f}fps")

    # ─── 2. Memory ───
    print(f"\n{'─'*50}")
    print(f"  Memory (peak during forward)")
    print(f"{'─'*50}")
    print(f"  Peak alloc:  {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    free, total_mem = torch.cuda.mem_get_info(0)
    print(f"  GPU memory:  {free/1024**2:.0f}/{total_mem/1024**2:.0f} MB free")

    # ─── 3. Expert distribution ───
    print(f"\n{'─'*50}")
    print(f"  MoE expert distribution (simulated)")
    print(f"{'─'*50}")
    with torch.no_grad():
        counts = torch.zeros(args.n_experts)
        z_t = model.forward_ssm_ictcp(ictcp)
        for _ in range(1000):
            x_batch = torch.randn(1, 3, H, W, device=device, dtype=dtype)
            ictcp_batch = yuv_to_ictcp(x_batch)
            z_t = model.forward_ssm_ictcp(ictcp_batch)
            idx, _ = model.router(z_t)
            counts[idx[0].item()] += 1
        counts = counts / counts.sum()
        active = (counts > 0).sum().item()
        uniform_kl = (counts * (counts + 1e-10).log()).sum().item()
        print(f"  Active experts: {active}/{args.n_experts}")
        print(f"  Max usage: {counts.max().item()*100:.1f}%  Min: {counts.min().item()*100:.1f}%")
        # load balancing loss
        bal_loss = model.router.load_balancing_loss(
            torch.zeros(1, args.n_experts, device=device), idx.unsqueeze(0))
        print(f"  Load balancing loss: {bal_loss.item():.4f} (0 = perfect)")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
