"""Standalone MambaFixer benchmark — no dataset/OpenCV needed."""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from models.mamba_fixer import MambaFixer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", action="store_true", help="Use fp16")
    args = parser.parse_args()
    dtype = torch.float16 if args.half else torch.float32

    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA required"

    model = MambaFixer(n_features=16, d_state=64).to(device)
    if args.half:
        model = model.half()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}  dtype: {dtype}")

    torch.backends.cudnn.benchmark = True

    # ─── 1. Single tile latency (256×256) ───
    x = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()

        # SSM alone
        model.reset_state(1, device)
        t0 = time.perf_counter()
        for _ in range(500):
            _, _ = model.forward_ssm(x)
        torch.cuda.synchronize()
        t_ssm = (time.perf_counter() - t0) / 500 * 1000

        # ChannelNet alone
        g, b = model.forward_ssm(x)
        t0 = time.perf_counter()
        for _ in range(500):
            model.forward_tiles(x, g, b)
        torch.cuda.synchronize()
        t_cn = (time.perf_counter() - t0) / 500 * 1000

        # Full single tile
        model.reset_state(1, device)
        t0 = time.perf_counter()
        for _ in range(500):
            model(x)
        torch.cuda.synchronize()
        t_full = (time.perf_counter() - t0) / 500 * 1000

    print(f"\n─── 256×256 single tile ───")
    print(f"  SSM alone:         {t_ssm:.3f} ms")
    print(f"  ChannelNet alone:  {t_cn:.3f} ms")
    print(f"  Full forward:      {t_full:.3f} ms")

    # ─── 2. Batch tile latency ───
    for bs in [8, 32, 72, 144]:
        tiles = torch.randn(bs, 3, 256, 256, device=device, dtype=dtype)
        model.reset_state(1, device)
        g, b = model.forward_ssm(x)
        gx = g.expand(bs, -1)
        bx = b.expand(bs, -1)
        with torch.no_grad():
            for _ in range(30):
                model.forward_tiles(tiles, gx, bx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                model.forward_tiles(tiles, gx, bx)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / 200 * 1000
        print(f"\n─── Batch {bs:3d} tiles ───")
        print(f"  ChannelNet: {t:.3f} ms | per tile: {t/bs*1000:.1f} µs")

    # ─── 3. ChannelNet conv-only (bypass colorspace) ───
    ictcp = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)
    i_ch = ictcp[:, 0:1]
    ct = ictcp[:, 1:2]
    cp = ictcp[:, 2:3]
    with torch.no_grad():
        for _ in range(30):
            model.net_i(i_ch, g, b)
            model.net_ct(ct, g, b)
            model.net_cp(cp, g, b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            model.net_i(i_ch, g, b)
            model.net_ct(ct, g, b)
            model.net_cp(cp, g, b)
        torch.cuda.synchronize()
        t_conv = (time.perf_counter() - t0) / 500 * 1000
    print(f"\n─── Conv only (3×ChannelNet, ICtCp input) ───")
    print(f"  Total: {t_conv:.3f} ms")

    # ─── 4. 4K tiled estimate ───
    from scripts.tiled_infer import tiled_infer
    frame = torch.randn(1, 3, 2160, 3840, device=device, dtype=dtype)
    with torch.no_grad():
        tiled_infer(model, frame)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = tiled_infer(model, frame)
        torch.cuda.synchronize()
        t_4k = (time.perf_counter() - t0) * 1000
    print(f"\n─── 4K (2160×3840) tiled ───")
    print(f"  Total: {t_4k:.1f} ms | within budget: {'YES' if t_4k < 16.7 else 'NO'}")
    print(f"  Over budget by: {t_4k/16.7:.1f}×")

    # ─── 5. Memory ───
    print(f"\n─── Memory ───")
    print(f"  Peak: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    for i in range(torch.cuda.device_count()):
        free, total_mem = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {free/1024**2:.0f}/{total_mem/1024**2:.0f} MB free")


if __name__ == "__main__":
    main()
