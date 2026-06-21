import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F


def bench_colormix(H=2160, W=3840, n_warmup=20, n_measure=100):
    device = 'cuda'
    kw = dict(device=device, dtype=torch.float16)
    C = 3
    delta = torch.randn(1, C, H, W, **kw)
    W_cm = torch.randn(C, C, 1, 1, **kw)
    b_cm = torch.randn(C, **kw)

    def via_conv2d():
        return 0.1 * torch.tanh(F.conv2d(delta, W_cm, b_cm))

    def via_matmul():
        # 1×1 conv = matmul on spatial dims
        xf = delta.flatten(2).transpose(1, 2)          # [B, H*W, C]
        Wo = W_cm.squeeze(-1).squeeze(-1).T             # [C, C]
        xf = F.linear(xf, Wo, b_cm)                     # [B, H*W, C]
        xf = xf.transpose(1, 2).view_as(delta)          # [B, C, H, W]
        return 0.1 * torch.tanh(xf)

    def via_bmm():
        # batched matmul (H*W → batch dim approach)
        Wo = W_cm.squeeze(-1).squeeze(-1)               # [C_out, C_in] — Woo, conv weight is [C_out, C_in, 1, 1]
        # Actually conv2d weight: [out_ch, in_ch, 1, 1] = [C, C, 1, 1]
        # For linear: in_features=C, out_features=C
        # F.linear(input, weight.T, bias) where weight is [out, in]
        # So we need: W_linear = W_cm.squeeze() = [C, C] (out, in)
        # Or equivalently: delta.reshape(H*W, C) @ W_linear.T + bias → [H*W, C]
        return via_matmul()  # same thing

    evt_s, evt_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    for _ in range(n_warmup):
        _ = via_conv2d()
        _ = via_matmul()
    torch.cuda.synchronize()

    def measure(fn):
        ts = []
        for _ in range(n_measure):
            evt_s.record()
            fn()
            evt_e.record()
            torch.cuda.synchronize()
            ts.append(evt_s.elapsed_time(evt_e))
        return sum(ts) / len(ts)

    t_conv = measure(via_conv2d)
    t_matm = measure(via_matmul)

    print(f'  {"W×H":<22s} {W}×{H}')
    print(f'  {"Method":<22s} {"ms":>8s}')
    print('  ' + '-' * 32)
    print(f'  {"F.conv2d (1×1)":<22s} {t_conv:>8.3f}')
    print(f'  {"F.linear (matmul)":<22s} {t_matm:>8.3f}')
    print(f'  {"Speedup":<22s} {t_conv/t_matm:>8.2f}x')

    # Also check: what if we remove tanh?
    def via_conv_no_tanh():
        return F.conv2d(delta, W_cm, b_cm)

    def via_lin_no_tanh():
        xf = delta.flatten(2).transpose(1, 2)
        Wo = W_cm.squeeze(-1).squeeze(-1).T
        xf = F.linear(xf, Wo, b_cm)
        return xf.transpose(1, 2).view_as(delta)

    for _ in range(n_warmup):
        _ = via_conv_no_tanh()
        _ = via_lin_no_tanh()
    torch.cuda.synchronize()

    t_conv_nt = measure(via_conv_no_tanh)
    t_lin_nt = measure(via_lin_no_tanh)

    print(f'  {"F.conv2d (no tanh)":<22s} {t_conv_nt:>8.3f}')
    print(f'  {"F.linear (no tanh)":<22s} {t_lin_nt:>8.3f}')
    print(f'  {"Speedup (no tanh)":<22s} {t_conv_nt/t_lin_nt:>8.2f}x')


if __name__ == '__main__':
    bench_colormix()
