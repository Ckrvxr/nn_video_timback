import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F


def bench_kernel(H=2160, W=3840, nf=2, d_rates=None, out_ch=1, n_warmup=20, n_measure=50):
    if d_rates is None:
        d_rates = [1]
    device = 'cuda'
    kw = dict(device=device, dtype=torch.float16)

    def make_chain(ks):
        k = ks
        pad = k // 2
        down_w = torch.randn(nf, out_ch, k, k, **kw)
        down_b = torch.randn(nf, **kw)
        conv_w = [torch.randn(nf, nf, k, k, **kw) for _ in d_rates]
        conv_b = [torch.randn(nf, **kw) for _ in d_rates]
        prelu_w = [torch.randn(nf, **kw) for _ in d_rates]
        up1_w = torch.randn(nf * 4, nf, k, k, **kw)
        up1_b = torch.randn(nf * 4, **kw)
        up2_w = torch.randn(1, nf, k, k, **kw)
        up2_b = torch.randn(1, **kw)
        dep_w = torch.randn(1, 1, k, k, **kw)
        dep_b = torch.randn(1, **kw)
        return down_w, down_b, conv_w, conv_b, prelu_w, up1_w, up1_b, up2_w, up2_b, dep_w, dep_b

    c3 = make_chain(3)
    c5 = make_chain(5)

    def run_chain(down_w, down_b, conv_w, conv_b, prelu_w, up1_w, up1_b, up2_w, up2_b, dep_w, dep_b):
        ks = down_w.shape[-1]
        pd = ks // 2
        x = F.conv2d(x_in, down_w, down_b, stride=2, padding=pd)
        skip = x
        for i, d in enumerate(d_rates):
            pad = d * (ks - 1) // 2
            x = F.conv2d(x, conv_w[i], conv_b[i], stride=1, padding=pad, dilation=d)
            x = F.prelu(x, prelu_w[i])
        x = x + skip
        hh, ww = x.shape[-2:]
        x = F.conv2d(x, up1_w, up1_b, stride=1, padding=pd)
        x = x.view(1, nf * 4, hh, ww)
        x = F.pixel_shuffle(x, 2)
        x = x.view(1, nf, H, W)
        x = F.conv2d(x, up2_w, up2_b, stride=1, padding=pd)
        x_d = F.conv2d(x_in, dep_w, dep_b, stride=1, padding=pd)
        return 0.1 * torch.tanh(x + x_d)

    x_in = torch.randn(1, 1, H, W, **kw)

    for _ in range(n_warmup):
        run_chain(*c3)
        run_chain(*c5)
    torch.cuda.synchronize()

    evt_s, evt_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def measure(fn):
        ts = []
        for _ in range(n_measure):
            evt_s.record()
            fn()
            evt_e.record()
            torch.cuda.synchronize()
            ts.append(evt_s.elapsed_time(evt_e))
        return sum(ts) / len(ts)

    t3 = measure(lambda: run_chain(*c3))
    t5 = measure(lambda: run_chain(*c5))

    print(f'  nf={nf}  d={d_rates}')
    print(f'  {"Kernel":<18s} {"ms":>8s}  {"fps":>8s}')
    print('  ' + '-' * 36)
    print(f'  {"3×3":<18s} {t3:>8.3f}  {1000/t3:>8.1f}')
    print(f'  {"5×5":<18s} {t5:>8.3f}  {1000/t5:>8.1f}')
    print(f'  {"Ratio (5×5/3×3)":<18s} {t5/t3:>8.2f}x')
    print(f'  {"| ×3 experts":<18s} {t5*3/t3*3:>8.2f}x')
    print()


if __name__ == '__main__':
    for nf in [1, 2]:
        for dr in [[1], [2, 3], [1, 4], [1, 2, 4, 8], [5]]:
            bench_kernel(nf=nf, d_rates=dr)
