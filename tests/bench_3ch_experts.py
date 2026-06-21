import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F


def bench_3ch(H=2160, W=3840, nf=2, d_rates=None, n_warmup=20, n_measure=50):
    if d_rates is None:
        d_rates = [1, 2, 4, 8]
    device = 'cuda'
    N = len(d_rates)

    # — weights: 3 separate (current) —
    kw = dict(device=device, dtype=torch.float16)
    sep = {
        'down_w': [torch.randn(nf, 1, 3, 3, **kw) for _ in range(3)],
        'down_b': [torch.randn(nf, **kw) for _ in range(3)],
        'conv_w': [[torch.randn(nf, nf, 3, 3, **kw) for _ in range(3)] for _ in range(N)],
        'conv_b': [[torch.randn(nf, **kw) for _ in range(3)] for _ in range(N)],
        'prelu_w': [[torch.randn(nf, **kw) for _ in range(3)] for _ in range(N)],
        'up1_w': [torch.randn(nf*4, nf, 3, 3, **kw) for _ in range(3)],
        'up1_b': [torch.randn(nf*4, **kw) for _ in range(3)],
        'up2_w': [torch.randn(1, nf, 3, 3, **kw) for _ in range(3)],
        'up2_b': [torch.randn(1, **kw) for _ in range(3)],
        'dep_w': [torch.randn(1, 1, 3, 3, **kw) for _ in range(3)],
        'dep_b': [torch.randn(1, **kw) for _ in range(3)],
    }

    # — merged (one call, groups=3 to keep each channel independent) —
    G = 3  # one group per color channel
    mer = {
        'down_w': torch.randn(G*nf, 1, 3, 3, **kw),
        'down_b': torch.randn(G*nf, **kw),
        'conv_w': [torch.randn(G*nf, nf, 3, 3, **kw) for _ in range(N)],
        'conv_b': [torch.randn(G*nf, **kw) for _ in range(N)],
        'prelu_w': [torch.randn(G*nf, **kw) for _ in range(N)],
        'up1_w': torch.randn(G*nf*4, nf, 3, 3, **kw),
        'up1_b': torch.randn(G*nf*4, **kw),
        'up2_w': torch.randn(G, nf, 3, 3, **kw),
        'up2_b': torch.randn(G, **kw),
        'dep_w': torch.randn(G, 1, 3, 3, **kw),
        'dep_b': torch.randn(G, **kw),
    }

    x_1ch = torch.randn(1, 1, H, W, **kw)
    x_3ch = torch.randn(1, 3, H, W, **kw)

    def run_separate():
        outs = []
        for c in range(3):
            x = x_1ch
            x = F.conv2d(x, sep['down_w'][c], sep['down_b'][c], stride=2, padding=1)
            skip = x
            for i in range(N):
                x = F.conv2d(x, sep['conv_w'][i][c], sep['conv_b'][i][c],
                             stride=1, padding=d_rates[i], dilation=d_rates[i])
                x = F.prelu(x, sep['prelu_w'][i][c])
            x = x + skip
            x = F.conv2d(x, sep['up1_w'][c], sep['up1_b'][c], padding=1)
            hh, ww = x.shape[-2:]
            x = x.view(1, nf*4, hh, ww)
            x = F.pixel_shuffle(x, 2)
            x = x.view(1, nf, H, W)
            x = F.conv2d(x, sep['up2_w'][c], sep['up2_b'][c], padding=1)
            x_d = F.conv2d(x_1ch, sep['dep_w'][c], sep['dep_b'][c], padding=1)
            outs.append(0.1 * torch.tanh(x + x_d))
        return torch.cat(outs, dim=1)

    def run_merged():
        x = F.conv2d(x_3ch, mer['down_w'], mer['down_b'], stride=2, padding=1, groups=G)
        skip = x
        for i in range(N):
            x = F.conv2d(x, mer['conv_w'][i], mer['conv_b'][i],
                         stride=1, padding=d_rates[i], dilation=d_rates[i], groups=G)
            x = F.prelu(x, mer['prelu_w'][i])
        x = x + skip
        x = F.conv2d(x, mer['up1_w'], mer['up1_b'], padding=1, groups=G)
        hh, ww = x.shape[-2:]
        x = x.view(G, nf*4, hh, ww)
        x = F.pixel_shuffle(x, 2)
        x = x.reshape(1, G*nf, H, W)
        x = F.conv2d(x, mer['up2_w'], mer['up2_b'], padding=1, groups=G)
        x_d = F.conv2d(x_3ch, mer['dep_w'], mer['dep_b'], padding=1, groups=G)
        return 0.1 * torch.tanh(x + x_d)

    # verify shapes match
    with torch.no_grad():
        for _ in range(n_warmup):
            o1 = run_separate()
            o2 = run_merged()
    torch.cuda.synchronize()
    assert o1.shape == o2.shape, f'{o1.shape} vs {o2.shape}'
    print(f'  Output shape: {list(o1.shape)} ✓')

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

    mt_sep = measure(run_separate)
    mt_mer = measure(run_merged)

    print(f'  nf={nf}  d={d_rates}')
    print(f'  {"Method":<22s} {"ms":>8s}  {"fps":>8s}')
    print('  ' + '-' * 40)
    print(f'  {"3× separate (1ch each)":<22s} {mt_sep:>8.3f}  {1000/mt_sep:>8.1f}')
    print(f'  {"1× merged (groups=3)":<22s} {mt_mer:>8.3f}  {1000/mt_mer:>8.1f}')
    print(f'  {"Speedup":<22s} {mt_sep/mt_mer:>8.2f}x')
    print()


if __name__ == '__main__':
    for nf in [1, 2, 3, 4]:
        for dr in [[1], [1, 2, 4, 8]]:
            bench_3ch(nf=nf, d_rates=dr)
