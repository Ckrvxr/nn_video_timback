import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def fmt_ms(ms: float) -> str:
    return f'{ms:.4f}'


def bench_conv(name: str, B: int, k: int, nf: int,
               in_ch_per_group: int, out_ch_per_group: int,
               H: int, W: int, stride: int = 1, dilation: int = 1,
               n_warmup: int = 20, n_measure: int = 200):
    """Benchmark grouped vs standard conv with real expert shapes.
    
    Current (grouped): F.conv2d(..., groups=B*k) — 1 call
    Standard: B*k individual F.conv2d calls — B*k calls
    """
    G = B * k  # number of groups = active experts
    total_in = G * in_ch_per_group
    total_out = G * out_ch_per_group

    x = torch.randn(1, total_in, H, W, dtype=torch.float16, device='cuda')

    # weight: [total_out, in_ch_per_group, 3, 3] for grouped conv
    w_group = torch.randn(total_out, in_ch_per_group, 3, 3, dtype=torch.float16, device='cuda')
    b_group = torch.randn(total_out, dtype=torch.float16, device='cuda')

    # standard conv weights: list of [out_ch_per_group, in_ch_per_group, 3, 3] per group
    w_std_list = [w_group[i*out_ch_per_group:(i+1)*out_ch_per_group] for i in range(G)]
    b_std_list = [b_group[i*out_ch_per_group:(i+1)*out_ch_per_group] for i in range(G)]

    pad = dilation

    # ---- grouped conv (current) ----
    for _ in range(n_warmup):
        _ = torch.nn.functional.conv2d(x, w_group, b_group,
                                        stride=stride, padding=pad,
                                        dilation=dilation, groups=G)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_measure):
        _ = torch.nn.functional.conv2d(x, w_group, b_group,
                                        stride=stride, padding=pad,
                                        dilation=dilation, groups=G)
    end.record()
    torch.cuda.synchronize()
    group_ms = start.elapsed_time(end) / n_measure

    # ---- standard conv (individual calls) ----
    for _ in range(n_warmup):
        outs = []
        for i in range(G):
            xi = x[:, i*in_ch_per_group:(i+1)*in_ch_per_group]
            o = torch.nn.functional.conv2d(xi, w_std_list[i], b_std_list[i],
                                            stride=stride, padding=pad,
                                            dilation=dilation)
            outs.append(o)
        _ = torch.cat(outs, dim=1)

    start.record()
    for _ in range(n_measure):
        outs = []
        for i in range(G):
            xi = x[:, i*in_ch_per_group:(i+1)*in_ch_per_group]
            o = torch.nn.functional.conv2d(xi, w_std_list[i], b_std_list[i],
                                            stride=stride, padding=pad,
                                            dilation=dilation)
            outs.append(o)
        _ = torch.cat(outs, dim=1)
    end.record()
    torch.cuda.synchronize()
    std_ms = start.elapsed_time(end) / n_measure

    if group_ms < std_ms:
        faster = 'group'
        speedup = std_ms / group_ms
    else:
        faster = 'std'
        speedup = group_ms / std_ms

    return {
        'layer': name,
        'groups': G,
        'in_per': in_ch_per_group,
        'out_per': out_ch_per_group,
        'total_in': total_in,
        'total_out': total_out,
        'group_ms': group_ms,
        'std_ms': std_ms,
        'faster': faster,
        'speedup': speedup,
    }


def main():
    if not torch.cuda.is_available():
        print('CUDA not available')
        return

    H, W = 2160, 3840
    B, k, nf = 1, 4, 2

    print(f'## Conv microbenchmark | 4K ({W}×{H}) | B={B} k={k} nf={nf}\n')
    print(f'Conv 3×3, groups=B*k={B*k}, 200 iters after 20 warmup\n')

    # Layer configs matching actual ParallelExperts shapes
    # (name, B, k, nf, in_ch_per_group, out_ch_per_group, H, W, stride, dilation)
    cases = [
        # down: 1 in → nf out, stride 2
        ('down s=2',   B, k, nf, 1,           nf,  H, W,    2, 1),
        # dilated: nf → nf, varies dilation
        ('dilate d=1', B, k, nf, nf,          nf,  H//2, W//2, 1, 1),
        ('dilate d=2', B, k, nf, nf,          nf,  H//2, W//2, 1, 2),
        ('dilate d=4', B, k, nf, nf,          nf,  H//2, W//2, 1, 4),
        ('dilate d=8', B, k, nf, nf,          nf,  H//2, W//2, 1, 8),
        # up1: nf → nf*4
        ('up1',        B, k, nf, nf,          nf*4, H//2, W//2, 1, 1),
        # up2: nf → 1
        ('up2',        B, k, nf, nf,          1,    H, W,    1, 1),
        # depth: 1 → 1
        ('depth',      B, k, nf, 1,           1,    H, W,    1, 1),
    ]

    header = (f'{"Layer":<14s} {"G":>3s} {"in/g":>4s} {"out/g":>5s} '
              f'{"分组(ms)":>10s} {"标准(ms)":>10s} {"更快":>6s} {"倍率":>6s}')
    print(header)
    print('─' * len(header))

    results = []
    for c in cases:
        r = bench_conv(*c)
        results.append(r)
        print(f'{r["layer"]:<14s} {r["groups"]:>3d} {r["in_per"]:>4d} '
              f'{r["out_per"]:>5d} '
              f'{fmt_ms(r["group_ms"]):>10s} {fmt_ms(r["std_ms"]):>10s} '
              f'{r["faster"]:>6s} {r["speedup"]:>5.2f}x')

    # Weighted total (simulated actual proportion)
    total_group = sum(r['group_ms'] for r in results)
    total_std = sum(r['std_ms'] for r in results)
    print('─' * len(header))
    print(f'{"Total (sum)":<14s} {"":>3s} {"":>4s} {"":>5s} '
          f'{fmt_ms(total_group):>10s} {fmt_ms(total_std):>10s} '
          f'{"group" if total_group < total_std else "std":>6s} '
          f'{(total_std/total_group if total_group < total_std else total_group/total_std):>5.2f}x')

    group_wins = sum(1 for r in results if r['faster'] == 'group')
    std_wins = sum(1 for r in results if r['faster'] == 'std')
    print(f'\n分组赢: {group_wins}/{len(results)}  标准赢: {std_wins}/{len(results)}')

    # Also test with nf=4
    print(f'\n--- nf=4 ---\n')
    nf = 4
    cases2 = [
        ('down s=2',   B, k, nf, 1,           nf,  H, W,    2, 1),
        ('dilate d=1', B, k, nf, nf,          nf,  H//2, W//2, 1, 1),
        ('dilate d=2', B, k, nf, nf,          nf,  H//2, W//2, 1, 2),
        ('dilate d=4', B, k, nf, nf,          nf,  H//2, W//2, 1, 4),
        ('dilate d=8', B, k, nf, nf,          nf,  H//2, W//2, 1, 8),
        ('up1',        B, k, nf, nf,          nf*4, H//2, W//2, 1, 1),
        ('up2',        B, k, nf, nf,          1,    H, W,    1, 1),
        ('depth',      B, k, nf, 1,           1,    H, W,    1, 1),
    ]

    print(header)
    print('─' * len(header))
    results2 = []
    for c in cases2:
        r = bench_conv(*c)
        results2.append(r)
        print(f'{r["layer"]:<14s} {r["groups"]:>3d} {r["in_per"]:>4d} '
              f'{r["out_per"]:>5d} '
              f'{fmt_ms(r["group_ms"]):>10s} {fmt_ms(r["std_ms"]):>10s} '
              f'{r["faster"]:>6s} {r["speedup"]:>5.2f}x')

    total_group2 = sum(r['group_ms'] for r in results2)
    total_std2 = sum(r['std_ms'] for r in results2)
    print('─' * len(header))
    print(f'{"Total (sum)":<14s} {"":>3s} {"":>4s} {"":>5s} '
          f'{fmt_ms(total_group2):>10s} {fmt_ms(total_std2):>10s} '
          f'{"group" if total_group2 < total_std2 else "std":>6s} '
          f'{(total_std2/total_group2 if total_group2 < total_std2 else total_group2/total_std2):>5.2f}x')

    group_wins2 = sum(1 for r in results2 if r['faster'] == 'group')
    std_wins2 = sum(1 for r in results2 if r['faster'] == 'std')
    print(f'\n分组赢: {group_wins2}/{len(results2)}  标准赢: {std_wins2}/{len(results2)}')


if __name__ == '__main__':
    main()
