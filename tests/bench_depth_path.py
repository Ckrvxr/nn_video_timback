import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

device = 'cuda'
H, W = 2160, 3840
B, k, E = 1, 1, 42
nf = 2
G = 3

x_rep = torch.randn(1, B*k*G, H, W, device=device, dtype=torch.float16)
flat_idx = torch.zeros(B*k, dtype=torch.long, device=device)

# depthwise: [E, G, 1, 3, 3]
W_dw = torch.randn(E, G, 1, 3, 3, device=device, dtype=torch.float16)
b_dw = torch.randn(E, G, device=device, dtype=torch.float16)
# conv2d : [E, G, G, 3, 3]
W_c2 = torch.randn(E, G, G, 3, 3, device=device, dtype=torch.float16)
b_c2 = torch.randn(E, G, device=device, dtype=torch.float16)

def time_it(fn, n_warmup=20, n_measure=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n_measure):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    return sum(ts)/len(ts)

def run_depthwise():
    W = W_dw[flat_idx].view(B*k*G, 1, 3, 3)
    b = b_dw[flat_idx].view(B*k*G)
    return F.conv2d(x_rep, W, b, stride=1, padding=1, groups=B*k*G)

def run_conv2d():
    W = W_c2[flat_idx].view(B*k*G, G, 3, 3)
    b = b_c2[flat_idx].view(B*k*G)
    return F.conv2d(x_rep, W, b, stride=1, padding=1, groups=B*k)

import torch.nn.functional as F

t_dw = time_it(run_depthwise)
t_c2 = time_it(run_conv2d)

print(f'Depthwise (groups=G): {t_dw:.3f}ms')
print(f'Conv2d    (groups=1): {t_c2:.3f}ms')
print(f'Ratio: {t_c2/t_dw:.2f}x')
print(f'Diff:  {t_c2 - t_dw:.3f}ms')
