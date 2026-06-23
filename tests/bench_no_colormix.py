import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from components import Timback

device = 'cuda'
H, W = 2160, 3840

for cfg in [(2, [1]), (2, [1, 2, 4, 8]), (1, [1])]:
    nf, dr = cfg
    model = Timback(42, n_active=1, num_features_stream=nf,
                       dilation_rates=dr, routing_threshold=1.0).to(device)
    model.eval().half()
    for p in model.parameters():
        if p.dtype == torch.float32 and p.is_floating_point():
            p.data = p.data.half()

    x = torch.randn(1, 3, H, W, dtype=torch.float16, device=device)

    evt_s = torch.cuda.Event(enable_timing=True)
    evt_e = torch.cuda.Event(enable_timing=True)

    model.reset_state(1, device)
    for _ in range(20):
        model(x)
    torch.cuda.synchronize()

    ts = []
    for _ in range(100):
        evt_s.record()
        model(x)
        evt_e.record()
        torch.cuda.synchronize()
        ts.append(evt_s.elapsed_time(evt_e))
    avg = sum(ts) / len(ts)
    fps = 1000 / avg

    print(f'  nf={nf}  d={dr}')
    print(f'    {avg:.3f}ms  {fps:.1f}fps')
