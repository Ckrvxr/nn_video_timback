import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from components import Timback


@torch.inference_mode()
def profile_sections(H: int, W: int, device: torch.device,
                     n_warmup: int = 20, n_measure: int = 50):
    model = Timback(num_experts=42, num_features_stream=2, n_active=1,
                       routing_threshold=1.0, dilation_rates=[1, 2, 4, 8]).to(device)
    model.eval().half()
    for p in model.parameters():
        if p.dtype == torch.float32 and p.is_floating_point():
            p.data = p.data.half()

    x = torch.randn(1, 3, H, W, dtype=torch.float16, device=device)

    def time_sec(fn):
        model.reset_state(1, device)
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_measure):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return sum(times) / len(times)

    T_total = time_sec(lambda: model(x))

    T_ssm = time_sec(lambda: model.forward_ssm_ictcp(x))

    z_c2, z_c3, z_c4, z_c5, z_out, h_t = model.forward_ssm_ictcp(x)
    router_in = torch.cat([z_c2, z_c3, z_c4, z_c5, z_out, h_t], dim=-1)
    T_fr = time_sec(lambda: (model.fusion(router_in),
                              model.router(model.fusion(router_in), x, k=1, threshold=1.0)))

    idx, weights, _ = model.router(model.fusion(router_in), x, k=1, threshold=1.0)
    T_exp = time_sec(lambda: model.experts(x, idx, weights))

    T_remaining = T_total - T_ssm - T_fr - T_exp

    print(f'  {W}×{H}  (42×1×nf=2)')
    print(f'  {"Section":<22s} {"ms":>8s}  {"%":>5s}')
    print('  ' + '-' * 37)
    print(f'  {"SSM total":<22s} {T_ssm:>8.3f}  {T_ssm/T_total*100:>5.1f}')
    print(f'  {"Fusion+Router":<22s} {T_fr:>8.3f}  {T_fr/T_total*100:>5.1f}')
    print(f'  {"Experts(3ch)":<22s} {T_exp:>8.3f}  {T_exp/T_total*100:>5.1f}')
    print(f'  {"Remaining(ret+…)":<22s} {T_remaining:>8.3f}  {T_remaining/T_total*100:>5.1f}')
    print(f'  {"Total":<22s} {T_total:>8.3f}')
    print(f'  fps: {1000/T_total:.1f}')
    print()


if __name__ == '__main__':
    profile_sections(2160, 3840, 'cuda')
