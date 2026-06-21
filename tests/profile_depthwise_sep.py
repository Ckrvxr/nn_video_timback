import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


nf = 2
k = 2
nfk = nf * k
dilations_a = [1, 2, 4, 8]
dilations_b = [1, 2, 4, 8, 16, 32]


class VersionA(nn.Module):
    """Current: down(s=2) → 4×dilated grouped → up1 → PixelShuffle → up2"""
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(1 * k, nfk, 3, 2, 1, groups=k)
        body = []
        for d in dilations_a:
            body.append(nn.Conv2d(nfk, nfk, 3, 1, d, dilation=d, groups=nfk))
            body.append(nn.PReLU(nfk))
        self.body = nn.Sequential(*body)
        self.up1 = nn.Conv2d(nfk, nfk * 4, 3, 1, 1, groups=nfk)
        self.up2 = nn.Conv2d(nfk, k, 3, 1, 1, groups=k)

    def forward(self, x):
        x = self.down(x)
        x = self.body(x)
        x = self.up1(x)
        x = F.pixel_shuffle(x, 2)
        x = self.up2(x)
        return 0.1 * torch.tanh(x)


class DepthwiseSepBlock(nn.Module):
    """depthwise(d) → pointwise 1×1"""
    def __init__(self, ch, d):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, d, dilation=d, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        self.prelu = nn.PReLU(ch)

    def forward(self, x):
        return self.prelu(self.pw(self.dw(x)))


class VersionB(nn.Module):
    """6-layer depthwise separable, full res, skip"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(k, nfk, 1)
        self.blocks = nn.ModuleList([DepthwiseSepBlock(nfk, d) for d in dilations_b])
        self.out = nn.Conv2d(nfk, k, 1)

    def forward(self, x):
        x = self.proj(x)
        skip = x
        for blk in self.blocks:
            x = blk(x)
        x = x + skip
        x = self.out(x)
        return 0.1 * torch.tanh(x)


class VersionC(nn.Module):
    """4-layer depthwise separable, full res, skip"""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(k, nfk, 1)
        self.blocks = nn.ModuleList([DepthwiseSepBlock(nfk, d) for d in dilations_a])
        self.out = nn.Conv2d(nfk, k, 1)

    def forward(self, x):
        x = self.proj(x)
        skip = x
        for blk in self.blocks:
            x = blk(x)
        x = x + skip
        x = self.out(x)
        return 0.1 * torch.tanh(x)


def calc_rf(dilations, has_down):
    if has_down:
        return 9 + 4 * sum(dilations)
    else:
        return 1 + 2 * sum(dilations)


def calc_macs(mod, H, W):
    macs = 0
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            k = m.kernel_size[0]
            _, c_in = m.in_channels, m.weight.shape[1]
            c_out = m.out_channels
            g = m.groups
            h = (H + 2 * m.padding[0] - k * m.dilation[0]) // m.stride[0] + 1 if m.stride[0] > 1 else H
            w = (W + 2 * m.padding[0] - k * m.dilation[0]) // m.stride[0] + 1 if m.stride[0] > 1 else W
            macs += k * k * (c_in // g) * c_out * h * w
    return macs


def test():
    devices = ['cuda'] if torch.cuda.is_available() else ['cpu']
    sizes = {
        '720p':  (720,  1280),
        '1080p': (1080, 1920),
    }

    for dev_name in devices:
        for size_name, (H, W) in sizes.items():
            print(f'\n## {dev_name} | {size_name} ({H}×{W})\n')
            x = torch.randn(1, k, H, W).to(dev_name)
            x_rep = x.repeat(1, 1, 1, 1)

            for ver_name, mod_cls, dilations, has_down in [
                ('A (current down/up [1,2,4,8])', VersionA, dilations_a, True),
                ('B (depthwise 6-layer [1,2,4,8,16,32])', VersionB, dilations_b, False),
                ('C (depthwise 4-layer [1,2,4,8])', VersionC, dilations_a, False),
            ]:
                mod = mod_cls().to(dev_name)
                if dev_name == 'cuda':
                    mod = mod.half()
                    x_h = x.half()
                else:
                    x_h = x

                rf = calc_rf(dilations, has_down)
                macs = calc_macs(mod, H, W)

                _ = mod(x_h)
                if dev_name == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                n_warmup = 50
                n_measure = 100
                for _ in range(n_warmup):
                    mod(x_h)
                if dev_name == 'cuda':
                    torch.cuda.synchronize()

                mem_before = torch.cuda.memory_allocated() if dev_name == 'cuda' else 0
                torch.cuda.reset_peak_memory_stats() if dev_name == 'cuda' else None

                t = time.perf_counter()
                for _ in range(n_measure):
                    mod(x_h)
                if dev_name == 'cuda':
                    torch.cuda.synchronize()
                t = (time.perf_counter() - t) / n_measure * 1000

                peak_mem = (torch.cuda.max_memory_allocated() - mem_before) / 1024**2 if dev_name == 'cuda' else 0

                print(f'  {ver_name}')
                print(f'    RF:         {rf}×{rf}')
                print(f'    MACs:       {macs/1e6:.1f}M')
                print(f'    Latency:    {t:.2f}ms')
                if dev_name == 'cuda':
                    print(f'    Peak VRAM:  {peak_mem:.1f}MB')
                print()


if __name__ == '__main__':
    test()
