import jax.numpy as jnp
import flax.linen as nn

from core.artrt_r1 import Fusion


def pixel_unshuffle(x, r):
    B, H, W, C = x.shape
    h2, w2 = H // r, W // r
    x = x.reshape(B, h2, r, w2, r, C)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(B, h2, w2, C * r * r)


def pixel_shuffle(x, r):
    B, H, W, C = x.shape
    c2 = C // (r * r)
    x = x.reshape(B, H, W, c2, r, r)
    x = jnp.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape(B, H * r, W * r, c2)


class DSConv(nn.Module):
    ch: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.ch, kernel_size=(3, 3), padding='SAME',
                    feature_group_count=self.ch,
                    kernel_dilation=(self.dilation, self.dilation), use_bias=False)(x)
        x = nn.Conv(features=self.ch, kernel_size=(1, 1), use_bias=False)(x)
        return x


class ResBlock(nn.Module):
    ch: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        ng = min(8, self.ch // 2)
        while self.ch % ng != 0:
            ng -= 1
        shortcut = x
        x = DSConv(self.ch, self.dilation)(x)
        x = nn.GroupNorm(num_groups=ng)(x)
        x = nn.relu(x)
        x = DSConv(self.ch, self.dilation)(x)
        x = nn.GroupNorm(num_groups=ng)(x)
        x = x + shortcut
        return nn.relu(x)


class Stem(nn.Module):
    in_ch: int
    ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.in_ch, kernel_size=(3, 3), padding='SAME',
                    feature_group_count=self.in_ch, use_bias=False)(x)
        x = nn.Conv(features=self.ch, kernel_size=(1, 1), use_bias=False)(x)
        return x


class Branch(nn.Module):
    ch_list: list
    n_blocks_list: list
    dil_list: list
    in_ch: int = 1
    out_ch: int = 1

    def setup(self):
        self.stem = Stem(self.in_ch, self.ch_list[0])
        blocks = []
        cur_ch = self.ch_list[0]
        for ch, n, dil in zip(self.ch_list, self.n_blocks_list, self.dil_list):
            if ch != cur_ch:
                blocks.append(nn.Conv(features=ch, kernel_size=(1, 1), use_bias=False))
                cur_ch = ch
            for _ in range(n):
                blocks.append(ResBlock(ch, dilation=dil))
        self.blocks = blocks
        self.head = nn.Sequential([
            DSConv(self.ch_list[-1]),
            nn.Conv(features=self.out_ch, kernel_size=(1, 1), use_bias=False),
        ])

    def __call__(self, x):
        h = self.stem(x)
        for layer in self.blocks:
            h = layer(h)
        return self.head(h)


class ICtCpNet(nn.Module):
    scale: int = 1

    def setup(self):
        s2 = self.scale * self.scale
        self.i_branch = Branch(ch_list=[16, 24, 32, 48], n_blocks_list=[2, 2, 2, 1],
                                dil_list=[1, 2, 4, 8], in_ch=s2, out_ch=s2)
        self.ct_branch = Branch(ch_list=[16, 16], n_blocks_list=[1, 1],
                                dil_list=[1, 2], in_ch=s2, out_ch=s2)
        self.cp_branch = Branch(ch_list=[16, 16], n_blocks_list=[1, 1],
                                dil_list=[1, 2], in_ch=s2, out_ch=s2)
        self.fusion = Fusion()

    def __call__(self, x):
        if self.scale > 1:
            x_low = pixel_unshuffle(x, self.scale)
        else:
            x_low = x
        s2 = self.scale * self.scale
        i = self.i_branch(x_low[..., :s2])
        ct = self.ct_branch(x_low[..., s2:2 * s2])
        cp = self.cp_branch(x_low[..., 2 * s2:3 * s2])
        out = jnp.concatenate([i, ct, cp], axis=-1)
        if self.scale > 1:
            out = pixel_shuffle(out, self.scale)
        delta = self.fusion(out)
        return x + delta
