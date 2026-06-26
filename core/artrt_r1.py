import jax.numpy as jnp
import flax.linen as nn


# ── pixel_unshuffle / shuffle (NHWC) ──

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


# ── Fusion (shared by all models) ──

class Fusion(nn.Module):
    @nn.compact
    def __call__(self, x):
        gap = x.mean(axis=(1, 2), keepdims=True)
        ca = nn.Conv(features=3, kernel_size=(1, 1), use_bias=True)(gap)
        ca = nn.sigmoid(ca)
        x = x * ca
        x = nn.Conv(features=3, kernel_size=(3, 3), padding='SAME', use_bias=True)(x)
        x = jnp.tanh(x)
        scale = self.param('delta_scale', lambda rng, shape: jnp.array([0.5, 0.08, 0.08]), (3,))
        return x * scale.reshape(1, 1, 1, 3)

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


# ── Building blocks ──

class DSConv(nn.Module):
    """Depthwise separable convolution."""
    ch: int
    kernel: tuple = (3, 3)
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.ch,
            kernel_size=self.kernel,
            padding='SAME',
            feature_group_count=self.ch,
            kernel_dilation=(self.dilation, self.dilation),
            use_bias=False,
        )(x)
        x = nn.Conv(features=self.ch, kernel_size=(1, 1), use_bias=False)(x)
        return x


class ResBlock(nn.Module):
    """Residual block with DSConv + LayerNorm + ReLU + optional SE."""
    ch: int
    kernel: tuple = (3, 3)
    dilation: int = 1
    use_se: bool = False

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = DSConv(self.ch, kernel=self.kernel, dilation=self.dilation)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = DSConv(self.ch, dilation=self.dilation)(x)
        x = nn.LayerNorm()(x)
        if self.use_se:
            gap = x.mean(axis=(1, 2), keepdims=True)
            w = nn.Conv(max(self.ch // 4, 4), kernel_size=(1, 1), use_bias=True)(gap)
            w = nn.relu(w)
            w = nn.Conv(self.ch, kernel_size=(1, 1), use_bias=True)(w)
            w = nn.sigmoid(w)
            x = x * w
        return nn.relu(x + shortcut)


# ── Branches ──

class HeavyBranch(nn.Module):
    """Heavy branch for the I (luma) component.

    4 ResBlocks with SE gating. Stacked dilations 1/2/4/8 with kernels
    5/3/5/3 provide ~41 internal pixel RF (~330 original px at scale=8).
    """
    out_ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(1, 1), use_bias=False)(x)
        x = ResBlock(32, kernel=(5, 5), dilation=1, use_se=True)(x)
        x = ResBlock(32, dilation=2, use_se=True)(x)
        x = nn.Conv(features=48, kernel_size=(1, 1), use_bias=False)(x)
        x = ResBlock(48, kernel=(5, 5), dilation=4, use_se=True)(x)
        x = ResBlock(48, dilation=4, use_se=True)(x)
        x = nn.Conv(features=self.out_ch, kernel_size=(1, 1), use_bias=False)(x)
        return x


class LightBranch(nn.Module):
    """Lightweight symmetric branch for Ct/Cp (chroma) components."""
    out_ch: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=24, kernel_size=(1, 1), use_bias=False)(x)
        x = ResBlock(24, dilation=1, use_se=True)(x)
        x = ResBlock(24, dilation=2, use_se=True)(x)
        x = nn.Conv(features=self.out_ch, kernel_size=(1, 1), use_bias=False)(x)
        return x


# ── ICtCpNet v2 ──

class DetailPath(nn.Module):
    """Detail refinement: Conv5×5(6→32) → Conv1×1(32→3), no activation."""

    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, 6] = concat(delta, frame_center)
        h = nn.Conv(
            features=6, kernel_size=(5, 5), padding='SAME',
            feature_group_count=6, use_bias=False,
            kernel_init=nn.initializers.zeros,
        )(x)
        h = nn.Conv(
            features=8, kernel_size=(1, 1),
            use_bias=False, kernel_init=nn.initializers.zeros,
        )(h)
        detail = nn.Conv(
            features=3, kernel_size=(1, 1),
            use_bias=False, kernel_init=nn.initializers.zeros,
        )(h)
        scale = self.param('scale', lambda rng, s: jnp.array([0.2, 0.04, 0.04]), (3,))
        return detail * scale.reshape(1, 1, 1, 3)


class ICtCpNetV2(nn.Module):
    """Patch-based compression artifact removal network.

    Defaults:
        scale=4  for 512x512 patches -> 128x128 internal resolution.
        The I branch is heavy; Ct/Cp branches are lightweight and symmetric.

    multi_frame=True:
        Input is [B, H, W, 9] = 3 frames of ICtCp stacked on channel dim.
        I branch sees I from all 3 frames; Ct/Cp branches see center frame only.
        Output is [B, H, W, 3] = enhanced center frame.

    detail_path=True:
        Adds a detail bypass: DWConv5×5 on the original-resolution center frame,
        bypassing the main pixel_unshuffle / branches path.
    """
    scale: int = 4
    multi_frame: bool = False
    detail_path: bool = False

    def setup(self):
        s2 = self.scale * self.scale
        self.i_branch = HeavyBranch(out_ch=s2)
        self.ct_branch = LightBranch(out_ch=s2)
        self.cp_branch = LightBranch(out_ch=s2)
        self.fusion = Fusion()
        if self.detail_path:
            self.detail = DetailPath()

    def __call__(self, x):
        s2 = self.scale * self.scale
        if self.multi_frame:
            # x: [B, H, W, 9] = 3 frames ICtCp, center frame is channels 3:6
            frame_center = x[..., 3:6]
            x_low = pixel_unshuffle(x, self.scale)
            # x_low: [B, H/s, W/s, 9*s²]
            # Concat I from all 3 frames (chunks 0, 1, 2 of s²)
            i0 = x_low[..., :s2]
            i1 = x_low[..., 3 * s2:4 * s2]
            i2 = x_low[..., 6 * s2:7 * s2]
            i_in = jnp.concatenate([i0, i1, i2], axis=-1)
            ct_in = x_low[..., 4 * s2:5 * s2]
            cp_in = x_low[..., 5 * s2:6 * s2]
        else:
            frame_center = x
            if self.scale > 1:
                x_low = pixel_unshuffle(x, self.scale)
            else:
                x_low = x
            i_in = x_low[..., :s2]
            ct_in = x_low[..., s2:2 * s2]
            cp_in = x_low[..., 2 * s2:3 * s2]

        i = self.i_branch(i_in)
        ct = self.ct_branch(ct_in)
        cp = self.cp_branch(cp_in)
        out = jnp.concatenate([i, ct, cp], axis=-1)
        if self.scale > 1:
            out = pixel_shuffle(out, self.scale)
        delta = self.fusion(out)
        if self.detail_path:
            detail_input = jnp.concatenate([delta, frame_center], axis=-1)
            delta = delta + self.detail(detail_input)
        return frame_center + delta
