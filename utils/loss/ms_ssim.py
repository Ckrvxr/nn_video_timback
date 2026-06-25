import jax
import jax.numpy as jnp
from jax import lax


def _gaussian_kernel(size, sigma):
    coords = jnp.arange(size, dtype=jnp.float32) - size // 2
    g = jnp.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel  # [size, size]


def _blur(x, kernel, padding):
    """Blur NHWC tensor with 2D Gaussian kernel. x: [B, H, W, C]."""
    C = x.shape[-1]
    k = kernel[None, :, :, None] * jnp.ones((1, 1, 1, C))  # [1, size, size, C]
    # grouped conv with groups=C
    out = lax.conv_general_dilated(
        x, k,
        window_strides=(1, 1),
        padding=padding,
        feature_group_count=C,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return out


def _ssim_per_scale(x, y, kernel, C1, C2):
    pad = kernel.shape[0] // 2
    padding = ((pad, pad), (pad, pad))

    mu_x = _blur(x, kernel, padding)
    mu_y = _blur(y, kernel, padding)
    mu_xx = _blur(x * x, kernel, padding)
    mu_yy = _blur(y * y, kernel, padding)
    mu_xy = _blur(x * y, kernel, padding)

    sigma_x = jnp.maximum(mu_xx - mu_x ** 2, 0)
    sigma_y = jnp.maximum(mu_yy - mu_y ** 2, 0)
    sigma_xy = mu_xy - mu_x * mu_y

    luminance = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)
    cs = (2 * sigma_xy + C2) / (sigma_x + sigma_y + C2)
    ssim_map = luminance * cs
    return ssim_map.mean(axis=(1, 2, 3))


def ms_ssim_loss(pred, target, n_channels=3, window_size=11, sigma=1.5, n_scales=5, K1=0.01, K2=0.03):
    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2
    kernel = _gaussian_kernel(window_size, sigma)

    max_scales = min(n_scales,
                     int(jnp.floor(jnp.log2(pred.shape[1])).astype(jnp.int32)),
                     int(jnp.floor(jnp.log2(pred.shape[2])).astype(jnp.int32)))

    msssim = 1.0
    cur_pred, cur_target = pred, target
    for _ in range(max_scales):
        ssim = _ssim_per_scale(cur_pred, cur_target, kernel, C1, C2)
        msssim = msssim * ssim
        # average pool 2x, NHWC
        cur_pred = lax.reduce_window(cur_pred, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        cur_pred = cur_pred / 4.0
        cur_target = lax.reduce_window(cur_target, 0.0, lax.add, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
        cur_target = cur_target / 4.0

    loss = 1.0 - jnp.mean(msssim)
    return jnp.where(jnp.isfinite(loss), loss, 0.0)
