import jax.numpy as jnp


def calculate_psnr_batch(pred, target, max_val=2.0):
    """PSNR per sample in batch. Input: NHWC [B, H, W, C], range [-1, 1]."""
    mse = jnp.mean((pred - target) ** 2, axis=(1, 2, 3))
    return 20 * jnp.log10(max_val) - 10 * jnp.log10(mse + 1e-8)


def calculate_ssim_batch(pred, target, max_val=2.0, K1=0.01, K2=0.03):
    """Simplified SSIM. Input: NHWC [B, H, W, C], range [-1, 1]."""
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    mu_pred = jnp.mean(pred, axis=(1, 2), keepdims=True)
    mu_target = jnp.mean(target, axis=(1, 2), keepdims=True)

    sigma_pred = jnp.mean(pred ** 2, axis=(1, 2), keepdims=True) - mu_pred ** 2
    sigma_target = jnp.mean(target ** 2, axis=(1, 2), keepdims=True) - mu_target ** 2
    sigma_pt = jnp.mean(pred * target, axis=(1, 2), keepdims=True) - mu_pred * mu_target

    ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pt + C2)) / \
           ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
    return jnp.mean(ssim, axis=(1, 2, 3))
