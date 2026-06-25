import jax
import jax.numpy as jnp


def haar_decomp(x):
    """2D Haar wavelet decomposition on NHWC input [B, H, W, C]. Returns LL, LH, HL, HH."""
    B, H, W, C = x.shape
    H2, W2 = H // 2, W // 2
    # subsample at even/odd positions
    tl = x[:, 0:H2 * 2:2, 0:W2 * 2:2, :]
    tr = x[:, 0:H2 * 2:2, 1:W2 * 2:2, :]
    bl = x[:, 1:H2 * 2:2, 0:W2 * 2:2, :]
    br = x[:, 1:H2 * 2:2, 1:W2 * 2:2, :]
    LL = (tl + tr + bl + br) / 4
    LH = (tl - tr + bl - br) / 4
    HL = (tl + tr - bl - br) / 4
    HH = (tl - tr - bl + br) / 4
    return LL, LH, HL, HH


def haarpsi_loss(pred, target, n_scales=3, C=0.001, alpha=4.2):
    pred_lum = pred[..., 0:1]
    target_lum = target[..., 0:1]

    # max depth handled by for loop range; input must be >= 2^n_scales
    max_scales = n_scales

    total_weight = 0.0
    total_score = 0.0
    cur_pred, cur_target = pred_lum, target_lum

    for _ in range(max_scales):
        cur_pred, LH_pred, HL_pred, _ = haar_decomp(cur_pred)
        cur_target, LH_target, HL_target, _ = haar_decomp(cur_target)

        for coeff_pred, coeff_target in [(LH_pred, HL_target), (HL_pred, LH_target)]:
            abs_pred = jnp.abs(coeff_pred)
            abs_target = jnp.abs(coeff_target)

            sim = (2 * abs_pred * abs_target + C) / (abs_pred ** 2 + abs_target ** 2 + C)
            max_abs = jnp.maximum(abs_pred, abs_target)
            weight = jax.nn.sigmoid(alpha * max_abs)

            B = sim.shape[0]
            sim_flat = sim.reshape(B, -1)
            weight_flat = weight.reshape(B, -1)
            total_score = total_score + jnp.sum(sim_flat * weight_flat, axis=1)
            total_weight = total_weight + jnp.sum(weight_flat, axis=1)

    haarpsi = total_score / (total_weight + 1e-8)
    loss = 1.0 - jnp.mean(haarpsi)
    return jnp.where(jnp.isfinite(loss), loss, 0.0)
