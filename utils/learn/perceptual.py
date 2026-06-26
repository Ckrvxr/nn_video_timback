"""VGG-based perceptual metrics (LPIPS-like) for validation."""

import jax
import jax.numpy as jnp
import flaxmodels as fm


_vgg = None
_vgg_params = None


def _ensure_vgg():
    global _vgg, _vgg_params
    if _vgg is None:
        _vgg = fm.VGG19(pretrained='imagenet', include_head=False, normalize=False)
        _vgg_params = _vgg.init(jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)))


def vgg_distance(pred_rgb, target_rgb):
    """Feature-space MSE (lower = more perceptually similar).

    Parameters
    ----------
    pred_rgb : jax.Array or np.ndarray  [B, H, W, 3]
        Linear RGB in [0, 1] range.
    target_rgb : same shape
        Reference RGB.

    Returns
    -------
    float
        Mean squared error in VGG19 conv feature space.
    """
    _ensure_vgg()

    # Downsample to 224x224 (VGG expects this size)
    B, H, W, C = pred_rgb.shape
    if H != 224 or W != 224:
        pred_rgb = jax.image.resize(pred_rgb, (B, 224, 224, C), method='bilinear')
        target_rgb = jax.image.resize(target_rgb, (B, 224, 224, C), method='bilinear')

    # VGG normalisation baked into flaxmodels: input [0, 1] → subtract mean / divide std
    pred_feat = _vgg.apply(_vgg_params, pred_rgb)
    target_feat = _vgg.apply(_vgg_params, target_rgb)

    return float(jnp.mean((pred_feat - target_feat) ** 2))
