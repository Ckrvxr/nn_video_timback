import jax.numpy as jnp
import numpy as np

from utils.colorspace import yuv_to_ictcp as _yuv_to_ictcp


def yuv_to_ictcp_np_batch(yuv_batch: np.ndarray) -> np.ndarray:
    B, H, W, _ = yuv_batch.shape
    bits = 8
    if yuv_batch.dtype == np.uint16:
        max_val = int(yuv_batch.max())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    f = yuv_batch.astype(np.float32)
    f[..., 0] = f[..., 0] / peak * 255.0
    f[..., 1:] = (f[..., 1:] - center) / peak * 255.0 + 128.0
    f = f / 127.5 - 1.0

    from utils.colorspace import yuv_to_ictcp_np
    return yuv_to_ictcp_np(f)


def batch_yuv_to_ictcp(yuv):
    # yuv: numpy array [B, H, W, 3] or [B, F, H, W, 3]
    orig_ndim = yuv.ndim
    if orig_ndim == 5:
        B, F, H, W, C = yuv.shape
        yuv = yuv.reshape(B * F, H, W, C)

    bits = 8
    if yuv.dtype == np.uint16:
        max_val = int(yuv.max())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    f = yuv.astype(np.float32)
    f[..., 0] = f[..., 0] / peak * 255.0
    f[..., 1:] = (f[..., 1:] - center) / peak * 255.0 + 128.0
    f = f / 127.5 - 1.0

    from utils.colorspace import yuv_to_ictcp_np
    out = yuv_to_ictcp_np(f)

    if orig_ndim == 5:
        out = out.reshape(B, F, H, W, 3)
    return out
