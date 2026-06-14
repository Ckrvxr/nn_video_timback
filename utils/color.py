"""RGB ↔ XYB color space conversion (per JPEG XL specification)."""

import numpy as np

_M_SRGB_TO_LMS = np.array([
    [0.31933987698903715, 0.64701320029629612, 0.03364659986505812],
    [0.15264961652926260, 0.72930854387164754, 0.11786275212267974],
    [0.03970380283847010, 0.22856755283962592, 0.73123767285223000],
], dtype=np.float32)

_M_LMS_TO_SRGB = np.linalg.inv(_M_SRGB_TO_LMS).astype(np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    mask = x <= 0.04045
    linear = np.where(mask, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return linear


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    mask = x <= 0.0031308
    s = np.where(mask, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
    return np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)


def rgb_to_xyb(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (H,W,3) → XYB float (H,W,3).

    X domain [-1, 1], Y domain [0, 1], B domain [-1, 1].
    Euclidean distance in XYB ≈ perceptual difference.
    """
    lin = _srgb_to_linear(rgb)
    lms = lin @ _M_SRGB_TO_LMS.T
    lms = np.cbrt(np.maximum(lms, 0.0))
    x = lms[..., 0] - lms[..., 1]
    y = (lms[..., 0] + lms[..., 1]) * 0.5
    b = y - lms[..., 2]
    return np.stack([x, y, b], axis=-1)


def xyb_to_rgb(xyb: np.ndarray) -> np.ndarray:
    """XYB float (H,W,3) → sRGB uint8 (H,W,3)."""
    x, y, b = xyb[..., 0], xyb[..., 1], xyb[..., 2]
    l = y + x * 0.5
    m = y - x * 0.5
    s = y - b
    lms = np.stack([np.maximum(l, 0.0) ** 3,
                    np.maximum(m, 0.0) ** 3,
                    np.maximum(s, 0.0) ** 3], axis=-1)
    lin = lms @ _M_LMS_TO_SRGB.T
    return _linear_to_srgb(lin)


def xyb_to_uint8(xyb: np.ndarray) -> np.ndarray:
    """XYB float → XYB uint8, 3 bytes per pixel.

    X: [-1, 1] → [0, 255], Y: [0, 1] → [0, 255], B: [-1, 1] → [0, 255].
    """
    x = np.clip(np.round((xyb[..., 0] + 1.0) * 127.5), 0, 255).astype(np.uint8)
    y = np.clip(np.round(xyb[..., 1] * 255.0), 0, 255).astype(np.uint8)
    b = np.clip(np.round((xyb[..., 2] + 1.0) * 127.5), 0, 255).astype(np.uint8)
    return np.stack([x, y, b], axis=-1)


def uint8_to_xyb(xyb_uint8: np.ndarray) -> np.ndarray:
    """XYB uint8 → XYB float."""
    x = xyb_uint8[..., 0].astype(np.float32) / 127.5 - 1.0
    y = xyb_uint8[..., 1].astype(np.float32) / 255.0
    b = xyb_uint8[..., 2].astype(np.float32) / 127.5 - 1.0
    return np.stack([x, y, b], axis=-1)


def rgb_to_xyb_uint8(rgb: np.ndarray) -> np.ndarray:
    """One-shot: sRGB uint8 (H,W,3) → XYB uint8 (H,W,3)."""
    return xyb_to_uint8(rgb_to_xyb(rgb))


def xyb_uint8_to_rgb(xyb_uint8: np.ndarray) -> np.ndarray:
    """One-shot: XYB uint8 (H,W,3) → sRGB uint8 (H,W,3)."""
    return xyb_to_rgb(uint8_to_xyb(xyb_uint8))


# ── XYB float16 planar + bytesplit ──────────────────────
# For zstd-friendly compression:
#   float16 → byte-split (hi/lo) → channel-planar
#   6 planes: [X_hi][X_lo][Y_hi][Y_lo][B_hi][B_lo]


def xyb_f32_to_planar_bytesplit(xyb: np.ndarray) -> np.ndarray:
    """XYB float32 (H,W,3) → flat uint8 array [X_hi][X_lo][Y_hi][Y_lo][B_hi][B_lo].

    Each float16 is byte-split: hi byte = sign(1) + exp(5) + mantissa(2),
    lo byte = mantissa(8).  Channel-planar layout helps zstd find long
    runs of identical bytes.
    """
    f16 = xyb.astype(np.float16)
    u16 = f16.view(np.uint16)
    hi = ((u16 >> 8) & 0xFF).astype(np.uint8)
    lo = (u16 & 0xFF).astype(np.uint8)
    # Planar: all channels' hi bytes first, then lo bytes
    return np.concatenate([
        hi[..., 0].ravel(), lo[..., 0].ravel(),
        hi[..., 1].ravel(), lo[..., 1].ravel(),
        hi[..., 2].ravel(), lo[..., 2].ravel(),
    ])


def planar_bytesplit_to_xyb_f32(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    """Inverse of xyb_f32_to_planar_bytesplit."""
    n = h * w
    x_hi = flat[0:n]
    x_lo = flat[n:2*n]
    y_hi = flat[2*n:3*n]
    y_lo = flat[3*n:4*n]
    b_hi = flat[4*n:5*n]
    b_lo = flat[5*n:6*n]

    def _decode(hi, lo):
        return ((hi.astype(np.uint16) << 8) | lo.astype(np.uint16)).view(np.float16).reshape(h, w)
    return np.stack([_decode(x_hi, x_lo), _decode(y_hi, y_lo), _decode(b_hi, b_lo)], axis=-1).astype(np.float32)
