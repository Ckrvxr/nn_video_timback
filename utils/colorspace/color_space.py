import jax
import jax.numpy as jnp
import numpy as np


# ── PQ Constants (float64 for internal EOTF/OETF accuracy)
M1 = 2610.0 / 16384.0
M2 = 2523.0 / 32.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0

# float32 variants (used only where float64 is not available, e.g. JAX without x64)
M1_f32 = np.float32(M1)
M2_f32 = np.float32(M2)
C1_f32 = np.float32(C1)
C2_f32 = np.float32(C2)
C3_f32 = np.float32(C3)
INV_M1_f32 = np.float32(1.0 / M1)
INV_M2_f32 = np.float32(1.0 / M2)
EPS_f32 = np.float32(1e-8)

# float64 variants (used internally by numpy EOTF/OETF for stability)
M1_f64 = np.float64(M1)
M2_f64 = np.float64(M2)
C1_f64 = np.float64(C1)
C2_f64 = np.float64(C2)
C3_f64 = np.float64(C3)
INV_M1_f64 = np.float64(1.0 / M1)
INV_M2_f64 = np.float64(1.0 / M2)
EPS_f64 = np.float64(1e-15)


# ── BT.2020 YUV ↔ RGB Matrices
MAT_BT2020_YUV2RGB = np.array([
    [1.0,  0.0,       1.4746],
    [1.0, -0.1646,   -0.5714],
    [1.0,  1.8814,    0.0   ],
], dtype=np.float64)

MAT_RGB2YUV = np.linalg.inv(MAT_BT2020_YUV2RGB)


# ── BT.2100 ICtCp Matrices
MAT_RGB2LMS = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
], dtype=np.float64) / 4096.0

MAT_LMS2RGB = np.linalg.inv(MAT_RGB2LMS)

MAT_LMS2ICTCP = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
], dtype=np.float64) / 4096.0

MAT_ICTCP2LMS = np.linalg.inv(MAT_LMS2ICTCP)


# ── JAX Matrices (float32 copies for JIT efficiency)
_M_YUV2RGB = jnp.array(MAT_BT2020_YUV2RGB.astype(np.float32))
_M_RGB2YUV = jnp.array(MAT_RGB2YUV.astype(np.float32))
_M_RGB2LMS = jnp.array(MAT_RGB2LMS.astype(np.float32))
_M_LMS2RGB = jnp.array(MAT_LMS2RGB.astype(np.float32))
_M_LMS2ICTCP = jnp.array(MAT_LMS2ICTCP.astype(np.float32))
_M_ICTCP2LMS = jnp.array(MAT_ICTCP2LMS.astype(np.float32))


# ── PQ Transfer Functions
#
# BT.2100 / ST 2084:
#   EOTF: PQ code V ∈ [0, 1)  →  normalised luminance Y ∈ [0, ∞)
#   OETF: normalised luminance Y ∈ [0, ∞)  →  PQ code V ∈ [0, 2)
# The formulas are bijective on these ranges.  The EOTF has a singularity at
# V ≈ (C2/C3)^(1/IM2) ≈ 1.99206 where the denominator vanishes.  We use
# float64 arithmetic internally so that values up to V ≈ 1.991 are computed
# accurately (covering the full BT.2020 gamut where max V ≈ 1.94).

_EOTF_V_MAX = 1.991  # safely below singularity ≈ 1.99206


# ── Float64-accelerated numpy EOTF / OETF ─────────────────────────────

def eotf_pq_np(v: np.ndarray) -> np.ndarray:
    """PQ → linear luminance.  Internally float64, returns float32."""
    v = np.clip(v, 0.0, _EOTF_V_MAX).astype(np.float64, copy=False)
    v_pow = np.power(v, INV_M2_f64)
    num = np.maximum(v_pow - C1_f64, 0.0)
    den = C2_f64 - C3_f64 * v_pow
    return np.power(num / (den + EPS_f64), INV_M1_f64).astype(np.float32)


def oetf_pq_np(lin: np.ndarray) -> np.ndarray:
    """Linear luminance → PQ.  Internally float64, returns float32."""
    lin = np.maximum(lin, 0.0).astype(np.float64, copy=False)
    l_pow = np.power(lin, M1_f64)
    num = C1_f64 + C2_f64 * l_pow
    den = 1.0 + C3_f64 * l_pow
    return np.power(num / den, M2_f64).astype(np.float32)


# ── Float32 JAX versions (used by the GPU pipeline) ───────────────────

def eotf_pq(v):
    v = jnp.clip(v, 0.0, _EOTF_V_MAX)
    v_safe = jnp.maximum(v, 1e-10)
    v_pow = v_safe ** INV_M2_f32
    num = jnp.maximum(v_pow - C1_f32, 0.0)
    den = C2_f32 - C3_f32 * v_pow
    return (num / (den + EPS_f32)) ** INV_M1_f32


def oetf_pq(lin):
    lin = jnp.maximum(lin, 0.0)
    l_pow = lin ** M1_f32
    num = C1_f32 + C2_f32 * l_pow
    den = 1.0 + C3_f32 * l_pow
    return (num / den) ** M2_f32


# ── Numpy Conversions (BT.2100 ICtCp) ─────────────────────────────────

def yuv_to_ictcp_np(yuv: np.ndarray, bits: int = 8) -> np.ndarray:
    peak = np.float32((1 << bits) - 1)
    center = np.float32(1 << (bits - 1))
    f = yuv.astype(np.float32)
    f[..., 1:] -= center
    f[..., 0] /= peak
    f[..., 1:] /= peak

    # YUV → RGB (BT.2020).  Values can exceed [0, 1] for saturated colours;
    # the float64 EOTF handles this stably up to V ≈ 1.94.
    rgb_nl = f @ MAT_BT2020_YUV2RGB.T
    rgb_nl = np.clip(rgb_nl, 0.0, None)  # only clamp negatives, PQ is undefined for V<0
    rgb_lin = eotf_pq_np(rgb_nl)
    lms = rgb_lin @ MAT_RGB2LMS.T
    lms_p = oetf_pq_np(lms)
    return lms_p @ MAT_LMS2ICTCP.T


def ictcp_to_yuv_np(ictcp: np.ndarray, bits: int = 8) -> np.ndarray:
    lms_p = ictcp @ MAT_ICTCP2LMS.T
    lms = eotf_pq_np(lms_p)
    rgb_lin = lms @ MAT_LMS2RGB.T
    rgb_nl = oetf_pq_np(rgb_lin)
    yuv_n = rgb_nl @ MAT_RGB2YUV.T

    peak = np.float32((1 << bits) - 1)
    center = np.float32(1 << (bits - 1))
    yuv_n[..., 0] *= peak
    yuv_n[..., 1] = yuv_n[..., 1] * peak + center
    yuv_n[..., 2] = yuv_n[..., 2] * peak + center
    # Guard against uint16 wraparound from float round-trip noise.
    yuv_n = np.clip(yuv_n, 0.0, peak * 2)
    if bits > 8:
        return np.round(yuv_n).astype(np.uint16)
    return np.round(yuv_n).astype(np.uint8)


def rgb_to_ictcp_np(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ MAT_RGB2LMS.T
    lms_p = oetf_pq_np(lms)
    return lms_p @ MAT_LMS2ICTCP.T


def ictcp_to_rgb_np(ictcp: np.ndarray) -> np.ndarray:
    lms_p = ictcp @ MAT_ICTCP2LMS.T
    lms = eotf_pq_np(lms_p)
    return lms @ MAT_LMS2RGB.T


# ── JAX Conversions (NHWC [B, H, W, C]) ───────────────────────────────

def yuv_to_ictcp(x: jnp.ndarray) -> jnp.ndarray:
    """YUV (normalized [-1,1]) → ICtCp. Input: [B,H,W,3] NHWC."""
    yuv = (x + 1) * 127.5
    y = yuv[..., 0:1] / 255.0
    u = (yuv[..., 1:2] - 128.0) / 255.0
    v = (yuv[..., 2:3] - 128.0) / 255.0
    yuv_s = jnp.concatenate([y, u, v], axis=-1)
    rgb_nl = yuv_s @ _M_YUV2RGB.T
    rgb_nl = jnp.clip(rgb_nl, 0.0, None)
    rgb_lin = eotf_pq(rgb_nl)
    lms = rgb_lin @ _M_RGB2LMS.T
    lms_p = oetf_pq(lms)
    return lms_p @ _M_LMS2ICTCP.T


def ictcp_to_yuv(x: jnp.ndarray) -> jnp.ndarray:
    """ICtCp → YUV (normalized [-1,1]). Input: [B,H,W,3] NHWC."""
    lms_p = x @ _M_ICTCP2LMS.T
    lms = eotf_pq(lms_p)
    rgb_lin = lms @ _M_LMS2RGB.T
    rgb_nl = oetf_pq(rgb_lin)
    yuv = rgb_nl @ _M_RGB2YUV.T

    y = yuv[..., 0:1] * 2.0 - 1.0
    u = yuv[..., 1:2] * 2.0
    v = yuv[..., 2:3] * 2.0
    return jnp.concatenate([y, u, v], axis=-1)


def rgb_to_ictcp(x: jnp.ndarray) -> jnp.ndarray:
    lms = x @ _M_RGB2LMS.T
    lms_p = oetf_pq(lms)
    return lms_p @ _M_LMS2ICTCP.T


def ictcp_to_rgb(x: jnp.ndarray) -> jnp.ndarray:
    lms_p = x @ _M_ICTCP2LMS.T
    lms = eotf_pq(lms_p)
    return lms @ _M_LMS2RGB.T


def yuv_to_rgb(x: jnp.ndarray) -> jnp.ndarray:
    yuv = (x + 1) * 127.5
    yuv_s = yuv[..., :3]
    y_ch = yuv_s[..., 0:1]
    u_ch = yuv_s[..., 1:2] - 128.0
    v_ch = yuv_s[..., 2:3] - 128.0
    rgb = (y_ch * _M_YUV2RGB[0] + u_ch * _M_YUV2RGB[1] + v_ch * _M_YUV2RGB[2])
    return rgb / 127.5 - 1.0


def rgb_to_yuv(x: jnp.ndarray) -> jnp.ndarray:
    rgb = (x + 1) * 127.5
    r_ch, g_ch, b_ch = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
    y = r_ch * _M_RGB2YUV[0, 0] + g_ch * _M_RGB2YUV[0, 1] + b_ch * _M_RGB2YUV[0, 2]
    u = r_ch * _M_RGB2YUV[1, 0] + g_ch * _M_RGB2YUV[1, 1] + b_ch * _M_RGB2YUV[1, 2] + 128.0
    v = r_ch * _M_RGB2YUV[2, 0] + g_ch * _M_RGB2YUV[2, 1] + b_ch * _M_RGB2YUV[2, 2] + 128.0
    return jnp.concatenate([y, u, v], axis=-1) / 127.5 - 1.0
