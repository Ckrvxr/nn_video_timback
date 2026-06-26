import numpy as np
import torch


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
    rgb = lms @ MAT_LMS2RGB.T
    return np.clip(rgb, 0.0, 1.0)


# ── PyTorch Conversions (NCHW [B, C, H, W]) ────────────────────────────

def ictcp_to_rgb_torch(x: torch.Tensor) -> torch.Tensor:
    """ICtCp → linear RGB [0,1]. Input: [B,3,H,W] NCHW. Output: [B,3,H,W]."""
    dt = x.dtype
    mat1 = torch.tensor(MAT_ICTCP2LMS.T.astype(np.float32), device=x.device, dtype=dt)
    lms_p = x.permute(0, 2, 3, 1) @ mat1
    lms = _eotf_pq_torch(lms_p.permute(0, 3, 1, 2))
    mat2 = torch.tensor(MAT_LMS2RGB.T.astype(np.float32), device=x.device, dtype=lms.dtype)
    rgb = lms.permute(0, 2, 3, 1) @ mat2
    return rgb.permute(0, 3, 1, 2).clamp(0.0, 1.0).to(dt)


def _eotf_pq_torch(v: torch.Tensor) -> torch.Tensor:
    v = v.float().clamp(0.0, _EOTF_V_MAX).double()
    v_pow = v ** INV_M2_f64
    num = (v_pow - C1_f64).clamp(min=0.0)
    den = (C2_f64 - C3_f64 * v_pow).clamp(min=EPS_f64)
    return ((num / den) ** INV_M1_f64).float()


def _oetf_pq_torch(lin: torch.Tensor) -> torch.Tensor:
    """Linear luminance → PQ code.  float32 in/out."""
    lin = lin.float().clamp(min=0.0)
    l_pow = lin ** M1_f32
    num = C1_f32 + C2_f32 * l_pow
    den = 1.0 + C3_f32 * l_pow
    return (num / den) ** M2_f32


def yuv_to_rgb_linear_np(yuv: np.ndarray, bits: int = 8) -> np.ndarray:
    """YUV uint16 → linear RGB [0,1] float32 NHWC.  CPU-only."""
    peak = np.float32((1 << bits) - 1)
    center = np.float32(1 << (bits - 1))
    f = yuv.astype(np.float32)
    f[..., 1:] -= center
    f[..., 0] /= peak
    f[..., 1:] /= peak
    rgb_nl = f @ MAT_BT2020_YUV2RGB.T
    rgb_nl = np.clip(rgb_nl, 0.0, None)
    rgb_lin = eotf_pq_np(rgb_nl)
    return np.clip(rgb_lin, 0.0, 1.0)


def yuv_to_rgb_linear_cuda(yuv: torch.Tensor, bits: int = 12) -> torch.Tensor:
    """YUV uint16 → linear RGB [0,1] bf16 NCHW.  GPU-only.  Input: [B,H,W,3] uint16."""
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))
    f = yuv.float()
    y = f[..., 0:1] / peak
    u = (f[..., 1:2] - center) / peak
    v = (f[..., 2:3] - center) / peak
    yuv_n = torch.cat([y, u, v], dim=-1)

    mat = torch.tensor(MAT_BT2020_YUV2RGB.T.astype(np.float32), device=yuv.device)
    rgb_nl = yuv_n @ mat
    rgb_nl = rgb_nl.clamp(min=0.0)
    rgb_lin = _eotf_pq_torch(rgb_nl.permute(0, 3, 1, 2)).clamp(0.0, 1.0)
    return rgb_lin.bfloat16()
