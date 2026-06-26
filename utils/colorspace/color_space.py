import numpy as np
import torch


# ── PQ Constants (float64 for internal EOTF/OETF accuracy)
M1 = 2610.0 / 16384.0
M2 = 2523.0 / 32.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0

# float64 variants (used internally by EOTF for stability)
M1_f64 = np.float64(M1)
M2_f64 = np.float64(M2)
C1_f64 = np.float64(C1)
C2_f64 = np.float64(C2)
C3_f64 = np.float64(C3)
INV_M1_f64 = np.float64(1.0 / M1)
INV_M2_f64 = np.float64(1.0 / M2)
EPS_f64 = np.float64(1e-15)


# ── BT.2020 YUV ↔ RGB
MAT_BT2020_YUV2RGB = np.array([
    [1.0,  0.0,       1.4746],
    [1.0, -0.1646,   -0.5714],
    [1.0,  1.8814,    0.0   ],
], dtype=np.float64)


# ── PQ EOTF
# ST 2084 / BT.2100: PQ code → linear luminance.
# Singularity at V ≈ 1.99206 where denominator vanishes.

_EOTF_V_MAX = 1.991


def eotf_pq_np(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, _EOTF_V_MAX).astype(np.float64, copy=False)
    v_pow = np.power(v, INV_M2_f64)
    num = np.maximum(v_pow - C1_f64, 0.0)
    den = C2_f64 - C3_f64 * v_pow
    return np.power(num / (den + EPS_f64), INV_M1_f64).astype(np.float32)


def _eotf_pq_torch(v: torch.Tensor) -> torch.Tensor:
    v = v.float().clamp(0.0, _EOTF_V_MAX).double()
    v_pow = v ** INV_M2_f64
    num = (v_pow - C1_f64).clamp(min=0.0)
    den = (C2_f64 - C3_f64 * v_pow).clamp(min=EPS_f64)
    return ((num / den) ** INV_M1_f64).float()


def yuv_to_rgb_linear_np(yuv: np.ndarray, bits: int = 8) -> np.ndarray:
    """YUV uint16 → linear RGB [0,1] float32 NHWC.  CPU-only.
       Uses BT.1886 (gamma 2.2) EOTF for BT.709 transfer content."""
    peak = np.float32((1 << bits) - 1)
    center = np.float32(1 << (bits - 1))
    f = yuv.astype(np.float32)
    f[..., 1:] -= center
    f[..., 0] /= peak
    f[..., 1:] /= peak
    rgb_nl = f @ MAT_BT2020_YUV2RGB.T
    rgb_nl = np.clip(rgb_nl, 0.0, None)
    rgb_lin = rgb_nl ** 2.2
    return np.clip(rgb_lin, 0.0, 1.0)


def yuv_to_rgb_linear_cuda(yuv: torch.Tensor, bits: int = 12) -> torch.Tensor:
    """YUV uint16 → linear RGB [0,1] fp16 NCHW.  GPU-only.  Input: [B,H,W,3] uint16.
       Uses BT.1886 (gamma 2.2) EOTF for BT.709 transfer content."""
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
    rgb_lin = (rgb_nl.permute(0, 3, 1, 2) ** 2.2).clamp(0.0, 1.0)
    return rgb_lin.half()
