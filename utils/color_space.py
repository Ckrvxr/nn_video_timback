import numpy as np
import torch


# ── PQ Constants
M1 = 2610.0 / 16384.0
M2 = 2523.0 / 32.0
C1 = 3424.0 / 4096.0
C2 = 2413.0 / 4096.0 * 32.0
C3 = 2392.0 / 4096.0 * 32.0

M1_f32 = np.float32(M1)
M2_f32 = np.float32(M2)
C1_f32 = np.float32(C1)
C2_f32 = np.float32(C2)
C3_f32 = np.float32(C3)
INV_M1_f32 = np.float32(1.0 / M1)
INV_M2_f32 = np.float32(1.0 / M2)
EPS_f32 = np.float32(1e-8)


# ── BT.2020 YUV ↔ RGB Matrices
MAT_BT2020_YUV2RGB = np.array([
    [1.0,  0.0,       1.4746],
    [1.0, -0.1646,   -0.5714],
    [1.0,  1.8814,    0.0   ],
], dtype=np.float32)

MAT_RGB2YUV = np.linalg.inv(MAT_BT2020_YUV2RGB)


# ── BT.2100 ICtCp Matrices
MAT_RGB2LMS = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
], dtype=np.float32) / 4096.0

MAT_LMS2RGB = np.linalg.inv(MAT_RGB2LMS)

MAT_LMS2ICTCP = np.array([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
], dtype=np.float32) / 4096.0

MAT_ICTCP2LMS = np.linalg.inv(MAT_LMS2ICTCP)


# ── Numpy PQ Transfer Functions (float32 precision)

def eotf_pq_np(v: np.ndarray) -> np.ndarray:
    v_pow = np.power(np.clip(v, 0.0, 1.0), INV_M2_f32)
    num = np.maximum(v_pow - C1_f32, 0.0)
    den = C2_f32 - C3_f32 * v_pow
    return np.power(num / (den + EPS_f32), INV_M1_f32)


def oetf_pq_np(l: np.ndarray) -> np.ndarray:
    l_pow = np.power(np.clip(l, 0.0, 1.0), M1_f32)
    num = C1_f32 + C2_f32 * l_pow
    den = 1.0 + C3_f32 * l_pow
    return np.power(num / den, M2_f32)


# ── Numpy Conversions (BT.2100 ICtCp, vectorized with @)

def yuv_to_ictcp_np(yuv: np.ndarray, bits: int = 8) -> np.ndarray:
    peak = np.float32((1 << bits) - 1)
    center = np.float32(1 << (bits - 1))
    f = yuv.astype(np.float32)
    f[..., 1:] -= center
    f[..., 0] /= peak
    f[..., 1:] /= peak

    rgb_nl = f @ MAT_BT2020_YUV2RGB.T
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
    return np.round(yuv_n).astype(np.uint8)


def rgb_to_ictcp_np(rgb: np.ndarray) -> np.ndarray:
    lms = rgb @ MAT_RGB2LMS.T
    lms_p = oetf_pq_np(lms)
    return lms_p @ MAT_LMS2ICTCP.T


def ictcp_to_rgb_np(ictcp: np.ndarray) -> np.ndarray:
    lms_p = ictcp @ MAT_ICTCP2LMS.T
    lms = eotf_pq_np(lms_p)
    return lms @ MAT_LMS2RGB.T


# ── Torch Matrices

_M_YUV2RGB = torch.tensor([
    [1.0,  0.0,       1.4746],
    [1.0, -0.1646,   -0.5714],
    [1.0,  1.8814,    0.0   ],
])

_M_RGB2YUV = torch.linalg.inv(_M_YUV2RGB)

_M_RGB2LMS = torch.tensor([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688],
]).float() / 4096.0

_M_LMS2RGB = torch.linalg.inv(_M_RGB2LMS)

_M_LMS2ICTCP = torch.tensor([
    [2048, 2048, 0],
    [6610, -13613, 7003],
    [17933, -17390, -543],
]).float() / 4096.0

_M_ICTCP2LMS = torch.linalg.inv(_M_LMS2ICTCP)

_device_cache: dict = {}


def _get_matrices(device, dtype):
    key = (device, dtype)
    if key not in _device_cache:
        _device_cache[key] = (
            _M_YUV2RGB.to(device=device, dtype=dtype),
            _M_RGB2LMS.to(device=device, dtype=dtype),
            _M_LMS2ICTCP.to(device=device, dtype=dtype),
            _M_ICTCP2LMS.to(device=device, dtype=dtype),
            _M_LMS2RGB.to(device=device, dtype=dtype),
            _M_RGB2YUV.to(device=device, dtype=dtype),
        )
    return _device_cache[key]


# ── Torch JIT PQ Transfer Functions (multi-channel)

@torch.jit.script
def _eotf_pq_torch(v):
    v_clamped = torch.clamp(v, 0.0, 1.0)
    v_pow = torch.pow(v_clamped, 1.0 / 78.84375)
    num = torch.clamp(v_pow - (3424.0 / 4096.0), min=0.0)
    den = (2413.0 / 4096.0 * 32.0) - (2392.0 / 4096.0 * 32.0) * v_pow
    return torch.pow(num / (den + 1e-8), 16384.0 / 2610.0)


@torch.jit.script
def _oetf_pq_torch(l):
    l_clamped = torch.clamp(l, 0.0, 1.0)
    l_pow = torch.pow(l_clamped, 2610.0 / 16384.0)
    num = (3424.0 / 4096.0) + (2413.0 / 4096.0 * 32.0) * l_pow
    den = 1.0 + (2392.0 / 4096.0 * 32.0) * l_pow
    return torch.pow(num / den, 78.84375)


# ── Torch JIT Color Conversions (vectorized)

@torch.jit.script
def _apply_3x3(m, x):
    B, C, H, W = x.shape
    x_f = x.reshape(B, C, -1)
    return torch.matmul(m, x_f).reshape(B, C, H, W)


@torch.jit.script
def _yuv2ictcp_jit(x, m_yuv2rgb, m_rgb2lms, m_lms2ictcp):
    yuv = (x + 1) * 127.5
    y = yuv[:, 0:1] / 255.0
    u = (yuv[:, 1:2] - 128.0) / 255.0
    v = (yuv[:, 2:3] - 128.0) / 255.0
    yuv_s = torch.cat([y, u, v], dim=1)
    rgb_nl = _apply_3x3(m_yuv2rgb, yuv_s)
    rgb_lin = _eotf_pq_torch(rgb_nl)
    lms = _apply_3x3(m_rgb2lms, rgb_lin)
    lms_p = _oetf_pq_torch(lms)
    return _apply_3x3(m_lms2ictcp, lms_p)


@torch.jit.script
def _ictcp2yuv_jit(x, m_ictcp2lms, m_lms2rgb, m_rgb2yuv):
    lms_p = _apply_3x3(m_ictcp2lms, x)
    lms = _eotf_pq_torch(lms_p)
    rgb_lin = _apply_3x3(m_lms2rgb, lms)
    rgb_nl = _oetf_pq_torch(rgb_lin)
    yuv = _apply_3x3(m_rgb2yuv, rgb_nl)

    y = yuv[:, 0:1] * 2.0 - 1.0
    u = yuv[:, 1:2] * 2.0
    v = yuv[:, 2:3] * 2.0
    return torch.cat([y, u, v], dim=1)


@torch.jit.script
def _rgb2ictcp_jit(x, m_rgb2lms, m_lms2ictcp):
    lms = _apply_3x3(m_rgb2lms, x)
    lms_p = _oetf_pq_torch(lms)
    return _apply_3x3(m_lms2ictcp, lms_p)


@torch.jit.script
def _ictcp2rgb_jit(x, m_ictcp2lms, m_lms2rgb):
    lms_p = _apply_3x3(m_ictcp2lms, x)
    lms = _eotf_pq_torch(lms_p)
    return _apply_3x3(m_lms2rgb, lms)


# ── Torch Wrappers

def yuv_to_ictcp(x: torch.Tensor) -> torch.Tensor:
    m_yuv2rgb, m_rgb2lms, m_lms2ictcp, _, _, _ = _get_matrices(x.device, x.dtype)
    return _yuv2ictcp_jit(x, m_yuv2rgb, m_rgb2lms, m_lms2ictcp)


def ictcp_to_yuv(x: torch.Tensor) -> torch.Tensor:
    _, _, _, m_ictcp2lms, m_lms2rgb, m_rgb2yuv = _get_matrices(x.device, x.dtype)
    return _ictcp2yuv_jit(x, m_ictcp2lms, m_lms2rgb, m_rgb2yuv)


def rgb_to_ictcp(x: torch.Tensor) -> torch.Tensor:
    _, m_rgb2lms, m_lms2ictcp, _, _, _ = _get_matrices(x.device, x.dtype)
    return _rgb2ictcp_jit(x, m_rgb2lms, m_lms2ictcp)


def ictcp_to_rgb(x: torch.Tensor) -> torch.Tensor:
    _, _, _, m_ictcp2lms, m_lms2rgb, _ = _get_matrices(x.device, x.dtype)
    return _ictcp2rgb_jit(x, m_ictcp2lms, m_lms2rgb)


# ── Fallback Torch Conversions (non-JIT, BT.2020 YUV↔RGB only)

def yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    yuv = (x + 1) * 127.5
    y_ch = yuv[:, 0:1]
    u_ch = yuv[:, 1:2] - 128.0
    v_ch = yuv[:, 2:3] - 128.0
    m = _M_YUV2RGB.to(device=x.device, dtype=x.dtype)
    r = m[0, 0] * y_ch + m[0, 1] * u_ch + m[0, 2] * v_ch
    g = m[1, 0] * y_ch + m[1, 1] * u_ch + m[1, 2] * v_ch
    b = m[2, 0] * y_ch + m[2, 1] * u_ch + m[2, 2] * v_ch
    return torch.cat([r, g, b], dim=1) / 127.5 - 1.0


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    rgb = (x + 1) * 127.5
    r_ch, g_ch, b_ch = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    m = _M_RGB2YUV.to(device=x.device, dtype=x.dtype)
    y = m[0, 0] * r_ch + m[0, 1] * g_ch + m[0, 2] * b_ch
    u = m[1, 0] * r_ch + m[1, 1] * g_ch + m[1, 2] * b_ch + 128.0
    v = m[2, 0] * r_ch + m[2, 1] * g_ch + m[2, 2] * b_ch + 128.0
    return torch.cat([y, u, v], dim=1) / 127.5 - 1.0
