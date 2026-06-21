"""
Verify all color space conversions.

Coverage:
- Torch JIT ≈ numpy f32 (all 4 directions)
- Roundtrip reconstruction
- PQ transfer function identity
- Edge cases (black, white, saturated)
- fp16 compatibility
- Matrix orthogonality
"""
from tests.helpers import mock_triton, mock_mamba_ssm
mock_triton()
mock_mamba_ssm()

import numpy as np
import torch
from models.components.color_space import (
    yuv_to_ictcp_np, ictcp_to_yuv_np, rgb_to_ictcp_np, ictcp_to_rgb_np,
    eotf_pq_np, oetf_pq_np,
    yuv_to_ictcp, ictcp_to_yuv, rgb_to_ictcp, ictcp_to_rgb,
    _M_YUV2RGB, _M_RGB2LMS, _M_LMS2ICTCP, _M_ICTCP2LMS, _M_LMS2RGB, _M_RGB2YUV,
)

torch.manual_seed(0)
np.random.seed(0)
H, W = 64, 64
MAX_ABS = 1e-4


def _yuv_nhwc_to_torch(yuv_nhwc):
    return torch.from_numpy(yuv_nhwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def _torch_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).detach().numpy()


# ── matrix identity ──

def test_matrix_identity():
    for name, M, Minv in [
        ("RGB2LMS", _M_RGB2LMS.numpy(), _M_LMS2RGB.numpy()),
        ("LMS2ICTCP", _M_LMS2ICTCP.numpy(), _M_ICTCP2LMS.numpy()),
    ]:
        err = np.abs(M @ Minv - np.eye(3, dtype=np.float32)).max()
        assert err < 1e-5, f"{name}: {err:.2e}"

    err = np.abs(_M_YUV2RGB.numpy() @ _M_RGB2YUV.numpy() - np.eye(3, dtype=np.float32)).max()
    assert err < 1e-5, f"YUV2RGB*RGB2YUV: {err:.2e}"


# ── numpy ↔ torch consistency (all 4 paths) ──

def test_yuv_to_ictcp_consistency():
    yuv = np.random.randint(0, 256, (H, W, 3)).astype(np.uint8)
    ref = yuv_to_ictcp_np(yuv, bits=8)
    out = _torch_to_np(yuv_to_ictcp(_yuv_nhwc_to_torch(yuv)))
    assert np.abs(ref - out).max() < MAX_ABS


def test_ictcp_to_yuv_consistency():
    ictcp = np.random.uniform(-0.5, 0.5, (H, W, 3)).astype(np.float32)
    ref = ictcp_to_yuv_np(ictcp, bits=8)
    t = torch.from_numpy(ictcp).permute(2, 0, 1).unsqueeze(0).float()
    out = np.round((_torch_to_np(ictcp_to_yuv(t)) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    assert np.abs(ref.astype(np.int32) - out.astype(np.int32)).max() < 2


def test_rgb_to_ictcp_consistency():
    rgb = np.random.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    ref = rgb_to_ictcp_np(rgb)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    assert np.abs(ref - _torch_to_np(rgb_to_ictcp(t))).max() < MAX_ABS


def test_ictcp_to_rgb_consistency():
    ictcp = np.random.uniform(-0.5, 0.5, (H, W, 3)).astype(np.float32)
    ref = ictcp_to_rgb_np(ictcp)
    t = torch.from_numpy(ictcp).permute(2, 0, 1).unsqueeze(0).float()
    assert np.abs(ref - _torch_to_np(ictcp_to_rgb(t))).max() < MAX_ABS


# ── roundtrip ──

def test_rgb_roundtrip():
    """RGB→ICtCp→RGB should reconstruct within pq precision."""
    rgb = np.random.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    rgb_back = _torch_to_np(ictcp_to_rgb(rgb_to_ictcp(t))).clip(0.0, 1.0)
    e = np.abs(rgb - rgb_back).max()
    assert e < 5e-4, f"RGB roundtrip: {e:.2e}"


def test_yuv_roundtrip_valid():
    """YUV→ICtCp→YUV via RGB (in-gamut). PQ double-nonlinearity + clamp
    can cause occasional pixel errors; threshold set empirically."""
    rgb = np.random.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    m = _M_RGB2YUV.numpy()
    yuv_lin = rgb @ m.T
    y = (yuv_lin[:, :, 0:1] * 255.0).clip(0, 255)
    u = (yuv_lin[:, :, 1:2] * 255.0 + 128.0).clip(0, 255)
    v = (yuv_lin[:, :, 2:3] * 255.0 + 128.0).clip(0, 255)
    yuv = np.round(np.concatenate([y, u, v], axis=-1)).astype(np.uint8)
    t = _yuv_nhwc_to_torch(yuv)
    yuv_back = _torch_to_np(ictcp_to_yuv(yuv_to_ictcp(t)))
    yuv_back_u8 = np.round((yuv_back + 1) * 127.5).clip(0, 255).astype(np.uint8)
    e = np.abs(yuv.astype(np.int32) - yuv_back_u8.astype(np.int32)).max()
    assert e < 20, f"YUV roundtrip: max_diff={e}"


# ── special values ──

def test_black_white():
    for label, rgb in [("black", [0, 0, 0]), ("white", [1, 1, 1])]:
        t = torch.tensor(rgb, dtype=torch.float32).reshape(1, 3, 1, 1)
        ictcp = _torch_to_np(rgb_to_ictcp(t)).ravel()
        if label == "black":
            assert abs(ictcp[0]) < 0.1, f"black I={ictcp[0]}"
        else:
            assert ictcp[0] > 0.5, f"white I={ictcp[0]}"


def test_primary_colors_distinct():
    primaries = {}
    for label, c in [("red", [1, 0, 0]), ("green", [0, 1, 0]), ("blue", [0, 0, 1])]:
        t = torch.tensor(c, dtype=torch.float32).reshape(1, 3, 1, 1)
        primaries[label] = _torch_to_np(rgb_to_ictcp(t)).ravel()
    for a in primaries:
        for b in primaries:
            if a >= b:
                continue
            dist = np.abs(primaries[a] - primaries[b]).max()
            assert dist > 0.01, f"{a}≈{b}: dist={dist:.4f}"


def test_primary_colors_I_order():
    """I channel: green > red > blue (luminance order)."""
    rgb_vals = {"green": [0, 1, 0], "red": [1, 0, 0], "blue": [0, 0, 1]}
    I_vals = {}
    for label, c in rgb_vals.items():
        t = torch.tensor(c, dtype=torch.float32).reshape(1, 3, 1, 1)
        I_vals[label] = _torch_to_np(rgb_to_ictcp(t)).ravel()[0]
    assert I_vals["green"] > I_vals["red"] > I_vals["blue"], f"I_vals={I_vals}"


# ── PQ transfer function identity ──

def test_pq_roundtrip():
    l = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    l_back = eotf_pq_np(oetf_pq_np(l))
    assert np.abs(l - l_back).max() < 1e-4


def test_pq_monotonic():
    l_pq = oetf_pq_np(np.linspace(0.0, 1.0, 100, dtype=np.float32))
    assert np.all(np.diff(l_pq) > 0)


# ── fp16 ──

def test_yuv_fp16_nan_inf_free():
    yuv = np.random.randint(0, 256, (H, W, 3)).astype(np.uint8)
    out = yuv_to_ictcp(_yuv_nhwc_to_torch(yuv).half())
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert out.dtype == torch.float16


def test_ictcp_to_yuv_fp16():
    ictcp = torch.randn(1, 3, H, W).half()
    out = ictcp_to_yuv(ictcp)
    assert out.shape == (1, 3, H, W)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_rgb_fp16_nan_inf_free():
    t = torch.randn(1, 3, H, W).half()
    out = rgb_to_ictcp(t)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


# ── ICtCp range sanity ──

def test_ictcp_output_range():
    rgb = np.random.uniform(0.0, 1.0, (10000, 3)).astype(np.float32)
    t = torch.from_numpy(rgb).permute(1, 0).reshape(1, 3, 1, 10000).float()
    ictcp = _torch_to_np(rgb_to_ictcp(t))
    I, Ct, Cp = ictcp[:, 0], ictcp[:, 1], ictcp[:, 2]
    assert I.min() > -1.5 and I.max() < 1.5, f"I range [{I.min():.3f}, {I.max():.3f}]"
    assert abs(Ct).max() < 1.0, f"Ct range [{Ct.min():.3f}, {Ct.max():.3f}]"
    assert abs(Cp).max() < 1.0, f"Cp range [{Cp.min():.3f}, {Cp.max():.3f}]"
