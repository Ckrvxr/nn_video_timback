import os
import re
import subprocess
import tempfile
from pathlib import Path

# Limit JAX GPU preallocation so tests with real video data fit.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from utils.colorspace.color_space import (
    MAT_BT2020_YUV2RGB,
    MAT_RGB2LMS,
    MAT_LMS2ICTCP,
    MAT_ICTCP2LMS,
    MAT_LMS2RGB,
    MAT_RGB2YUV,
    eotf_pq,
    eotf_pq_np,
    ictcp_to_yuv_np,
    oetf_pq,
    oetf_pq_np,
    yuv_to_ictcp_np,
)
from utils.data.mkv_loader import _decode_clip, _ffmpeg_to_yuv


_PEAK = 4095.0
_CENTER = 2048.0
# ~17% of pixels have RGB<0 after YUV→BT.2020→RGB (extreme chroma); those
# must be clipped to 0 before EOTF (PQ undefined for negatives).  This caps
# L2 PSNR at ≈21 dB for real video.
_PSNR_FLOOR = 18.0


# ── Helpers ───────────────────────────────────────────────────────────

def _yuv_norm(yuv: np.ndarray) -> np.ndarray:
    f = yuv.astype(np.float32)
    y = f[..., 0:1] / _PEAK
    u = (f[..., 1:2] - _CENTER) / _PEAK
    v = (f[..., 2:3] - _CENTER) / _PEAK
    return np.concatenate([y, u, v], axis=-1)


def _psnr(a: np.ndarray, b: np.ndarray, peak: float = _PEAK) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(20 * np.log10(peak / (np.sqrt(mse) + 1e-30)))


# ── Tests ─────────────────────────────────────────────────────────────


class TestPQFunctions:

    def test_eotf_oetf_roundtrip(self):
        v = np.linspace(0.0, 0.999, 1000, dtype=np.float32)
        lin = eotf_pq_np(v)
        v2 = oetf_pq_np(lin)
        err = np.abs(v - v2).max()
        assert err < 5e-5, f"EOTF/OETF roundtrip error {err:.2e}"

    def test_oetf_eotf_roundtrip(self):
        lin = np.logspace(-4, 0, 1000, dtype=np.float32)
        v = oetf_pq_np(lin)
        lin2 = eotf_pq_np(v)
        rel_err = np.abs(lin - lin2).max()
        assert rel_err < 2e-4, f"OETF/EOTF roundtrip error {rel_err:.2e}"

    def test_no_nan_at_boundary(self):
        v = np.array([0.0, 0.5, 0.99, 1.0, 1.5, 1.94, 1.97], dtype=np.float32)
        result = eotf_pq_np(v)
        assert np.all(np.isfinite(result)), "Non-finite EOTF at boundary"

    def test_oetf_large_input(self):
        lin = np.array([0.0, 1.0, 10.0, 1e3, 1e6], dtype=np.float32)
        result = oetf_pq_np(lin)
        assert np.all(np.isfinite(result)), "OETF produced NaN for large input"
        assert result[-1] <= 2.0, "OETF saturates below 2.0"


class TestMatrixRoundtrip:

    def test_yuv2rgb_then_rgb2yuv(self):
        yuv = np.random.randn(100, 3).astype(np.float32)
        rgb = yuv @ MAT_BT2020_YUV2RGB.T
        yuv2 = rgb @ np.linalg.inv(MAT_BT2020_YUV2RGB).T
        assert np.allclose(yuv, yuv2, atol=1e-6)

    def test_lms_ictcp_chain(self):
        lms = np.random.randn(100, 3).astype(np.float32)
        ictcp = lms @ MAT_LMS2ICTCP.T
        lms2 = ictcp @ MAT_ICTCP2LMS.T
        assert np.allclose(lms, lms2, atol=1e-6)


@pytest.fixture(scope="module")
def real_clip_dir():
    d = Path("datasets/val/alysa_liu_stateside_025x/00m21s_00m22s_h264_crf31_pfast_gop300_yuv420p")
    if not (d / "HR.mkv").exists():
        pytest.skip("real video data not available")
    return d


class TestFullRoundtrip:

    def test_synthetic_gray_roundtrip(self):
        rng = np.random.default_rng(42)
        yuv = np.full((1, 8, 8, 3), _CENTER, dtype=np.uint16)
        yuv[..., 0] = rng.integers(0, 4096, (1, 8, 8))
        ictcp = yuv_to_ictcp_np(yuv, bits=12)
        yuv2 = ictcp_to_yuv_np(ictcp, bits=12)
        err = np.abs(yuv.astype(np.float32) - yuv2.astype(np.float32))
        assert err.mean() < 50, f"Mean error {err.mean():.1f} >= 50"

    def test_black_white_roundtrip(self):
        for y_val in [0, _PEAK]:
            yuv = np.full((1, 4, 4, 3), _CENTER, dtype=np.uint16)
            yuv[..., 0] = y_val
            ictcp = yuv_to_ictcp_np(yuv, bits=12)
            yuv2 = ictcp_to_yuv_np(ictcp, bits=12)
            err = np.abs(yuv.astype(np.int32) - yuv2.astype(np.int32)).max()
            assert err <= 2, f"Luma={y_val} error {err} > 2"

    def test_real_video_roundtrip_cpu(self, real_clip_dir):
        yuv = _ffmpeg_to_yuv(real_clip_dir / "HR.mkv")
        ictcp = yuv_to_ictcp_np(yuv, bits=12)
        yuv2 = ictcp_to_yuv_np(ictcp, bits=12)
        psnr = _psnr(yuv, yuv2)
        assert psnr > _PSNR_FLOOR, f"CPU roundtrip PSNR {psnr:.2f} < {_PSNR_FLOOR}"

    def test_decoder_plus_validation(self, real_clip_dir):
        lr, hr = _decode_clip(real_clip_dir / "LR.mkv", real_clip_dir / "HR.mkv")
        assert np.all(np.isfinite(lr)), "LR ICtCp has NaN/Inf"
        assert np.all(np.isfinite(hr)), "HR ICtCp has NaN/Inf"
        yuv_pred = ictcp_to_yuv_np(hr[:4], bits=12)
        yuv_ref = _ffmpeg_to_yuv(real_clip_dir / "HR.mkv")[:4]
        psnr = _psnr(yuv_ref, yuv_pred)
        assert psnr > _PSNR_FLOOR, f"Round-trip PSNR {psnr:.2f} < {_PSNR_FLOOR}"


class TestDataLoaderIntegration:

    def test_batch_yield_shape(self):
        from utils.data.mkv_loader import load_mkv_batch
        paths = ["./datasets/val/alysa_liu_stateside_025x"]
        gen = load_mkv_batch(paths, batch_size=4, shuffle=False)
        batch = next(gen)
        lr_batch, hr_batch = batch
        assert lr_batch.shape[0] <= 4
        assert lr_batch.ndim == 4 and lr_batch.shape[-1] == 3
        assert np.all(np.isfinite(lr_batch)), "LR batch has NaN/Inf"
        assert np.all(np.isfinite(hr_batch)), "HR batch has NaN/Inf"


class TestMisc:

    def test_yuv_get_video_resolution(self):
        d = Path("datasets/val/alysa_liu_stateside_025x/00m21s_00m22s_h264_crf31_pfast_gop300_yuv420p")
        yuv = _ffmpeg_to_yuv(d / "HR.mkv")
        assert yuv.shape[1:] == (512, 512, 3)
        assert yuv.dtype == np.uint16
        assert yuv.min() >= 0 and yuv.max() <= 4095

    def test_no_crash_on_10bit_clip(self):
        d = Path("datasets/val/alysa_liu_stateside_025x/00m05s_00m07s_h265_crf27_pslow_gop96_yuv420p10le")
        if not (d / "HR.mkv").exists():
            pytest.skip("10-bit clip not present")
        yuv = _ffmpeg_to_yuv(d / "HR.mkv")
        assert yuv.shape[1:] == (512, 512, 3)
        ictcp = yuv_to_ictcp_np(yuv, bits=12)
        assert np.all(np.isfinite(ictcp))
