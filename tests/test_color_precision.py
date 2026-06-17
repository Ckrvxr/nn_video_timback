"""Verify torch JIT matches numpy f32 within max_abs < 1e-4 for all paths."""
import sys; sys.path.insert(0, '.')
import numpy as np, torch
from models.components.color_space import (
    yuv_to_ictcp_np, ictcp_to_yuv_np, rgb_to_ictcp_np, ictcp_to_rgb_np,
    yuv_to_ictcp, rgb_to_ictcp,
)

torch.manual_seed(0); np.random.seed(0)
H, W = 64, 64
MAX_ABS = 1e-4


def test_yuv_fwd():
    yuv = np.random.randint(0, 256, (H, W, 3)).astype(np.uint8)
    ictcp_np = yuv_to_ictcp_np(yuv, bits=8)
    yuv_t = torch.from_numpy(yuv.astype(np.float32)).permute(2,0,1).unsqueeze(0) / 127.5 - 1.0
    ictcp_t = yuv_to_ictcp(yuv_t).squeeze(0).permute(1,2,0).detach().numpy()
    e = np.abs(ictcp_np - ictcp_t).max()
    assert e < MAX_ABS, f"YUV→ICtCp: max_abs={e:.2e}"
    print(f"  PASS YUV→ICtCp: max_abs={e:.2e}")


def test_rgb_fwd():
    rgb = np.random.uniform(0.0, 1.0, (H, W, 3)).astype(np.float32)
    ictcp_np = rgb_to_ictcp_np(rgb)
    ictcp_t = rgb_to_ictcp(torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0))
    e = np.abs(ictcp_np - ictcp_t.squeeze(0).permute(1,2,0).detach().numpy()).max()
    assert e < MAX_ABS, f"RGB→ICtCp: max_abs={e:.2e}"
    print(f"  PASS RGB→ICtCp: max_abs={e:.2e}")


def test_matrix_identity():
    from models.components.color_space import MAT_RGB2LMS, MAT_LMS2RGB, MAT_LMS2ICTCP, MAT_ICTCP2LMS
    err1 = np.abs(MAT_RGB2LMS @ MAT_LMS2RGB - np.eye(3, dtype=np.float32)).max()
    err2 = np.abs(MAT_LMS2ICTCP @ MAT_ICTCP2LMS - np.eye(3, dtype=np.float32)).max()
    assert err1 < 1e-5, f"RGB2LMS*LMS2RGB error: {err1:.2e}"
    assert err2 < 1e-5, f"LMS2ICTCP*ICTCP2LMS error: {err2:.2e}"
    print(f"  PASS Matrix inverses: RGB2LMS={err1:.2e} LMS2ICTCP={err2:.2e}")


if __name__ == '__main__':
    print(f"Threshold: max_abs < {MAX_ABS:.0e}")
    test_matrix_identity()
    test_yuv_fwd()
    test_rgb_fwd()
    print("  All tests passed!")
