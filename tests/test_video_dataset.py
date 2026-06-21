"""Tests for video dataset pipeline: frame cache, ICtCp, crop, DataLoader, model forward."""
from pathlib import Path

import numpy as np
import torch
import pytest

from tests.helpers import mock_mamba_ssm
mock_mamba_ssm()

from utils.data.video_loader import probe_frame_count, probe_resolution
from utils.data.frame_cache import LazyFrameRange
from utils.data.dataset import CompressedVideoDataset, VideoBatchSampler, collate_video, _yuv_to_ictcp
from torch.utils.data import DataLoader
from models import MambaFixer
from utils.training.losses.composite import CompositeLoss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_VIDEO = PROJECT_ROOT / 'data' / 'val' / 'HR' / 'Hoppers_2026_seg17.mkv'
VAL_DIR = PROJECT_ROOT / 'data' / 'val'
HAS_REAL_VIDEO = VAL_VIDEO.exists()

# ── Mock-level tests (no real video files) ───────────────────────


def test_yuv_to_ictcp_float16():
    fake_yuv = np.random.randint(0, 256, (1, 64, 64, 3)).astype(np.uint16)
    fake_yuv[..., 0] = 256
    result = _yuv_to_ictcp(fake_yuv)
    assert result.dtype == np.float16


def test_model_forward_shape(device):
    x = torch.randn((4, 3, 512, 512), device=device)
    model = MambaFixer(64, 32, 2, 4, 2, [1, 2, 4]).to(device)
    model.reset_state(4, device)
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (4, 3, 512, 512)


def test_loss_backward(device):
    model = MambaFixer(64, 32, 2, 4, 2, [1, 2, 4]).to(device)
    x = torch.randn((4, 3, 512, 512), device=device, requires_grad=True)
    criterion = CompositeLoss({'charbonnier': 1.0, 'wavelet': 0.5}, device=device)
    loss_dict = criterion(x, x)
    loss_dict['total'].backward()
    assert x.grad is not None


# ── Real video tests (skipped if data files missing) ─────────────


@pytest.mark.skipif(not HAS_REAL_VIDEO, reason="requires data/val/ video files")
class TestFrameCache:
    def test_probe_frame_count(self):
        n = probe_frame_count(str(VAL_VIDEO))
        assert n == 87

    def test_frame_cache_shape(self):
        h, w = probe_resolution(str(VAL_VIDEO))
        fr = LazyFrameRange(str(VAL_VIDEO), 87, window_size=9)
        f0 = fr[0]
        assert f0.shape == (h, w, 3)

    def test_incremental_decode(self):
        fr = LazyFrameRange(str(VAL_VIDEO), 87, window_size=9)
        cache_sizes = []
        for i in range(12):
            _ = fr[i]
            cache_sizes.append(len(fr._cache))
        assert cache_sizes[0] == 9
        growths = [cache_sizes[i] - cache_sizes[i - 1] for i in range(1, len(cache_sizes))]
        assert max(growths[1:]) == 0

    def test_fp16_cache_dtype(self):
        fr = LazyFrameRange(str(VAL_VIDEO), 87, window_size=9)
        _ = fr[0]
        assert fr._cache[0].dtype == np.float16


@pytest.mark.skipif(not HAS_REAL_VIDEO, reason="requires data/val/ video files")
class TestDatasetPipeline:
    def test_frame_cache_yuv_uint16(self):
        fr = LazyFrameRange(str(VAL_VIDEO), 87, window_size=9)
        f = fr[0]
        assert f.dtype == np.uint16

    def test_crop_first_ictcp(self):
        fr = LazyFrameRange(str(VAL_VIDEO), 87, window_size=9)
        f = fr[0]
        yuv_crop = f[0:512, 0:512]
        ictcp = _yuv_to_ictcp(yuv_crop[np.newaxis, ...])
        assert ictcp.dtype == np.float16

    def test_dataset_getitem_float32(self):
        ds = CompressedVideoDataset(
            datasets=[str(VAL_DIR)], patch_size=512, frames=9, is_train=True)
        item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40,
                ds.videos[0]['ds_root'], 0, 0)
        out = ds[item]
        assert out['lr_frames'].dtype == torch.float32

    def test_grid_cell_count(self):
        ds = CompressedVideoDataset(
            datasets=[str(VAL_DIR)], patch_size=512, frames=9, is_train=True)
        cells = (ds.videos[0]['h'] // 512) * (ds.videos[0]['w'] // 512)
        assert cells == 28

    def test_dataloader_batch_shape(self):
        ds = CompressedVideoDataset(
            datasets=[str(VAL_DIR)], patch_size=512, frames=9, is_train=True)
        sampler = VideoBatchSampler(ds, batch_size=4, clip_repeat=1)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_video, num_workers=0)
        b = next(iter(loader))
        assert b['lr_frames'].shape == (4, 9, 3, 512, 512)
        assert b['hr'].shape == (4, 3, 512, 512)

    def test_e2e_model_forward(self, device):
        ds = CompressedVideoDataset(
            datasets=[str(VAL_DIR)], patch_size=512, frames=9, is_train=True)
        item = (ds.videos[0]['name'], ds.videos[0]['variants'][0], 40,
                ds.videos[0]['ds_root'], 0, 0)
        out = ds[item]
        model = MambaFixer(64, 32, 2, 100, 2, [1, 2, 4, 32]).to(device).eval()
        x = out['hr'].unsqueeze(0).to(device)
        model.reset_state(1, device)
        with torch.no_grad():
            p = model(x)
        assert p.shape == (1, 3, 512, 512)
