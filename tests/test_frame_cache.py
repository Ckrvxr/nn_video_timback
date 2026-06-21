"""Test LazyFrameRange live decode path (cache disabled)."""
import tempfile
from pathlib import Path

import numpy as np
import cv2

from tests.helpers import mock_mamba_ssm
mock_mamba_ssm()

from utils.data.frame_cache import LazyFrameRange


def _create_dummy_video(path: Path, num_frames=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, 30, (320, 240))
    for i in range(num_frames):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(img, (160, 120), 100, (255, 128, 64), -1)
        writer.write(img)
    writer.release()


def test_lazy_live_decode():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / "test_video.mp4"
        _create_dummy_video(video_path, num_frames=10)

        lr = LazyFrameRange(str(video_path), 10, window_size=5)

        frames = [lr[i] for i in range(10)]
        assert len(frames) == 10
        for f in frames:
            assert f.shape == (240, 320, 3)
            assert f.dtype == np.float32
            assert f[:, :, 0].min() >= -0.05 and f[:, :, 0].max() <= 1.05

        f0 = lr[0]
        assert f0.shape == (240, 320, 3)
