"""Test live decode path via LazyFrameRange (cache disabled)."""
import sys
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

import numpy as np
import cv2

# Full mock of mamba_ssm to avoid GPU deps
_sm = MagicMock()
sys.modules['mamba_ssm'] = _sm
sys.modules['mamba_ssm.ops'] = MagicMock()
sys.modules['mamba_ssm.ops.selective_scan_interface'] = _sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.frame_cache import LazyFrameRange


def create_dummy_video(path: Path, width=320, height=240, num_frames=10, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(img, (width // 2 + int(50 * np.sin(i / 5)), height // 2), 100, (255, 128, 64), -1)
        cv2.putText(img, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(img)
    writer.release()


def test_lazy_live_decode():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_path = tmp_path / "test_video.mp4"
        create_dummy_video(video_path, width=320, height=240, num_frames=10)

        lr = LazyFrameRange(str(video_path), 10, window_size=5)

        # Sequential access
        frames = [lr[i] for i in range(10)]
        assert len(frames) == 10
        for f in frames:
            assert f.shape == (240, 320, 3)
            assert f.dtype == np.float32
            # ICtCp: I in [0,1], Ct/Cp in [-0.5, 0.5]
            assert f[:,:,0].min() >= -0.05 and f[:,:,0].max() <= 1.05

        # Random access (should hit cache)
        f0 = lr[0]
        assert f0.shape == (240, 320, 3)

        print(f"Live decode test passed! {len(frames)} frames, "
              f"I range [{frames[0][:,:,0].min():.4f}, {frames[0][:,:,0].max():.4f}]")


if __name__ == '__main__':
    test_lazy_live_decode()
