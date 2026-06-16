import tempfile
from pathlib import Path
import numpy as np
import cv2
import sys
from unittest.mock import MagicMock

# Mock Triton for Windows / non-Triton environments
class TritonMock(MagicMock):
    @classmethod
    def jit(cls, *args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(f):
            return f
        return decorator

sys.modules['triton'] = TritonMock()
sys.modules['triton.language'] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.blosc_cache import BloscCache  # noqa: E402
from utils.video_loader import load_video_frames_raw  # noqa: E402

def create_dummy_video(path: Path, width=1280, height=720, num_frames=30, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(img, (width // 2 + int(50 * np.sin(i / 5)), height // 2), 100, (255, 128, 64), -1)
        cv2.putText(img, f"Frame {i}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(img)
    writer.release()

def test_cache_yuv():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_path = tmp_path / "test_video.mp4"
        cache_root = tmp_path / "cache_yuv"
        
        # Create a dummy video
        create_dummy_video(video_path, width=1280, height=720, num_frames=30)
        
        # Load raw frames
        raw_frames = load_video_frames_raw(str(video_path))
        
        # Write to BloscCache
        blc = BloscCache(cache_root)
        blc.put_raw(video_path, raw_frames)
        
        # Check cache file exists
        cp = blc.cache_path(video_path)
        assert cp.exists()
        assert cp.stat().st_size > 0
        
        # Read from BloscCache
        res = blc.get(video_path)
        assert res is not None
        arr, h, w, n_frames = res
        assert h == 720
        assert w == 1280
        assert n_frames == 30
        
        print(f"YUV Cache created successfully! Size: {cp.stat().st_size} bytes")

if __name__ == "__main__":
    test_cache_yuv()
