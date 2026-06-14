import av
import cv2
import numpy as np


def load_video_frames(video_path: str) -> list[np.ndarray]:
    """Decode video into list of YUV444 uint8 arrays (H×W×3).

    Y channel is native from decoder (lossless). U/V are bilinear
    upsampled from half-resolution YUV420 planes.
    """
    frames = []
    with av.open(video_path) as container:
        for frame in container.decode(video=0):
            h, w = frame.height, frame.width
            y = np.frombuffer(bytes(frame.planes[0]), dtype=np.uint8).reshape(h, w)
            u = cv2.resize(
                np.frombuffer(bytes(frame.planes[1]), dtype=np.uint8).reshape(h // 2, w // 2),
                (w, h), interpolation=cv2.INTER_CUBIC)
            v = cv2.resize(
                np.frombuffer(bytes(frame.planes[2]), dtype=np.uint8).reshape(h // 2, w // 2),
                (w, h), interpolation=cv2.INTER_CUBIC)
            frames.append(np.stack([y, u, v], axis=-1))
    return frames


def probe_frame_count(video_path: str) -> int:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        n = stream.frames
    return n
