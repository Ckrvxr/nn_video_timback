import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import cv2
import numpy as np
import torch
from yaml import safe_load

from models import MambaFixer
from utils.video_loader import load_video_frames


def _yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    y = (x[:, 0:1] + 1) * 127.5
    u_centered = (x[:, 1:2] + 1) * 127.5 - 128.0
    v_centered = (x[:, 2:3] + 1) * 127.5 - 128.0
    r = y + 1.402 * v_centered
    g = y - 0.344 * u_centered - 0.714 * v_centered
    b = y + 1.772 * u_centered
    return torch.cat([r, g, b], dim=1) / 127.5 - 1.0


def _yuv_np_to_rgb_np(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def parse_args():
    parser = argparse.ArgumentParser(description='MambaFixer inference')
    parser.add_argument('checkpoint', type=str, help='Path to .pth checkpoint')
    parser.add_argument('input', type=str, help='Input compressed video (.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video path')
    parser.add_argument('--config', type=str, default='configs/mamba.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = safe_load(open(args.config))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = MambaFixer(
        num_features=config['model_architecture'].get('num_features', 16),
        state_dimension=config['model_architecture'].get('state_dimension', 16),
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    # Stream frames instead of loading all at once to avoid OOM
    with av.open(args.input) as container:
        stream = container.streams.video[0]
        fps = Fraction(stream.average_rate).limit_denominator(10000)
        h, w = stream.height, stream.width
        print(f'Input: {w}x{h}, {float(fps):.2f} fps')

        out_path = args.output or args.input.replace('.mp4', '_mamba_restored.mp4')
        container_out = av.open(out_path, mode='w')
        stream_out = container_out.add_stream('libx264', rate=fps)
        stream_out.width = w
        stream_out.height = h
        stream_out.pix_fmt = 'yuv444p'
        stream_out.options = {'crf': '18', 'preset': 'slow'}

        model.reset_state(1, device)

        for frame in container.decode(video=0):
            # Convert to YUV444
            y = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, frame.planes[0].line_size)[:, :frame.width]
            u = cv2.resize(np.frombuffer(frame.planes[1], np.uint8).reshape(frame.height//2, frame.planes[1].line_size)[:, :frame.width//2], (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(np.frombuffer(frame.planes[2], np.uint8).reshape(frame.height//2, frame.planes[2].line_size)[:, :frame.width//2], (w, h), interpolation=cv2.INTER_LINEAR)
            f_np = np.stack([y, u, v], axis=-1)
            
            f = torch.from_numpy(f_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
            pred = model(f)
            pred_np = (pred.squeeze(0).permute(1, 2, 0).cpu().float() + 1) * 127.5
            pred_np = pred_np.clamp(0, 255).numpy().astype(np.uint8)
            
            rgb = cv2.cvtColor(pred_np, cv2.COLOR_YUV2RGB)
            out_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream_out.encode(out_frame):
                container_out.mux(packet)

        for packet in stream_out.encode():
            container_out.mux(packet)
        container_out.close()
    print(f'Done → {out_path}')


if __name__ == '__main__':
    main()
