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

from models import HyperFixer
from utils.video_loader import load_video_frames


def parse_args():
    parser = argparse.ArgumentParser(description='HyperFixer inference')
    parser.add_argument('checkpoint', type=str, help='Path to .pth checkpoint')
    parser.add_argument('input', type=str, help='Input compressed video (.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video path')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fp16', action='store_true', default=True, dest='fp16')
    parser.add_argument('--no-fp16', action='store_false', dest='fp16')
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = safe_load(open(args.config))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = HyperFixer(
        num_features=config['model_architecture']['num_features'],
        num_blocks=config['model_architecture']['num_blocks'],
        latent_dimension=config['model_architecture'].get('latent_dimension', 256),
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    if args.fp16 and device.type == 'cuda':
        model = model.half()

    frames = load_video_frames(args.input)
    h, w = frames[0].shape[:2]
    n_frames = len(frames)
    fps = Fraction(30, 1)
    with av.open(args.input) as container:
        stream = container.streams.video[0]
        fps = Fraction(stream.average_rate).limit_denominator(10000)
    print(f'Input: {n_frames} frames, {w}x{h}, {float(fps):.2f} fps')

    out_path = args.output or args.input.replace('.mp4', '_restored.mp4')
    container_out = av.open(out_path, mode='w')
    stream_out = container_out.add_stream('libx264', rate=fps)
    stream_out.width = w
    stream_out.height = h
    stream_out.pix_fmt = 'yuv444p'
    stream_out.options = {'crf': '18', 'preset': 'slow'}

    out_frames = [None] * n_frames
    out_frames[0] = frames[0]
    if n_frames > 1:
        out_frames[-1] = frames[-1]

    for i in range(1, n_frames - 1):
        f_prev = torch.from_numpy(frames[i - 1]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
        f_cur = torch.from_numpy(frames[i]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
        f_next = torch.from_numpy(frames[i + 1]).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0

        if args.fp16:
            f_prev = f_prev.half()
            f_cur = f_cur.half()
            f_next = f_next.half()

        pred = model(f_prev, f_cur, f_next)
        pred_np = (pred.squeeze(0).permute(1, 2, 0).cpu().float() + 1) * 127.5
        pred_np = pred_np.clamp(0, 255).numpy().astype(np.uint8)
        out_frames[i] = pred_np

        if (i + 1) % 50 == 0 or i == n_frames - 2:
            print(f'  {i + 1}/{n_frames}')

    for arr in out_frames:
        rgb = cv2.cvtColor(arr, cv2.COLOR_YUV2RGB)
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        for packet in stream_out.encode(frame):
            container_out.mux(packet)
    for packet in stream_out.encode():
        container_out.mux(packet)
    container_out.close()

    print(f'Done → {out_path}')


if __name__ == '__main__':
    main()
