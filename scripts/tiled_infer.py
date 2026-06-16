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
from models.components import yuv_to_ictcp, ictcp_to_yuv


@torch.no_grad()
def tiled_mamba_inference(model, x, tile_size=1024, overlap=64):
    ictcp = yuv_to_ictcp(x)
    # 1. Global SSM pass on downsampled frame (to maintain sequence state correctly)
    z_t = model.forward_ssm_ictcp(ictcp)
    idx, _ = model.router(z_t)
    
    # 2. Tiled Expert application (where the OOM usually happens)
    B, C, H, W = ictcp.shape
    output = torch.zeros_like(ictcp)
    
    for y in range(0, H, tile_size - overlap):
        for x_tile in range(0, W, tile_size - overlap):
            y_end = min(y + tile_size, H)
            x_end = min(x_tile + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = ictcp[:, :, y_start:y_end, x_start:x_end]
            tile_out = model.apply_experts(tile, idx)
            
            # Stitching with half-overlap crop
            out_y_start = y_start + (overlap // 2 if y_start > 0 else 0)
            out_x_start = x_start + (overlap // 2 if x_start > 0 else 0)
            out_y_end = y_end - (overlap // 2 if y_end < H else 0)
            out_x_end = x_end - (overlap // 2 if x_end < W else 0)
            
            tile_y_start = overlap // 2 if y_start > 0 else 0
            tile_x_start = overlap // 2 if x_start > 0 else 0
            tile_y_end = (y_end - y_start) - (overlap // 2 if y_end < H else 0)
            tile_x_end = (x_end - x_start) - (overlap // 2 if x_end < W else 0)
            
            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                tile_out[:, :, tile_y_start:tile_y_end, tile_x_start:tile_x_end]
                
    return ictcp_to_yuv(output)


def parse_args():
    parser = argparse.ArgumentParser(description='Tiled MambaFixer inference')
    parser.add_argument('checkpoint', type=str, help='Path to .pth checkpoint')
    parser.add_argument('input', type=str, help='Input compressed video (.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video path')
    parser.add_argument('--config', type=str, default='configs/mamba.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=64)
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

    with av.open(args.input) as container:
        stream = container.streams.video[0]
        fps = Fraction(stream.average_rate).limit_denominator(10000)
        h, w = stream.height, stream.width
        print(f'Input: {w}x{h}, {float(fps):.2f} fps')

        out_path = args.output or args.input.replace('.mp4', '_tiled_mamba.mp4')
        container_out = av.open(out_path, mode='w')
        stream_out = container_out.add_stream('libx264', rate=fps)
        stream_out.width = w
        stream_out.height = h
        stream_out.pix_fmt = 'yuv444p'
        stream_out.options = {'crf': '18', 'preset': 'slow'}

        model.reset_state(1, device)

        for i, frame in enumerate(container.decode(video=0)):
            # Convert to YUV444
            y = np.frombuffer(frame.planes[0], np.uint8).reshape(frame.height, frame.planes[0].line_size)[:, :frame.width]
            u = cv2.resize(np.frombuffer(frame.planes[1], np.uint8).reshape(frame.height//2, frame.planes[1].line_size)[:, :frame.width//2], (w, h), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(np.frombuffer(frame.planes[2], np.uint8).reshape(frame.height//2, frame.planes[2].line_size)[:, :frame.width//2], (w, h), interpolation=cv2.INTER_LINEAR)
            f_np = np.stack([y, u, v], axis=-1)
            
            f = torch.from_numpy(f_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
            
            # Tiled inference
            if w > args.tile_size or h > args.tile_size:
                pred = tiled_mamba_inference(model, f, tile_size=args.tile_size, overlap=args.overlap)
            else:
                pred = model(f)
                
            pred_np = (pred.squeeze(0).permute(1, 2, 0).cpu().float() + 1) * 127.5
            pred_np = pred_np.clamp(0, 255).numpy().astype(np.uint8)
            
            rgb = cv2.cvtColor(pred_np, cv2.COLOR_YUV2RGB)
            out_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream_out.encode(out_frame):
                container_out.mux(packet)
            
            if (i + 1) % 10 == 0:
                print(f'Processed {i+1} frames')

        for packet in stream_out.encode():
            container_out.mux(packet)
        container_out.close()
    print(f'Done → {out_path}')


if __name__ == '__main__':
    main()
