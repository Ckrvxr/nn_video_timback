import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.av1_vsr import AV1VSR


def parse_args():
    parser = argparse.ArgumentParser(description='Run AV1-VSR inference on video')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--scale', type=int, default=1, choices=[1, 2, 4])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tile', type=int, default=0, help='Tile size for large frames (0=disable)')
    return parser.parse_args()


def tiled_inference(model, frames, scale, tile_size, overlap, device):
    b, c, h, w = frames.shape
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(b, 3, out_h, out_w, device=device)
    weight = torch.zeros(b, 1, out_h, out_w, device=device)

    stride = tile_size - overlap
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile_in = frames[:, :, y:y_end, x:x_end]
            h_t, w_t = tile_in.shape[2:]
            h_to, w_to = h_t * scale, w_t * scale
            tile_out = model(tile_in[:, 0:1], tile_in[:, 1:2], tile_in[:, 2:3], scale=scale)
            output[:, :, y*scale:y*scale+h_to, x*scale:x*scale+w_to] += tile_out
            weight[:, :, y*scale:y*scale+h_to, x*scale:x*scale+w_to] += 1.0

    output = output / (weight + 1e-8)
    return output


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded checkpoint, scale={args.scale}x')

    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out, h_out = w_in * args.scale, h_in * args.scale

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w_out, h_out))

    buffer = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        buffer.append(frame_tensor.to(device))

        if len(buffer) < 3:
            if frame_idx == 0:
                buffer = [buffer[0]] * 2 + buffer
            elif frame_idx == 1:
                buffer.insert(1, buffer[0])

        if len(buffer) >= 3:
            f_prev, f_cur, f_next = buffer[-3], buffer[-2], buffer[-1]

            with torch.no_grad():
                pred = model(f_prev, f_cur, f_next, scale=args.scale)

            pred_img = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_img = np.clip((pred_img + 1) * 127.5, 0, 255).astype(np.uint8)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
            writer.write(pred_img)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f'Processed {frame_idx} frames')

    while len(buffer) > 3:
        buffer.pop(0)
        if len(buffer) == 3:
            f_prev, f_cur, f_next = buffer
            with torch.no_grad():
                pred = model(f_prev, f_cur, f_next, scale=args.scale)
            pred_img = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_img = np.clip((pred_img + 1) * 127.5, 0, 255).astype(np.uint8)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
            writer.write(pred_img)

    cap.release()
    writer.release()
    print(f'Done! Output: {args.output}')


if __name__ == '__main__':
    main()
