import argparse
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
import numpy as np
import torch
from yaml import safe_load

from models import MambaFixer
from models.components import yuv_to_ictcp
from utils.video_loader import frame_to_yuv


def parse_args():
    parser = argparse.ArgumentParser(description='MambaFixer inference')
    parser.add_argument('checkpoint', type=str, help='Path to .pth checkpoint')
    parser.add_argument('input', type=str, help='Input compressed video')
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

    arch_cfg = config['model_architecture']
    model = MambaFixer(
        num_features=arch_cfg.get('num_features', 64),
        state_dimension=arch_cfg.get('state_dimension', 32),
        num_features_stream=arch_cfg.get('num_features_stream', 2),
        num_experts=arch_cfg.get('num_experts', 100),
        n_active=arch_cfg.get('n_active', 2),
        dilation_rates=arch_cfg.get('dilation_rates', [1, 2, 4, 32]),
        routing_threshold=arch_cfg.get('routing_threshold', 0.90)
    ).to(device)
    model.eval()

    use_half = config.get('inference_settings', {}).get('use_half_precision', True)
    if use_half:
        model = model.half()

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    ckpt.pop('_t_state', None)
    model.load_state_dict(ckpt, strict=False)

    input_path = Path(args.input)
    # Pre-convert to yuv444p so chroma upsampling matches training pipeline
    tmp_input = Path(tempfile.mktemp(suffix='.mkv'))
    print('Converting to yuv444p...')
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_path),
        '-vf', 'format=yuv444p',
        '-c:v', 'ffv1', '-an',
        str(tmp_input),
    ], check=True, capture_output=True)

    try:
        with av.open(str(tmp_input)) as container:
            stream = container.streams.video[0]
            fps = Fraction(stream.average_rate).limit_denominator(10000)
            h, w = stream.height, stream.width
            print(f'Input: {w}x{h}, {float(fps):.2f} fps')

            out_path = args.output or str(input_path.with_name(input_path.stem + '_ictcp.mkv'))
            container_out = av.open(out_path, mode='w')
            stream_out = container_out.add_stream('ffv1', rate=fps)
            stream_out.width = w
            stream_out.height = h
            stream_out.pix_fmt = 'yuv444p'
            stream_out.color_primaries = 9
            stream_out.color_trc = 16
            stream_out.colorspace = 14
            stream_out.color_range = 1

            model.reset_state(1, device)

            for frame in container.decode(video=0):
                f_np = frame_to_yuv(frame)

                if f_np.dtype == np.uint16:
                    max_val = int(f_np.max())
                    bits = 12 if max_val > 1023 else 10
                    peak = float((1 << bits) - 1)
                    center = float(1 << (bits - 1))
                    f = torch.from_numpy(f_np.astype(np.float32, copy=False))
                    f = f.permute(2, 0, 1).unsqueeze(0).to(device)
                    f[:, 0:1] = f[:, 0:1] / peak * 255.0
                    f[:, 1:] = (f[:, 1:] - center) / peak * 255.0 + 128.0
                    f = f / 127.5 - 1.0
                else:
                    f = torch.from_numpy(f_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0

                if use_half:
                    f = f.half()

                pred_ictcp = model(yuv_to_ictcp(f))

                i = pred_ictcp[:, 0:1] * 255.0
                ct = pred_ictcp[:, 1:2] * 255.0 + 128.0
                cp = pred_ictcp[:, 2:3] * 255.0 + 128.0
                out = torch.cat([i, ct, cp], dim=1).clamp(0, 255).squeeze(0).cpu().float().numpy().astype(np.uint8)

                out_frame = av.VideoFrame(width=w, height=h, format='yuv444p')
                out_frame.pict_type = av.video.frame.PictureType.I
                out_frame.planes[0].update(out[0])
                out_frame.planes[1].update(out[1])
                out_frame.planes[2].update(out[2])
                for packet in stream_out.encode(out_frame):
                    container_out.mux(packet)

            for packet in stream_out.encode():
                container_out.mux(packet)
            container_out.close()
        print(f'Done \u2192 {out_path}')
    finally:
        tmp_input.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
