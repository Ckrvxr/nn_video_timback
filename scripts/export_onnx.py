"""
Export AV1-VSR model to ONNX for TensorRT deployment.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.av1_vsr import AV1VSR


def parse_args():
    parser = argparse.ArgumentParser(description='Export AV1-VSR to ONNX')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--height', type=int, default=360, help='Input height for tracing')
    parser.add_argument('--width', type=int, default=640, help='Input width for tracing')
    parser.add_argument('--fp16', action='store_true', help='Export with FP16')
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_rcab=config['model']['n_rcab'],
        scales=[args.scale],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dummy_frames = torch.randn(1, 3, args.height, args.width, device=device)

    if args.fp16:
        model = model.half()
        dummy_frames = dummy_frames.half()

    output_path = args.output or f'av1_vsr_scale{args.scale}.onnx'

    torch.onnx.export(
        model,
        (dummy_frames, dummy_frames, dummy_frames, torch.tensor(args.scale)),
        output_path,
        input_names=['frame_prev', 'frame_cur', 'frame_next', 'scale'],
        output_names=['output'],
        dynamic_axes={
            'frame_prev': {2: 'height', 3: 'width'},
            'frame_cur': {2: 'height', 3: 'width'},
            'frame_next': {2: 'height', 3: 'width'},
            'output': {2: 'out_height', 3: 'out_width'},
        },
        opset_version=17,
    )

    print(f'ONNX exported to {output_path}')


if __name__ == '__main__':
    main()
