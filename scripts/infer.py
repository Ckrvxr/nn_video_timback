import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from core import ICtCpNetV2, ICtCpNet
from utils.colorspace import rgb_to_ictcp, ictcp_to_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', default='runs/prod/run_027/last.pkl')
    parser.add_argument('video', nargs='?', default=str(Path.home() / 'Downloads/ds2_clip_430_530.mp4'))
    parser.add_argument('--model', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--detail-path', action='store_true')
    parser.add_argument('--multi-frame', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == 'v2':
        model = ICtCpNetV2(scale=args.scale, detail_path=args.detail_path, multi_frame=args.multi_frame)
    else:
        model = ICtCpNet(scale=args.scale)

    key = jax.random.PRNGKey(0)
    if args.multi_frame:
        dummy = jnp.zeros((1, 512, 512, 9))
    else:
        dummy = jnp.zeros((1, 512, 512, 3))
    params = model.init(key, dummy)

    ckpt_path = Path(args.checkpoint)
    print(f'Loading checkpoint: {ckpt_path}')
    with open(ckpt_path, 'rb') as f:
        import pickle
        ckpt = pickle.load(f)
    ckpt_params = ckpt.get('params', ckpt)
    # Map v1 params to v2 model if needed
    if args.model == 'v2' and not any(k.startswith('i_branch') for k in jax.tree.leaves(ckpt_params)):
        print('WARNING: checkpoint does not contain v2 params, falling back to random init')
    else:
        params = ckpt_params
    print(f'Checkpoint epoch: {ckpt.get("epoch", "N/A")}')

    video_path = Path(args.video)
    out_path = video_path.parent / f'{video_path.stem}_enhanced{video_path.suffix}'

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Frames: {n_frames}, Size: {H}x{W}, FPS: {fps}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4

    apply_fn = jax.jit(lambda p, x: model.apply(p, x))

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        x = jnp.expand_dims(jnp.array(frame_rgb), 0)

        if pad_h or pad_w:
            x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))

        if args.multi_frame:
            x = jnp.concatenate([x, x, x], axis=-1)

        x_ictcp = rgb_to_ictcp(x)
        pred = apply_fn(params, x_ictcp)
        pred = pred[:, :H, :W, :]

        out_rgb = ictcp_to_rgb(pred)
        out_rgb = ((out_rgb + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out_frame = cv2.cvtColor(out_rgb[0].copy(), cv2.COLOR_RGB2BGR)
        writer.write(out_frame)

        if (i + 1) % 30 == 0:
            print(f'  processed {i+1}/{n_frames}')

    cap.release()
    writer.release()
    print(f'Done! Output: {out_path}')


if __name__ == '__main__':
    main()
