import time
import sys
import math
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    video = 'data/validate/16079851_3840_2160_30fps.mp4'
    cache_root = Path('runs/validate/cache/16079851_3840_2160_30fps_3840x2160')
    hr_dir = cache_root / 'hr'
    lr_dir = cache_root / 'svt_crf40_p10' / 'lr_frames'
    output_path = 'runs/validate/16079851_onnx.mp4'

    hr_files = sorted(hr_dir.glob('*.png'))
    lr_files = sorted(lr_dir.glob('*.png'))
    n_frames = min(len(hr_files), len(lr_files))
    print(f'Frames: {n_frames}')

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print('Loading ONNX model...')
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
        }),
        'CPUExecutionProvider',
    ]
    session = ort.InferenceSession('runs/model.onnx', sess_options=so, providers=providers)
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, int(fps), (orig_w, orig_h))

    frames_lr = []
    frames_hr = []
    times = []
    psnrs = []
    warmup = 3

    print('Inference...')
    for i in range(n_frames):
        lr = cv2.cvtColor(cv2.imread(str(lr_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(cv2.imread(str(hr_files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        lr_t = (lr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        frames_lr.append(lr_t)
        frames_hr.append((hr.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))

        if len(frames_lr) < 3:
            continue

        if warmup > 0:
            session.run([output_name], {
                'prev': frames_lr[-3].astype(np.float32),
                'cur': frames_lr[-2].astype(np.float32),
                'next': frames_lr[-1].astype(np.float32),
            })
            warmup -= 1
            continue

        t0 = time.perf_counter()
        pred = session.run([output_name], {
            'prev': frames_lr[-3].astype(np.float32),
            'cur': frames_lr[-2].astype(np.float32),
            'next': frames_lr[-1].astype(np.float32),
        })[0]
        times.append((time.perf_counter() - t0) * 1000)

        pred = pred[0]
        hr_ref = frames_hr[i]
        if pred.shape != hr_ref.shape:
            import torch.nn.functional as F
            import torch
            hr_ref = F.interpolate(torch.from_numpy(hr_ref).unsqueeze(0), size=pred.shape[1:], mode='bicubic').squeeze(0).numpy()

        hr_ref = (hr_ref + 1) * 127.5
        pred_ref = (pred + 1) * 127.5
        mse = ((pred_ref - hr_ref) ** 2).mean()
        psnr = 20 * np.log10(255.0 / max(math.sqrt(mse), 1e-8))
        psnrs.append(psnr)

        pred_np = np.clip((pred.transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))

    writer.release()

    print()
    print('═══ ONNX Validation Report ═══')
    print(f'  Resolution: {orig_w}x{orig_h} (x4 via AV1 CRF40 p10)')
    print(f'  ---- Timing ----')
    print(f'  Avg latency: {np.mean(times):.0f} ms/frame')
    print(f'  FPS:         {1000/np.mean(times):.1f}')
    print(f'  ---- Quality ----')
    print(f'  PSNR:        {np.mean(psnrs):.2f}')
    print(f'  Frames:      {len(times)}')
    print(f'\nOutput: {output_path}')


if __name__ == '__main__':
    main()
