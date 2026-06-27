import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from core.torch import ArtRT
from utils.console import console, section, sub_section, metric, detail, divider
from utils.loss.torch_composite import CompositeLoss


def _fmt(v):
    return f'{v:.8f}'.rstrip('0').rstrip('.')


def build_model_and_optimizer(config):
    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']

    section("Training Setup")

    model_name = model_cfg.get('model_name', 'ArtRT')
    sub_section("Model")
    metric("Name", model_name)

    dim = model_cfg.get('dim', 48)
    n1 = model_cfg.get('n1', 6)
    n2 = model_cfg.get('n2', 8)
    n3 = model_cfg.get('n3', 4)
    nmid = model_cfg.get('nmid', 4)

    model = ArtRT(dim=dim, n1=n1, n2=n2, n3=n3, nmid=nmid)
    n_params = sum(p.numel() for p in model.parameters())
    metric("Parameters", f"{n_params:,}")

    optimizer_name = training_cfg.get('optimizer', 'prodigy').lower()
    opt_cfg = training_cfg[optimizer_name]

    def _as_float(v, default=0.0):
        return float(v) if v is not None else default

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=_as_float(opt_cfg.get('lr'), 5e-5),
            betas=opt_cfg.get('betas', (0.9, 0.999)),
            weight_decay=_as_float(opt_cfg.get('weight_decay'), 0.01),
        )
    elif optimizer_name == 'prodigy':
        import prodigyopt
        optimizer = prodigyopt.Prodigy(
            model.parameters(),
            lr=_as_float(opt_cfg.get('lr'), 1.0),
            betas=opt_cfg.get('betas', (0.9, 0.999)),
            weight_decay=_as_float(opt_cfg.get('weight_decay'), 0.0),
            beta3=_as_float(opt_cfg.get('beta3'), 0.99),
            d0=_as_float(opt_cfg.get('d0'), 1e-6),
            safeguard_warmup=bool(opt_cfg.get('safeguard_warmup', True)),
        )
    elif optimizer_name == 'sophia':
        from core.optim.sophia import SophiaG
        optimizer = SophiaG(
            model.parameters(),
            lr=_as_float(opt_cfg.get('lr'), 1e-4),
            betas=opt_cfg.get('betas', (0.9, 0.95)),
            rho=_as_float(opt_cfg.get('rho'), 0.04),
            weight_decay=_as_float(opt_cfg.get('weight_decay'), 1e-1),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    schedule = config.get('schedule', [])
    if schedule:
        initial_weights = dict(schedule[0]['weights'])
    else:
        initial_weights = {'charbonnier': 1.0, 'fft': 0.0}
    criterion = CompositeLoss(initial_weights)

    sub_section("Training")
    loss_str = "  ".join(f"{k}={_fmt(v)}" for k, v in initial_weights.items())
    metric("Loss Weights", loss_str)
    metric("Batch Size", str(training_cfg['batch_size']))
    metric("Grad Accum", str(opt_cfg.get('gradient_accumulation_steps', 1)))
    metric("Optimizer", optimizer_name.capitalize())
    metric("Learning Rate", f"{opt_cfg.get('lr', 1.0):.2e}")
    metric("Epochs", str(training_cfg['num_epochs']))

    return model, optimizer, criterion, opt_cfg


def build_lr_schedule(opt_cfg, config):
    training_cfg = config['training_settings']
    n_epochs = training_cfg['num_epochs']
    lr = opt_cfg['lr']
    return lambda _: lr, n_epochs


class MKVIterableDataset(IterableDataset):
    def __init__(self, clip_list, batch_size, prefetch=8, shuffle=True, mirror=False):
        self.clip_list = clip_list
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.shuffle = shuffle
        self.mirror = mirror

    def __iter__(self):
        import queue
        import threading
        q = queue.Queue(maxsize=self.prefetch)
        sentinel = object()

        def worker():
            try:
                for batch in self._generator():
                    q.put(batch)
            except Exception as e:
                q.put(e)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def _generator(self):
        import torch
        from utils.data.mkv_loader import decode_yuv
        from utils.colorspace.color_space import yuv_to_rgb_linear_cuda

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clips = list(self.clip_list)
        if self.shuffle:
            import random
            random.shuffle(clips)

        bs = self.batch_size

        for clip in clips:
            lr_yuv, hr_yuv = decode_yuv(clip['lr_path'], clip['hr_path'])
            n = lr_yuv.shape[0]
            lr_buf = torch.empty(bs, *lr_yuv.shape[1:], dtype=torch.uint16, pin_memory=True)
            hr_buf = torch.empty(bs, *hr_yuv.shape[1:], dtype=torch.uint16, pin_memory=True)
            lr_gpu = torch.empty(bs, *lr_yuv.shape[1:], dtype=torch.uint16, device=device)
            hr_gpu = torch.empty(bs, *hr_yuv.shape[1:], dtype=torch.uint16, device=device)

            for start in range(0, n, bs):
                end = min(start + bs, n)
                actual_bs = end - start
                lr_buf[:actual_bs] = torch.from_numpy(lr_yuv[start:end].copy())
                hr_buf[:actual_bs] = torch.from_numpy(hr_yuv[start:end].copy())
                lr_gpu[:actual_bs].copy_(lr_buf[:actual_bs], non_blocking=True)
                hr_gpu[:actual_bs].copy_(hr_buf[:actual_bs], non_blocking=True)
                lr = yuv_to_rgb_linear_cuda(lr_gpu[:actual_bs])
                hr = yuv_to_rgb_linear_cuda(hr_gpu[:actual_bs])
                torch.cuda.current_stream().synchronize()
                yield lr, hr
                if self.mirror:
                    yield lr.flip(-1), hr.flip(-1)


def build_dataloaders(config):
    dataset_cfg = config['dataset']
    training_cfg = config['training_settings']

    sub_section("Data")

    from utils.data.mkv_loader import discover_clips

    mirror = dataset_cfg.get('mirror', False)

    clips = discover_clips(dataset_cfg['dataset_paths'])
    total_frames = sum(c['n_frames'] for c in clips)
    bs = training_cfg['batch_size']
    n_batches = max(total_frames // bs, 1)
    if mirror:
        n_batches *= 2

    metric("Training", f"{len(dataset_cfg['dataset_paths'])} sources")
    for p in dataset_cfg['dataset_paths']:
        name = Path(p).name or p
        n_clips = len(discover_clips([p]))
        detail(name, f"{n_clips} clips")
    metric("Total frames", f"{total_frames:,}")
    metric("Batch size", str(bs))
    metric("Batches/epoch", str(n_batches))

    dataset = MKVIterableDataset(clips, bs, shuffle=True, mirror=mirror)
    if mirror:
        metric("Mirror", "horizontal flip (2× data)")

    n_workers = dataset_cfg.get('num_workers', 0)

    def make_train_loader():
        return DataLoader(dataset, batch_size=None, num_workers=n_workers)

    val_clips = {}
    val_paths = dataset_cfg.get('val_dataset_paths', [])
    if val_paths:
        metric("Validation", f"{len(val_paths)} sources")
        for vp in val_paths:
            name = Path(vp).name
            clips = discover_clips([vp])
            val_clips[name] = clips
            detail(name, f"{len(clips)} clips")

    return make_train_loader, val_clips, n_batches


def load_checkpoint(args, model, optimizer):
    start_epoch = 0
    if args.resume:
        from utils.learn.io import load_checkpoint as _load
        ckpt = _load(args.resume)
        model.load_state_dict(ckpt.get('params', ckpt.get('model', ckpt)))
        if 'opt_state' in ckpt:
            optimizer.load_state_dict(ckpt['opt_state'])
        elif 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', -1) + 1
    elif args.pretrained:
        from utils.learn.io import load_checkpoint as _load
        ckpt = _load(args.pretrained)
        state = ckpt.get('params', ckpt.get('model', ckpt))
        if isinstance(state, dict):
            model.load_state_dict(state)
    return model, optimizer, start_epoch


def compute_baseline(config, val_clips, logging_cfg, run_dir, output_dir):
    dataset_cfg = config['dataset']
    model_cfg = config['model_architecture']

    config_hash_payload = json.dumps({
        'dataset_paths': dataset_cfg.get('dataset_paths'),
        'val_dataset_paths': dataset_cfg.get('val_dataset_paths'),
        'patch_size': dataset_cfg.get('patch_size'),
        'model_name': model_cfg.get('model_name'),
        'd_model': model_cfg.get('d_model'),
        'd_state': model_cfg.get('d_state'),
    }, sort_keys=True)
    cfg_hash = hashlib.md5(config_hash_payload.encode()).hexdigest()[:8]
    baseline_path = output_dir / f'baseline_{cfg_hash}.json'

    if baseline_path.exists():
        baseline = json.load(open(baseline_path))
        sub_section("Baseline (cached)")
        for name, m in baseline.items():
            vgg_str = f"  vgg={m['vgg']:.6f}" if 'vgg' in m else ""
            metric(name, f"psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  vmaf={m['vmaf']:.4f}{vgg_str}")
    else:
        from utils.learn.torch_validate import baseline_clip
        sub_section("Computing Baseline")
        baseline = {}
        for name, clips in val_clips.items():
            total = {'psnr': 0.0, 'ssim': 0.0, 'vmaf': 0.0, 'vgg': 0.0}
            n = 0
            for clip in clips:
                try:
                    m = baseline_clip(clip)
                    for k in total:
                        total[k] += m[k]
                    n += 1
                except Exception as e:
                    console.error(f"Baseline failed for {clip.get('clip_name', '?')}: {e}")
            if n > 0:
                b_psnr = total['psnr'] / n
                b_ssim = total['ssim'] / n
                b_vmaf = total['vmaf'] / n
                b_vgg = total.get('vgg', 0.0) / n
            else:
                b_psnr = b_ssim = b_vmaf = b_vgg = float('nan')
            baseline[name] = {'psnr': b_psnr, 'ssim': b_ssim, 'vmaf': b_vmaf, 'vgg': b_vgg}
            metric(name, f"psnr={b_psnr:.2f}  ssim={b_ssim:.4f}  vmaf={b_vmaf:.4f}  vgg={b_vgg:.6f}")
        json.dump(baseline, open(baseline_path, 'w'))
    json.dump(baseline, open(run_dir / 'baseline.json', 'w'))
    divider()

    return baseline
