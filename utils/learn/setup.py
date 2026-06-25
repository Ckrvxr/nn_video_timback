import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from core import ICtCpNet, ICtCpNetV2
from utils.console import section, sub_section, metric, divider
from utils.loss.composite import CompositeLoss
from utils.learn.validate import validate


def build_model_and_optimizer(config):
    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']

    section("Training Setup")

    model_name = model_cfg.get('model_name', 'ArtRT_R1')
    sub_section("Model")
    metric("Name", model_name)

    # Create model
    scale = model_cfg.get('scale', 4)
    if model_name == 'ictcp':
        model = ICtCpNet(scale=scale)
    else:
        detail_path = model_cfg.get('detail_path', False)
        multi_frame = model_cfg.get('multi_frame', False)
        model = ICtCpNetV2(scale=scale, detail_path=detail_path, multi_frame=multi_frame)

    # Init params with dummy input
    key = jax.random.PRNGKey(42)
    in_ch = 9 if multi_frame else 3
    dummy = jnp.zeros((1, 512, 512, in_ch), dtype=jnp.float32)
    params = model.init(key, dummy)
    n_params = sum(p.size for p in jax.tree.leaves(params))
    metric("Parameters", f"{n_params:,}")

    # Build optimizer
    lr_cfg = training_cfg.get('lr')
    opt_lr = lr_cfg['warmup_peak'] if lr_cfg else training_cfg['learning_rate']

    optimizer = optax.chain(
        optax.clip_by_global_norm(training_cfg.get('gradient_clipping_threshold', 1.0)),
        optax.adamw(
            learning_rate=opt_lr,
            weight_decay=training_cfg.get('weight_decay', 0.01),
            b1=training_cfg.get('adam_beta1', 0.9),
            b2=training_cfg.get('adam_beta2', 0.999),
        ),
    )
    opt_state = optimizer.init(params)

    criterion = CompositeLoss(config['loss_weights'])

    sub_section("Training")
    if 'schedule' in config:
        config['loss_weights'] = config['schedule'][0]['weights'].copy()
    else:
        metric("Loss Weights", str(config['loss_weights']))
    metric("Batch Size", str(training_cfg['batch_size']))
    metric("Grad Accum", str(training_cfg.get('gradient_accumulation_steps', 1)))
    if lr_cfg:
        metric("LR Config", f"warmup_peak={lr_cfg['warmup_peak']:.2e}  true_peak={lr_cfg['true_peak']:.2e}  min={lr_cfg['min']:.2e}  warmup={lr_cfg['warmup_epochs']}ep")
    else:
        metric("Learning Rate", f"{opt_lr:.2e}")
    metric("Epochs", str(training_cfg['num_epochs']))

    return model, params, optimizer, opt_state, criterion


def build_lr_schedule(config):
    training_cfg = config['training_settings']
    lr_cfg = training_cfg.get('lr')
    n_epochs = training_cfg['num_epochs']

    if lr_cfg:
        warmup = lr_cfg['warmup_epochs']
        peak = lr_cfg['warmup_peak']
        true_peak = lr_cfg['true_peak']
        min_lr = lr_cfg['min']

        def schedule_fn(epoch):
            if epoch < warmup:
                return peak * (epoch + 1) / warmup
            frac = (epoch - warmup) / max(n_epochs - warmup, 1)
            cosine = 0.5 * (1 + jnp.cos(jnp.pi * frac))
            progress = jnp.clip(epoch / n_epochs, 0.0, 1.0)
            current_peak = peak + (true_peak - peak) * progress
            return min_lr + (current_peak - min_lr) * cosine

        return schedule_fn, n_epochs
    else:
        lr = training_cfg['learning_rate']
        return lambda _: lr, n_epochs


def load_checkpoint(args, params, opt_state):
    start_epoch = 0
    if args.resume:
        import pickle
        with open(args.resume, 'rb') as f:
            ckpt = pickle.load(f)
        params = ckpt['params']
        opt_state = ckpt['opt_state']
        start_epoch = ckpt['epoch'] + 1
    elif args.pretrained:
        import pickle
        with open(args.pretrained, 'rb') as f:
            ckpt = pickle.load(f)
        params = ckpt['params'] if 'params' in ckpt else ckpt
    return params, opt_state, start_epoch


def build_dataloaders(config):
    dataset_cfg = config['dataset']
    training_cfg = config['training_settings']
    model_cfg = config.get('model_architecture', {})

    sub_section("Data")
    metric("Training", f"{len(dataset_cfg['dataset_paths'])} sources")

    from utils.data.mkv_loader import discover_clips, load_mkv_batch
    from utils.data.prefetch import PrefetchIterator

    frames = model_cfg.get('multi_frame', False) * 2 + 1  # 1 or 3
    if frames > 1:
        metric("Multi-frame", f"{frames} frames")

    clips = discover_clips(dataset_cfg['dataset_paths'])
    total_frames = sum(c['n_frames'] for c in clips)
    bs = training_cfg['batch_size']
    n_batches = max(total_frames // bs, 1)
    metric("Batches/epoch", str(n_batches))

    _raw_factory = lambda: load_mkv_batch(
        dataset_cfg['dataset_paths'], bs, shuffle=True, frames=frames,
    )
    train_loader = PrefetchIterator(_raw_factory, n_workers=3, queue_size=6)

    val_clips = {}
    for vp in dataset_cfg.get('val_dataset_paths', []):
        name = Path(vp).name
        clips = discover_clips([vp])
        val_clips[name] = clips

    return train_loader, val_clips, n_batches


def compute_baseline(config, val_clips, model, params, logging_cfg, run_dir, output_dir):
    dataset_cfg = config['dataset']
    model_cfg = config['model_architecture']

    config_hash_payload = json.dumps({
        'dataset_paths': dataset_cfg.get('dataset_paths'),
        'val_dataset_paths': dataset_cfg.get('val_dataset_paths'),
        'patch_size': dataset_cfg.get('patch_size'),
        'model_name': model_cfg.get('model_name'),
        'scale': model_cfg.get('scale'),
    }, sort_keys=True)
    cfg_hash = hashlib.md5(config_hash_payload.encode()).hexdigest()[:8]
    baseline_path = output_dir / f'baseline_{cfg_hash}.json'

    if baseline_path.exists():
        baseline = json.load(open(baseline_path))
        sub_section("Baseline (cached)")
        for name, m in baseline.items():
            metric(name, f"psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  vmaf={m['vmaf']:.4f}")
    else:
        sub_section("Computing Baseline")
        baseline = {}
        val_batch_size = dataset_cfg.get('validation_batch_size', 2)
        for name, clips in val_clips.items():
            b_psnr, b_ssim, b_vmaf = validate(model, params, clips,
                                              val_batch_size, run_dir, name,
                                              baseline=True)
            baseline[name] = {'psnr': b_psnr, 'ssim': b_ssim, 'vmaf': b_vmaf}
            metric(name, f"psnr={b_psnr:.2f}  ssim={b_ssim:.4f}  vmaf={b_vmaf:.4f}")
        json.dump(baseline, open(baseline_path, 'w'))
    json.dump(baseline, open(run_dir / 'baseline.json', 'w'))
    divider()

    return baseline
