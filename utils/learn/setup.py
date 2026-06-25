import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from core import ICtCpNetV2
from utils.console import console, section, sub_section, metric, divider
from utils.loss.composite import CompositeLoss
from utils.learn.validate import validate


def build_model_and_optimizer(config):
    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']

    section("Training Setup")

    model_name = model_cfg.get('model_name', 'ArtRT_R1')
    sub_section("Model")
    metric("Name", model_name)

    # Create model — always ICtCpNetV2 (ArtRT_R1 / ArtRT_Test)
    scale = model_cfg.get('scale', 4)
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
        optax.scale_by_adam(
            b1=training_cfg.get('adam_beta1', 0.9),
            b2=training_cfg.get('adam_beta2', 0.999),
        ),
        optax.add_decayed_weights(training_cfg.get('weight_decay', 0.01)),
    )
    opt_state = optimizer.init(params)

    # Loss weights: from schedule if available, otherwise defaults.
    schedule = config.get('schedule', [])
    if schedule:
        initial_weights = dict(schedule[0]['weights'])
    else:
        initial_weights = {'charbonnier': 1.0, 'rgb': 0.0, 'haarpsi': 0.0}
    criterion = CompositeLoss(initial_weights)

    sub_section("Training")
    metric("Loss Weights", str({k: f'{v:.8f}' for k, v in initial_weights.items()}))
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
        from utils.learn.io import load_checkpoint as _load
        ckpt = _load(args.resume)
        params = ckpt['params']
        opt_state = ckpt['opt_state']
        start_epoch = ckpt['epoch'] + 1
    elif args.pretrained:
        from utils.learn.io import load_checkpoint as _load
        ckpt = _load(args.pretrained)
        params = ckpt['params'] if 'params' in ckpt else ckpt
    return params, opt_state, start_epoch


def build_dataloaders(config):
    dataset_cfg = config['dataset']
    training_cfg = config['training_settings']

    sub_section("Data")
    metric("Training", f"{len(dataset_cfg['dataset_paths'])} sources")

    from utils.data.mkv_loader import discover_clips
    from utils.data.prefetch import PrefetchIterator

    clips = discover_clips(dataset_cfg['dataset_paths'])
    total_frames = sum(c['n_frames'] for c in clips)
    bs = training_cfg['batch_size']
    n_batches = max(total_frames // bs, 1)
    metric("Batches/epoch", str(n_batches))

    def make_train_loader():
        return PrefetchIterator(
            dataset_cfg['dataset_paths'], batch_size=bs, n_workers=3, frame_queue_size=300,
        )

    val_clips = {}
    for vp in dataset_cfg.get('val_dataset_paths', []):
        name = Path(vp).name
        clips = discover_clips([vp])
        val_clips[name] = clips

    return make_train_loader, val_clips, n_batches


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
            try:
                b_psnr, b_ssim, b_vmaf = validate(model, params, clips,
                                                  val_batch_size, run_dir, name,
                                                  baseline=True)
            except Exception as e:
                console.error(f"Baseline failed for {name}: {e}")
                b_psnr = b_ssim = b_vmaf = float('nan')
            baseline[name] = {'psnr': b_psnr, 'ssim': b_ssim, 'vmaf': b_vmaf}
            metric(name, f"psnr={b_psnr:.2f}  ssim={b_ssim:.4f}  vmaf={b_vmaf:.4f}")
        json.dump(baseline, open(baseline_path, 'w'))
    json.dump(baseline, open(run_dir / 'baseline.json', 'w'))
    divider()

    return baseline
