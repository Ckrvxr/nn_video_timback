import hashlib
import json
from pathlib import Path

import torch
import torch.optim as optim

from components import Timback
from utils.console import console, section, sub_section, metric, divider
from utils.data.dataset import create_dataloader
from utils.training.losses.composite import CompositeLoss
from utils.training.sam import SAM

from .cli import set_seed


def build_model_and_optimizer(config, device):
    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']
    model_name = model_cfg.get('model_name', 'hyper_fixer')

    section("Training Setup")

    if model_name == 'timback':
        model = Timback(
            num_features=model_cfg.get('num_features', 16),
            state_dimension=model_cfg.get('state_dimension', 32),
            num_features_stream=model_cfg.get('num_features_stream', 2),
            num_experts=model_cfg.get('num_experts', 100),
            n_active=model_cfg.get('n_active', 4),
            dilation_rates=model_cfg.get('dilation_rates', [1, 2, 4, 32]),
            routing_threshold=model_cfg.get('routing_threshold', 1.0),
        ).to(device, memory_format=torch.channels_last)

    sub_section("Model")
    metric("Name", model_name)
    metric("Device", str(device))
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metric("Parameters", f"{n_params:,}")
    metric("Trainable", f"{n_trainable:,}")

    if training_cfg.get('enable_torch_compile', False):
        compile_mode = training_cfg.get('compile_mode', 'default')
        compile_backend = training_cfg.get('compile_backend', 'inductor')
        console.info(f'Compiling model with torch.compile (mode={compile_mode}, backend={compile_backend})...')
        try:
            model = torch.compile(model, mode=compile_mode, backend=compile_backend)
        except Exception as e:
            console.warning(f'Compilation failed: {e}')
            console.warning('Falling back to uncompiled model')

    sub_section("Training")
    if 'schedule' in config:
        config['loss_weights'] = config['schedule'][-1]['weights'].copy()
    else:
        metric("Loss Weights", str(config['loss_weights']))
    metric("Batch Size", str(training_cfg['batch_size']))
    metric("Grad Accum", str(training_cfg.get('gradient_accumulation_steps', 1)))
    lr_cfg = training_cfg.get('lr')
    if lr_cfg:
        metric("LR Config", f"peak={lr_cfg['warmup_peak']:.2e}→true={lr_cfg['true_peak']:.2e}  min={lr_cfg['min']:.2e}  warmup={lr_cfg['warmup_epochs']}ep")
    else:
        metric("Learning Rate", f"{training_cfg['learning_rate']:.2e}")
    metric("Epochs", str(training_cfg['num_epochs']))
    metric("Mixed Precision", str(training_cfg.get('use_mixed_precision', False)))
    metric("SAM", str(training_cfg.get('enable_sam', False)))
    metric("MoE Weight", str(config['loss_weights'].get('moe', 0.01)))

    criterion = CompositeLoss(config['loss_weights'])

    lr_cfg = training_cfg.get('lr')
    opt_lr = lr_cfg['warmup_peak'] if lr_cfg else training_cfg['learning_rate']

    use_sam = training_cfg.get('enable_sam', False)
    if use_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optim.AdamW,
            rho=training_cfg.get('sam_rho', 0.05),
            lr=opt_lr,
            weight_decay=training_cfg['weight_decay'],
            betas=(training_cfg['adam_beta1'], training_cfg['adam_beta2']),
            fused=device.type == 'cuda',
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_lr,
            weight_decay=training_cfg['weight_decay'],
            betas=(training_cfg['adam_beta1'], training_cfg['adam_beta2']),
            fused=device.type == 'cuda',
        )

    return model, criterion, optimizer


def build_scheduler(optimizer, config):
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from utils.training.lr_scheduler import WarmupCosineLR

    training_cfg = config['training_settings']
    n_epochs = training_cfg['num_epochs']
    lr_cfg = training_cfg.get('lr')

    if lr_cfg:
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_peak=lr_cfg['warmup_peak'],
            true_peak=lr_cfg['true_peak'],
            min_lr=lr_cfg['min'],
            warmup_epochs=lr_cfg['warmup_epochs'],
            n_epochs=n_epochs,
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=training_cfg['min_learning_rate'])

    return scheduler, n_epochs


def load_checkpoint(args, model, optimizer, scheduler, device):
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        for pg, lr_val in zip(optimizer.param_groups, scheduler.get_last_lr()):
            pg['lr'] = lr_val
        start_epoch = ckpt['epoch'] + 1
    elif args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    return start_epoch


def build_dataloaders(config, data_type):
    dataset_cfg = config['dataset']
    training_cfg = config['training_settings']

    sub_section("Data")
    train_loader = create_dataloader(
        datasets=dataset_cfg['dataset_paths'],
        batch_size=training_cfg['batch_size'],
        patch_size=dataset_cfg['patch_size'],
        frames=dataset_cfg['num_frames'],
        workers=dataset_cfg['num_workers'],
        is_train=True,
        segment_repeat=dataset_cfg.get('segment_repeat', 1),
        sequential=dataset_cfg.get('sequential_mode', False),
        data_type=data_type,
        max_cached_segments=dataset_cfg.get('max_cached_segments', 8),
    )
    metric("Training", f"{len(train_loader.dataset)} samples from {len(dataset_cfg['dataset_paths'])} sources")

    val_dataset_paths = dataset_cfg.get('val_dataset_paths', [])
    val_loaders = {}
    for vp in val_dataset_paths:
        name = Path(vp).name
        val_loaders[name] = create_dataloader(
            datasets=[vp],
            batch_size=dataset_cfg.get('validation_batch_size', 2),
            patch_size=dataset_cfg['patch_size'],
            frames=dataset_cfg['num_frames'],
            workers=dataset_cfg.get('validation_num_workers', 2),
            is_train=False,
            shuffle=True,
            data_type=data_type,
            max_cached_segments=dataset_cfg.get('max_cached_segments', 8),
        )
        metric(f"Val [{name}]", f"{len(val_loaders[name].dataset)} samples")

    return train_loader, val_loaders


def compute_baseline(config, val_loaders, model, device, logging_cfg, run_dir, output_dir):
    dataset_cfg = config['dataset']
    model_cfg = config['model_architecture']

    from .validate import validate

    config_hash_payload = json.dumps({
        'dataset_paths': dataset_cfg.get('dataset_paths'),
        'val_dataset_paths': dataset_cfg.get('val_dataset_paths'),
        'patch_size': dataset_cfg.get('patch_size'),
        'num_frames': dataset_cfg.get('num_frames'),
        'model_name': model_cfg.get('model_name'),
        'num_features': model_cfg.get('num_features'),
        'routing_threshold': model_cfg.get('routing_threshold'),
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
        for name, loader in val_loaders.items():
            b_psnr, b_ssim, b_vmaf = validate(model, loader, device,
                                               num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0),
                                               baseline=True)
            baseline[name] = {'psnr': b_psnr, 'ssim': b_ssim, 'vmaf': b_vmaf}
            metric(name, f"psnr={b_psnr:.2f}  ssim={b_ssim:.4f}  vmaf={b_vmaf:.4f}")
        json.dump(baseline, open(baseline_path, 'w'))
    json.dump(baseline, open(run_dir / 'baseline.json', 'w'))
    divider()

    return baseline
