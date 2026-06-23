import signal
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yaml import safe_load

import torch

# ── Triton compat: inductor expects triton_key() which was removed in triton ≥3.7
try:
    import triton
    import triton.compiler.compiler as _tcc
    if not hasattr(_tcc, 'triton_key'):
        _tcc.triton_key = lambda: triton.__version__
except ImportError:
    pass
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`')

from utils.console import console, section, sub_section, metric, divider

from utils.training.cli import EXIT_FLAG, RUN_DIR, sigint_handler, parse_args
from utils.training.io import start_io_worker, stop_io_worker, save_checkpoint
from utils.training.signals import check_run_signals
from utils.training.schedule import get_epoch_weights, log_validation
from utils.training.epoch import train_epoch, validate
from utils.training.setup import (
    build_model_and_optimizer, build_scheduler, load_checkpoint,
    build_dataloaders, compute_baseline,
)


signal.signal(signal.SIGINT, sigint_handler)


def main():
    global EXIT_FLAG, RUN_DIR
    start_io_worker()
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('random_seed')
    if seed:
        from utils.training.cli import set_seed
        set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']
    dataset_cfg = config['dataset']
    data_type = dataset_cfg.get('data_type', 'compressed')
    logging_cfg = config['logging_settings']

    model, criterion, optimizer = build_model_and_optimizer(config, device)
    scheduler, n_epochs = build_scheduler(optimizer, config)

    scaler = torch.amp.GradScaler('cuda') if training_cfg.get('use_mixed_precision', False) and device.type == 'cuda' else None

    start_epoch = load_checkpoint(args, model, optimizer, scheduler, device)

    train_loader, val_loaders = build_dataloaders(config, data_type)

    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob('run_*'))
    next_id = max(int(d.name.split('_')[1]) for d in existing) + 1 if existing else 1
    run_dir = output_dir / f'run_{next_id:03d}'
    run_dir.mkdir()
    RUN_DIR = run_dir

    metric("Run Dir", str(run_dir))
    divider()

    if val_loaders:
        baseline = compute_baseline(config, val_loaders, model, device, logging_cfg, run_dir, output_dir)
    else:
        baseline = {}
        sub_section("Baseline")
        console.info("  No validation sets configured, skipping baseline.")
        divider()

    try:
        exit_flag_ref = [EXIT_FLAG]
        for epoch in range(start_epoch, n_epochs):
            wu_weights = get_epoch_weights(epoch, config.get('schedule', []), config['loss_weights'])
            if wu_weights is not None:
                merged = dict(config['loss_weights'])
                merged.update(wu_weights)
                criterion.update_weights(merged)
                console.info(f"Epoch {epoch+1} weights: {', '.join(f'{k}={v}' for k, v in wu_weights.items() if k != 'moe')}")

            if check_run_signals(run_dir, exit_flag_ref):
                EXIT_FLAG = True
                save_checkpoint(model, optimizer, scheduler, max(0, epoch - 1), run_dir, output_dir)
                console.warning(f"Training gracefully stopped. Saved checkpoint for epoch {max(0, epoch - 1)}.")
                break

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler, run_dir=run_dir, epoch=epoch, n_epochs=n_epochs)
            scheduler.step()

            if EXIT_FLAG:
                save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
                console.warning(f"Training gracefully stopped. Saved checkpoint for epoch {epoch}.")
                break

            if epoch % logging_cfg.get('validation_interval', 1) == 0:
                if logging_cfg.get('empty_cuda_cache_on_validation', False):
                    torch.cuda.empty_cache()

                section(f"Epoch {epoch+1}/{n_epochs}  —  loss={train_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
                for name, loader in val_loaders.items():
                    psnr, ssim, vmaf = validate(
                        model, loader, device,
                        num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                    log_validation(name, psnr, ssim, vmaf, baseline)
                divider()

                if logging_cfg.get('empty_cuda_cache_on_validation', False):
                    torch.cuda.empty_cache()

            save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
    finally:
        stop_io_worker()

if __name__ == '__main__':
    main()
