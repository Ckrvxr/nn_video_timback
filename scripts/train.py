import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yaml import safe_load

import jax
import jax.numpy as jnp

from utils.console import console, section, sub_section, metric, divider
from utils.learn.cli import EXIT_FLAG, RUN_DIR, sigint_handler, parse_args
from utils.learn.io import start_io_worker, stop_io_worker, save_checkpoint
from utils.learn.signals import check_run_signals
from utils.learn.schedule import get_epoch_weights, log_validation
from utils.learn.epoch import train_epoch
from utils.learn.validate import validate
from utils.learn.setup import (
    build_model_and_optimizer, build_lr_schedule, load_checkpoint,
    build_dataloaders, compute_baseline,
)
from utils.learn.step import make_train_step


signal.signal(signal.SIGINT, sigint_handler)


def main():
    global EXIT_FLAG, RUN_DIR
    start_io_worker()
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('random_seed')
    if seed:
        from utils.learn.cli import set_seed
        set_seed(seed)

    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']
    dataset_cfg = config['dataset']
    logging_cfg = config['logging_settings']

    model, params, optimizer, opt_state, criterion = build_model_and_optimizer(config)
    lr_schedule_fn, n_epochs = build_lr_schedule(config)
    start_epoch = 0

    if args.resume or args.pretrained:
        params, opt_state, start_epoch = load_checkpoint(args, params, opt_state)

    train_loader, val_clips, n_batches = build_dataloaders(config)
    train_step = make_train_step(model, optimizer, criterion)

    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f'run_{start_epoch+1:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    RUN_DIR = run_dir

    metric("Run Dir", str(run_dir))
    divider()

    dataset_cfg = config['dataset']
    val_batch_size = dataset_cfg.get('validation_batch_size', 2)
    baseline = compute_baseline(config, val_clips, model, params, logging_cfg, run_dir, output_dir) if val_clips else {}

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
                save_checkpoint(params, opt_state, max(0, epoch - 1), run_dir, output_dir)
                break

            train_loss = train_epoch(
                model, params, opt_state, train_step, train_loader,
                criterion, config, lr_schedule_fn, run_dir=run_dir,
                epoch=epoch, n_epochs=n_epochs, n_batches=n_batches,
            )

            if EXIT_FLAG:
                save_checkpoint(params, opt_state, epoch, run_dir, output_dir)
                break

            if epoch % logging_cfg.get('validation_interval', 1) == 0:
                section(f"Epoch {epoch+1}/{n_epochs}  —  loss={train_loss:.4f}")
                for name, clips in val_clips.items():
                    psnr, ssim, vmaf = validate(model, params, clips,
                                                val_batch_size, run_dir, name)
                    log_validation(name, psnr, ssim, vmaf, baseline)
                divider()

            save_checkpoint(params, opt_state, epoch, run_dir, output_dir)
    finally:
        stop_io_worker()


if __name__ == '__main__':
    main()
