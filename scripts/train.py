import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yaml import safe_load


from utils.console import console, section, metric, divider
from utils.learn import cli
from utils.learn.cli import EXIT_FLAG, sigint_handler, parse_args
from utils.learn.io import start_io_worker, stop_io_worker, save_epoch_checkpoint
from utils.learn.metrics import composite_score
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
    global EXIT_FLAG
    start_io_worker()
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('random_seed')
    if seed:
        from utils.learn.cli import set_seed
        set_seed(seed)

    dataset_cfg = config['dataset']
    logging_cfg = config['logging_settings']

    model, params, optimizer, opt_state, criterion = build_model_and_optimizer(config)
    lr_schedule_fn, n_epochs = build_lr_schedule(config)
    start_epoch = 0

    if args.resume or args.pretrained:
        params, opt_state, start_epoch = load_checkpoint(args, params, opt_state)

    make_train_loader, val_clips, n_batches = build_dataloaders(config)
    train_step = make_train_step(model, optimizer, criterion)

    if args.limit > 0:
        n_batches = min(n_batches, args.limit)

    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith('run_'):
            try:
                existing.append(int(p.name.split('_')[1]))
            except ValueError:
                continue
    run_n = max(existing) + 1 if existing else 1
    run_dir = output_dir / f'run_{run_n:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    cli.RUN_DIR = run_dir

    metric("Run Dir", str(run_dir))
    divider()

    val_batch_size = dataset_cfg.get('validation_batch_size', 2)
    baseline = compute_baseline(config, val_clips, model, params, logging_cfg, run_dir, output_dir) if val_clips else {}

    schedule = config.get('schedule', [])
    if schedule:
        loss_weights = dict(schedule[0]['weights'])
    else:
        loss_weights = {'charbonnier': 1.0, 'rgb': 0.0, 'haarpsi': 0.0}
    criterion.update_weights(loss_weights)

    try:
        exit_flag_ref = [EXIT_FLAG]
        for epoch in range(start_epoch, n_epochs):
            wu_weights = get_epoch_weights(epoch, schedule, loss_weights)
            if wu_weights is not None:
                merged = dict(loss_weights)
                merged.update(wu_weights)
                criterion.update_weights(merged)
                loss_weights = merged
                console.info(f"Epoch {epoch+1} weights: {', '.join(f'{k}={v:.8f}' for k, v in wu_weights.items() if k != 'moe')}")

            if check_run_signals(run_dir, exit_flag_ref):
                EXIT_FLAG = True
                save_epoch_checkpoint(
                    max(0, epoch - 1), params, opt_state, train_loss=0.0,
                    metrics={}, score=0.0, lr=0.0,
                    loss_weights=loss_weights, run_dir=run_dir, output_dir=output_dir,
                )
                break

            train_loader = make_train_loader()
            try:
                params, opt_state, train_loss = train_epoch(
                    model, params, opt_state, train_step, train_loader,
                    criterion, config, lr_schedule_fn, run_dir=run_dir,
                    epoch=epoch, n_epochs=n_epochs, n_batches=n_batches,
                )
            finally:
                train_loader.close()

            if EXIT_FLAG:
                save_epoch_checkpoint(
                    epoch, params, opt_state, train_loss,
                    metrics={}, score=-train_loss * 10, lr=0.0,
                    loss_weights=loss_weights, run_dir=run_dir, output_dir=output_dir,
                )
                break

            metrics = {}
            avg_lr = 0.0
            if epoch % logging_cfg.get('validation_interval', 1) == 0:
                section(f"Epoch {epoch+1}/{n_epochs}  —  loss={train_loss:.8f}")
                for name, clips in val_clips.items():
                    psnr = ssim = vmaf = vgg = float('nan')
                    try:
                        psnr, ssim, vmaf, vgg = validate(model, params, clips,
                                                        val_batch_size, name)
                    except Exception as e:
                        console.error(f"RGB validation failed for {name}: {e}")
                    metrics[name] = {'psnr': psnr, 'ssim': ssim, 'vmaf': vmaf,
                                     'vgg': vgg}
                    log_validation(name, psnr, ssim, vmaf, baseline, vgg=vgg)
                avg_lr = float(lr_schedule_fn(epoch + n_batches / max(n_batches, 1)))
                divider()

            score = composite_score(metrics, baseline, train_loss)
            save_epoch_checkpoint(
                epoch, params, opt_state, train_loss, metrics, score, avg_lr,
                loss_weights, run_dir, output_dir,
            )
    finally:
        stop_io_worker()


if __name__ == '__main__':
    main()
