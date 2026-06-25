import time
from collections import deque
from pathlib import Path

import jax.numpy as jnp
from tqdm import tqdm

from utils.console import console
from utils.learn import cli


def check_pause_exit_signals(run_dir: Path | None) -> bool:
    if not run_dir:
        return False
    pause_file = run_dir / '.pause'
    exit_file = run_dir / '.exit'
    if exit_file.exists():
        console.warning("\nExit signal detected. Stopping gracefully...")
        cli.EXIT_FLAG = True
        try:
            exit_file.unlink()
        except Exception:
            pass
        return True
    first_pause_msg = True
    while pause_file.exists() and not cli.EXIT_FLAG:
        if first_pause_msg:
            console.warning("\nTraining paused. Delete .pause to resume.")
            first_pause_msg = False
        time.sleep(1.0)
        if exit_file.exists():
            cli.EXIT_FLAG = True
            try:
                exit_file.unlink()
            except Exception:
                pass
            return True
    return cli.EXIT_FLAG


def train_epoch(model, params, opt_state, train_step, loader, criterion, config,
                lr_schedule_fn, run_dir=None, epoch=0, n_epochs=0, n_batches=0):
    loss_window = config.get('logging_settings', {}).get('loss_window', 100)

    total_loss = 0.0
    running_losses: dict[str, deque] = {}

    _postfix_order = ['total', 'char', 'rgb', 'haarpsi']
    _display_map = {'total': 'loss'}

    pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{n_epochs}",
                unit='batch', leave=False, dynamic_ncols=True)

    for batch_idx, batch in enumerate(loader):
        if check_pause_exit_signals(run_dir):
            break
        if batch_idx >= n_batches:
            break

        if hasattr(batch, 'keys'):
            x = jnp.array(batch['lr'], dtype=jnp.float32)
            target = jnp.array(batch['hr'], dtype=jnp.float32)
        else:
            x, target = jnp.array(batch[0]), jnp.array(batch[1])

        lr_val = lr_schedule_fn(epoch + batch_idx / max(n_batches, 1))
        params, opt_state, loss_val, loss_dict, skipped = train_step(params, opt_state, x, target, lr_val)

        if bool(skipped):
            console.warning(f"Skipped batch {batch_idx} due to non-finite loss or gradients")
            continue

        batch_loss = float(loss_val)
        total_loss += batch_loss
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                running_losses.setdefault(k, deque(maxlen=loss_window)).append(v)
            else:
                running_losses.setdefault(k, deque(maxlen=loss_window)).append(float(v))

        def avg_loss(name):
            d = running_losses.get(name)
            return sum(d) / len(d) if d else 0.0
        postfix = {}
        for name in _postfix_order:
            val = avg_loss(name)
            if val != 0.0:
                postfix[_display_map.get(name, name)] = f'{val:.4f}'
        postfix['lr'] = f'{float(lr_val):.2e}'
        pbar.set_postfix(**postfix)
        pbar.update(1)

    pbar.close()
    return params, opt_state, total_loss / max(1, batch_idx + 1)
