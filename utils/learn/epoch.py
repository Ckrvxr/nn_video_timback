import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp

from utils.console import console
from utils.learn import cli
from utils.learn.cli import check_memory


def check_pause_exit_signals(run_dir: Path | None) -> bool:
    if not run_dir:
        return False
    pause_file = run_dir / '.pause'
    exit_file = run_dir / '.exit'
    if exit_file.exists():
        console.warning(f"\nExit signal detected. Stopping gracefully...")
        cli.EXIT_FLAG = True
        try:
            exit_file.unlink()
        except Exception:
            pass
        return True
    was_paused = False
    first_pause_msg = True
    while pause_file.exists() and not cli.EXIT_FLAG:
        was_paused = True
        if first_pause_msg:
            console.warning(f"\nTraining paused. Delete .pause to resume.")
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
    training_cfg = config['training_settings']
    logging_cfg = config.get('logging_settings', {})
    loss_window = logging_cfg.get('loss_window', 100)
    log_interval = logging_cfg.get('log_interval', 50)
    grad_accum = training_cfg.get('gradient_accumulation_steps', 1)
    bs = training_cfg['batch_size']
    mem_check_interval = config.get('memory_settings', {}).get('check_interval', 512)

    total_loss = 0.0
    running_losses: dict[str, deque] = {}

    _loss_key_map = {'charbonnier': 'char'}
    _postfix_order = []
    for name, w in config.get('loss_weights', {}).items():
        if w == 0.0:
            continue
        mapped = _loss_key_map.get(name, name)
        _postfix_order.append(mapped)

    batch_idx = 0
    t_batch_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        if check_pause_exit_signals(run_dir):
            break

        # Convert batch to JAX arrays (NHWC)
        if hasattr(batch, 'keys'):
            x = jnp.array(batch['lr'], dtype=jnp.float32)
            target = jnp.array(batch['hr'], dtype=jnp.float32)
        else:
            x, target = jnp.array(batch[0]), jnp.array(batch[1])

        lr_val = lr_schedule_fn(epoch + batch_idx / max(n_batches, 1))
        params, opt_state, loss_val, loss_dict = train_step(params, opt_state, x, target)

        batch_loss = float(loss_val)
        total_loss += batch_loss
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                running_losses.setdefault(k, deque(maxlen=loss_window)).append(v)
            else:
                running_losses.setdefault(k, deque(maxlen=loss_window)).append(float(v))

        if (batch_idx + 1) % log_interval == 0:
            def avg_loss(name):
                d = running_losses.get(name)
                return sum(d) / len(d) if d else 0.0
            parts = [f"loss={avg_loss('total'):.6f}"]
            for name in _postfix_order:
                val = avg_loss(name)
                if val != 0.0:
                    parts.append(f"{name}={val:.6f}")
            parts.append(f"lr={lr_val:.2e}")
            console.info(f"  batch {batch_idx+1}/{n_batches}  {'  '.join(parts)}")

        t_batch_start = time.perf_counter()

    return total_loss / max(1, batch_idx + 1)
