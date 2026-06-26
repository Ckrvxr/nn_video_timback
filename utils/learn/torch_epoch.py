import time
from collections import deque
from pathlib import Path

import torch
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


def train_epoch(model, loader, criterion, optimizer, config,
                lr_schedule_fn, device, dtype=torch.float32, run_dir=None, epoch=0, n_epochs=0, n_batches=0):
    loss_window = config.get('logging_settings', {}).get('loss_window', 100)

    total_loss = 0.0
    running_losses: dict[str, deque] = {}

    _display_order = ['total', 'char', 'gmsd', 'haarpsi']
    _display_names = {'total': 'loss'}

    pbar = tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{n_epochs}",
                unit='batch', leave=False, dynamic_ncols=True)

    model.train()

    for batch_idx, batch in enumerate(loader):
        if check_pause_exit_signals(run_dir):
            break
        if batch_idx >= n_batches:
            break

        x, target = batch[0].to(device, dtype=dtype), batch[1].to(device, dtype=dtype)

        lr_val = lr_schedule_fn(epoch + batch_idx / max(n_batches, 1))
        for g in optimizer.param_groups:
            g['lr'] = lr_val

        pred = model(x)
        loss_dict = criterion(pred, target)
        loss_val = loss_dict['total']

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            console.warning(f"Skipped batch {batch_idx} due to non-finite loss")
            continue

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('training_settings', {}).get('gradient_clipping_threshold', 1.0))
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss_val.detach().item()
        total_loss += batch_loss
        for k, v in loss_dict.items():
            running_losses.setdefault(k, deque(maxlen=loss_window)).append(float(v))

        def avg_loss(name):
            d = running_losses.get(name)
            return sum(d) / len(d) if d else 0.0

        parts = []
        for name in _display_order:
            val = avg_loss(name)
            if val != 0.0:
                label = _display_names.get(name, name)
                parts.append(f"{label}={val:.8f}")
        parts.append(f"lr={lr_val:.2e}")
        pbar.set_description_str(f"Epoch {epoch+1}/{n_epochs} | {'  '.join(parts)}")
        pbar.update(1)

    pbar.close()
    return total_loss / max(1, batch_idx + 1)
