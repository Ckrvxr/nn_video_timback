import time
from collections import deque
from pathlib import Path

import torch
from tqdm import tqdm

from utils.console import console
from utils.learn import cli
from core.optim.sophia import SophiaG


def check_pause_exit_signals(run_dir: Path | None) -> bool:
    if not run_dir:
        return False
    pause_file = run_dir / ".pause"
    exit_file = run_dir / ".exit"
    if exit_file.exists():
        console.warning("Exit signal detected. Stopping gracefully...")
        cli.EXIT_FLAG = True
        try:
            exit_file.unlink()
        except Exception:
            pass
        return True
    first_pause_msg = True
    while pause_file.exists() and not cli.EXIT_FLAG:
        if first_pause_msg:
            console.warning("Training paused. Delete .pause to resume.")
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


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    config,
    opt_cfg,
    lr_schedule_fn,
    device,
    dtype=torch.float32,
    run_dir=None,
    epoch=0,
    n_epochs=0,
    n_batches=0,
    scaler=None,
    writer=None,
):
    loss_window = config.get("logging_settings", {}).get("loss_window", 100)
    log_interval = config.get("logging_settings", {}).get("log_interval", 100)
    accum_steps = opt_cfg.get("gradient_accumulation_steps", 1)

    total_loss = 0.0
    running_losses: dict[str, deque] = {}

    model.train()

    pbar = tqdm(
        loader,
        total=n_batches,
        desc=f"Epoch {epoch+1}/{n_epochs}",
        bar_format="{l_bar}{bar:10}{r_bar}",
        unit="batch",
    )

    for batch_idx, batch in enumerate(pbar):
        if check_pause_exit_signals(run_dir):
            break
        if batch_idx >= n_batches:
            break

        x, target = batch[0].to(device, dtype=dtype), batch[1].to(device, dtype=dtype)

        lr_val = lr_schedule_fn(epoch + batch_idx / max(n_batches, 1))
        for g in optimizer.param_groups:
            g["lr"] = lr_val

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss_dict = criterion(pred, target)
        else:
            pred = model(x)
            loss_dict = criterion(pred, target)

        loss_val = loss_dict["total"]

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            console.warning(f"Skipped batch {batch_idx} due to non-finite loss")
            continue

        loss_to_backward = loss_val / accum_steps

        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        should_step = (batch_idx + 1) % accum_steps == 0 or batch_idx == n_batches - 1
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    opt_cfg.get("gradient_clipping_threshold", 1.0),
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    opt_cfg.get("gradient_clipping_threshold", 1.0),
                )
                optimizer.step()
            if isinstance(optimizer, SophiaG):
                optimizer.update_hessian()
            optimizer.zero_grad()

        batch_loss = loss_val.detach().item()
        total_loss += batch_loss
        for k, v in loss_dict.items():
            running_losses.setdefault(k, deque(maxlen=loss_window)).append(float(v.detach()))

        def avg_loss(name):
            d = running_losses.get(name)
            return sum(d) / len(d) if d else 0.0

        # ── TensorBoard (every log_interval batches) ──
        if (batch_idx % log_interval == 0 or batch_idx == n_batches - 1) and writer:
            gs = epoch * n_batches + batch_idx
            for k, v in loss_dict.items():
                writer.add_scalar(f"Loss/{k}", float(v.detach()), gs)
            writer.add_scalar("LR", lr_val, gs)

        # ── tqdm live display ──
        pbar.set_postfix({
            "loss": f"{avg_loss('total'):.6f}",
            "char": f"{avg_loss('char'):.6f}",
            "fft": f"{avg_loss('fft'):.6f}",
            "fbin": f"{avg_loss('fbin'):.6f}",
            "lr": f"{lr_val:.2e}",
        })

    avg_train = total_loss / max(1, batch_idx + 1)

    if writer:
        writer.add_scalar("Loss/train_epoch", avg_train, epoch)

    return avg_train
