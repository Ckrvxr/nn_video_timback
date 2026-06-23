import time
from collections import deque
from pathlib import Path

import torch
from tqdm import tqdm

from utils.console import console
from utils.data.ictcp import batch_yuv_to_ictcp
from utils.memory import gpu_low, get_memory_manager

from . import cli
from .cli import check_memory
from .step import standard_optimizer_step


def check_pause_exit_signals(run_dir: Path | None) -> bool:
    """Check for pause/exit signal files and handle them appropriately.
    
    Returns:
        bool: True if training should exit, False otherwise.
    """
    from . import cli
    if not run_dir:
        return False
        
    pause_file = run_dir / '.pause'
    exit_file = run_dir / '.exit'

    if exit_file.exists():
        console.warning(f"\nExit signal detected (.exit file found in {run_dir}). Stopping gracefully...")
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
            console.warning(f"\nTraining paused (.pause file found in {run_dir}). Delete the file to resume.")
            first_pause_msg = False
        time.sleep(1.0)
        if exit_file.exists():
            console.warning(f"\nExit signal detected during pause (.exit file found in {run_dir}). Stopping gracefully...")
            cli.EXIT_FLAG = True
            try:
                exit_file.unlink()
            except Exception:
                pass
            return True

    if was_paused and not cli.EXIT_FLAG:
        console.success("Resuming training...")

    return cli.EXIT_FLAG


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None, batch_times=None, run_dir=None, epoch=0, n_epochs=0):
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    n_samples = n_batches * config['training_settings']['batch_size']
    running_losses: dict[str, deque] = {}

    training_cfg = config['training_settings']
    model_cfg = config['model_architecture']
    dataset_cfg = config['dataset']
    logging_cfg = config.get('logging_settings', {})
    loss_window = logging_cfg.get('loss_window', 100)
    log_interval = logging_cfg.get('log_interval', 50)

    grad_accum = training_cfg.get('gradient_accumulation_steps', 1)
    clip_grad = training_cfg['gradient_clipping_threshold']
    bs = training_cfg['batch_size']
    cuda_cache_interval = logging_cfg.get('cuda_cache_interval', 200)
    memory_cfg = config.get('memory_settings', {})
    mem_check_interval = memory_cfg.get('check_interval', 512)
    
    # Initialize memory manager
    max_ram_ratio = memory_cfg.get('max_ram_ratio', 0.3)
    max_vram_ratio = memory_cfg.get('max_vram_ratio', 0.5)
    mem_manager = get_memory_manager(max_ram_ratio, max_vram_ratio)

    is_pure_cnn = model_cfg.get('model_name', 'hyper_fixer') == 'pure_cnn'

    _loss_key_map = {'charbonnier': 'char', 'temporal_consistency': 'temporal'}
    _postfix_order = []
    for name, w in config.get('loss_weights', {}).items():
        if w == 0.0:
            continue
        mapped = _loss_key_map.get(name, name)
        _postfix_order.append(mapped)

    pbar = tqdm(total=n_samples, desc="Training", unit="sample", leave=False, miniters=10)
    t_batch_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if check_pause_exit_signals(run_dir):
            break
        data_end = time.perf_counter()
        lr = batch_yuv_to_ictcp(batch['lr_frames'], device)
        hr = batch_yuv_to_ictcp(batch['hr'], device)

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            if is_pure_cnn:
                x = lr.reshape(lr.size(0), -1, lr.size(3), lr.size(4)).to(memory_format=torch.channels_last)
                pred = model(x)
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
            else:
                last = lr.size(1) - 1
                pred = model(lr[:, last-2], lr[:, last-1], lr[:, last])
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
            loss = loss_dict['total'] / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_accum = ((batch_idx + 1) % grad_accum == 0) or (batch_idx == n_batches - 1)
        if is_last_accum:
            standard_optimizer_step(model, optimizer, scaler, clip_grad)

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss
        for k, v in loss_dict.items():
            running_losses.setdefault(k, deque(maxlen=loss_window)).append(v.item())

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0

        def avg_loss(name):
            d = running_losses.get(name)
            return sum(d) / len(d) if d else 0.0
        _postfix_parts = [f"loss={avg_loss('total'):.6f}"]
        for name in _postfix_order:
            val = avg_loss(name)
            if val != 0.0:
                _postfix_parts.append(f"{name}={val:.6f}")
        _postfix_parts.append(f"lr={optimizer.param_groups[0]['lr']:.2e}")
        pbar.set_postfix_str("  ".join(_postfix_parts))
        pbar.update(bs)

        if device.type == 'cuda':
            if (batch_idx + 1) % cuda_cache_interval == 0 or gpu_low(0.15, device):
                torch.cuda.empty_cache()

        if mem_check_interval > 0 and (batch_idx + 1) % mem_check_interval == 0:
            ram_pressure, vram_pressure = check_memory(config)
            
            # Dynamic batch size adjustment based on memory pressure
            if ram_pressure or vram_pressure:
                console.warning(f"Memory pressure detected at batch {batch_idx + 1}, reducing effective batch size")
                # Force gradient sync and cleanup more frequently under pressure
                if (batch_idx + 1) % (mem_check_interval // 2) == 0:
                    torch.cuda.empty_cache() if device.type == 'cuda' else None

    pbar.close()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if mem_check_interval > 0:
        check_memory(config, force=True)
        # Log final memory status
        console.info(mem_manager.log_memory_status("Final epoch memory: "))

    return total_loss / n_batches
