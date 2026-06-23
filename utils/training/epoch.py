import time
from pathlib import Path

import torch
from tqdm import tqdm

from utils.console import console
from utils.data.ictcp import batch_yuv_to_ictcp
from utils.memory import gpu_low, get_memory_manager

from . import cli
from .cli import check_memory
from .step import (
    sam_first_pass, sam_second_pass, standard_optimizer_step,
    log_expert_utilization,
)
from .validate import validate


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
    running_losses: dict[str, float] = {}

    training_cfg = config['training_settings']
    model_cfg = config['model_architecture']
    dataset_cfg = config['dataset']
    logging_cfg = config.get('logging_settings', {})
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

    model_name = model_cfg.get('model_name', 'hyper_fixer')
    is_timback = model_name == 'timback'
    sequential = dataset_cfg.get('sequential_mode', False) and is_timback
    last_idx = dataset_cfg['num_frames'] - 1
    expert_counts = torch.zeros(model_cfg.get('num_experts', 100), device='cpu')
    use_sam = training_cfg.get('enable_sam', False)

    if is_timback:
        model.reset_state(bs or 1, device)

    pbar = tqdm(total=n_samples, desc="Training", unit="sample", leave=False, miniters=10)
    t_batch_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if check_pause_exit_signals(run_dir):
            break
        data_end = time.perf_counter()
        lr = batch_yuv_to_ictcp(batch['lr_frames'], device)
        hr = batch_yuv_to_ictcp(batch['hr'], device)

        def run_forward():
            moe_weight = config['loss_weights'].get('moe', 0.01)
            if sequential:
                model.reset_state(lr.size(0), device)

                for t in range(lr.size(1)):
                    if t == last_idx:
                        pred = model(lr[:, t].to(memory_format=torch.channels_last))
                        loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                        loss_dict['moe'] = model._balancing_loss * moe_weight if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                        if hasattr(model, '_balancing_loss'):
                            model._balancing_loss = None
                        loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                        loss = loss_dict['total'] / grad_accum
                    else:
                        with torch.no_grad():
                            model.forward_ssm_ictcp(lr[:, t].to(memory_format=torch.channels_last))

                if hasattr(model, '_last_expert_idx'):
                    idx_flat = model._last_expert_idx.cpu().flatten()
                    valid = idx_flat >= 0
                    if valid.any():
                        expert_counts.index_add_(0, idx_flat[valid],
                                                 torch.ones_like(idx_flat[valid], dtype=torch.float))
                model._t_state = model._t_state.detach()

            else:
                model.reset_state(lr.size(0), device)
                pred = model(lr[:, last_idx].to(memory_format=torch.channels_last))
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                loss_dict['moe'] = model._balancing_loss * moe_weight if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                if hasattr(model, '_balancing_loss'):
                    model._balancing_loss = None
                loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                loss = loss_dict['total'] / grad_accum
                if hasattr(model, '_last_expert_idx'):
                    idx_flat = model._last_expert_idx.cpu().flatten()
                    valid = idx_flat >= 0
                    if valid.any():
                        expert_counts.index_add_(0, idx_flat[valid],
                                                 torch.ones_like(idx_flat[valid], dtype=torch.float))
            return loss, loss_dict

        if use_sam and is_timback and hasattr(model, '_t_state') and model._t_state is not None:
            saved_state = model._t_state.clone()
        else:
            saved_state = None

        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            loss, loss_dict = run_forward()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_accum = ((batch_idx + 1) % grad_accum == 0) or (batch_idx == n_batches - 1)
        if is_last_accum:
            if use_sam:
                moe_weight = sam_first_pass(model, optimizer, scaler, config, run_forward)
                if moe_weight is not None:
                    sam_second_pass(model, optimizer, scaler, clip_grad, run_forward, saved_state)
            else:
                standard_optimizer_step(model, optimizer, scaler, clip_grad)

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

        if (batch_idx + 1) % log_interval == 0:
            log_expert_utilization(expert_counts, loss_dict, batch_loss, optimizer, epoch, n_epochs, batch_idx, n_batches)

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0

        _postfix_order = ['ms_ssim', 'gmsd', 'haarpsi', 'fft', 'rgb', 'char', 'moe']
        count = batch_idx + 1
        _postfix_parts = [f"loss={running_losses['total'] / count:.6f}"]
        for name in _postfix_order:
            val = running_losses.get(name)
            if val is not None:
                _postfix_parts.append(f"{name}={val / count:.6f}")
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
