import argparse
import json
import os
import queue
import random
import shutil
import signal
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

# Global control flags for training loop pause/graceful exit
EXIT_FLAG = False
RUN_DIR = None
MAIN_PID = os.getpid()

def sigint_handler(signum, frame):
    global EXIT_FLAG, RUN_DIR
    if os.getpid() != MAIN_PID:
        return
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print()
    print("\033[1;33m═══ Training Paused (Ctrl+C detected) ═══\033[0m")
    print("  \033[1;37m[c]\033[0m Continue training")
    if RUN_DIR:
        print("  \033[1;37m[p]\033[0m Pause (create \033[3m.pause\033[0m file, delete to resume)")
    print("  \033[1;37m[s]\033[0m Save checkpoint and exit")
    print("  \033[1;37m[e]\033[0m Exit immediately")

    while True:
        try:
            prompt = "\033[1;36mChoice [c/p/s/e]: \033[0m" if RUN_DIR else "\033[1;36mChoice [c/s/e]: \033[0m"
            choice = input(prompt).strip().lower()
            if choice == 'c':
                print("\033[32mResuming training...\033[0m")
                if RUN_DIR:
                    pause_file = RUN_DIR / '.pause'
                    if pause_file.exists():
                        try:
                            pause_file.unlink()
                        except Exception:
                            pass
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 'p' and RUN_DIR:
                pause_file = RUN_DIR / '.pause'
                try:
                    pause_file.touch()
                    print(f"\033[33mCreated '{pause_file}'. Training is now paused.\033[0m")
                    print("To resume: delete the .pause file, or press Ctrl+C again.")
                except Exception as e:
                    print(f"\033[31mError creating pause file: {e}\033[0m")
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 's':
                print("\033[33mGraceful exit requested. Will save checkpoint...\033[0m")
                EXIT_FLAG = True
                signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
                return
            elif choice == 'e':
                print("\033[31mExiting immediately...\033[0m")
                os._exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n\033[31mExiting immediately...\033[0m")
            os._exit(1)
        except Exception:
            pass

signal.signal(signal.SIGINT, sigint_handler)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
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
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.training.lr_scheduler import WarmupCosineLR
from yaml import safe_load
from tqdm import tqdm

from models import MambaFixer
from models.components.color_space import yuv_to_rgb, ictcp_to_yuv
from utils.training.losses.composite import CompositeLoss
from utils.console import console, section, sub_section, metric, divider
from utils.data.dataset import create_dataloader
from utils.evaluation.metrics import calculate_psnr_batch, calculate_ssim_batch
from utils.training.sam import SAM


# Global background writer for checkpoints / metrics so that disk IO does not
# block the training loop.
_IO_QUEUE = queue.Queue()
_IO_THREAD = None


def _io_worker():
    while True:
        item = _IO_QUEUE.get()
        if item is None:
            break
        fn, args, kwargs = item
        try:
            fn(*args, **kwargs)
        except Exception as e:
            console.error(f'Background IO failed: {e}')


def start_io_worker():
    global _IO_THREAD
    if _IO_THREAD is None:
        _IO_THREAD = threading.Thread(target=_io_worker, daemon=True)
        _IO_THREAD.start()


def stop_io_worker():
    global _IO_THREAD
    if _IO_THREAD is not None:
        _IO_QUEUE.put(None)
        _IO_THREAD.join(timeout=30)
        _IO_THREAD = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Load only model weights, start fresh training')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--limit', type=int, default=0,
                        help='Use only first N videos (0 = all)')
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_memory(config, force: bool = False):
    mem_cfg = config.get('memory_settings', {})
    max_ram_ratio = mem_cfg.get('max_ram_ratio', 0.0)
    max_vram_ratio = mem_cfg.get('max_vram_ratio', 0.0)

    ram_exceeded = False
    vram_exceeded = False

    if max_ram_ratio > 0 and psutil is not None:
        proc = psutil.Process()
        rss = proc.memory_info().rss
        total = psutil.virtual_memory().total
        ratio = rss / total
        if ratio > max_ram_ratio or force:
            gc.collect()
            rss_after = proc.memory_info().rss
            ratio_after = rss_after / total
            if ratio_after > max_ram_ratio * 1.1:
                console.warning(f"RAM still above threshold: {ratio_after*100:.0f}% (limit {max_ram_ratio*100:.0f}%)")
                gc.collect()
            elif ratio_after > max_ram_ratio:
                console.info(f"RAM after GC: {ratio_after*100:.0f}% (limit {max_ram_ratio*100:.0f}%)")
            ram_exceeded = ratio_after > max_ram_ratio

    if max_vram_ratio > 0 and torch.cuda.is_available():
        dev = torch.cuda.current_device()
        total_vram = torch.cuda.get_device_properties(dev).total_memory
        allocated = torch.cuda.memory_allocated(dev)
        ratio = allocated / total_vram
        if ratio > max_vram_ratio or force:
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated(dev)
            ratio_after = allocated_after / total_vram
            if ratio_after > max_vram_ratio * 1.1:
                console.warning(f"VRAM still above threshold: {ratio_after*100:.0f}% (limit {max_vram_ratio*100:.0f}%)")
            elif ratio_after > max_vram_ratio:
                console.info(f"VRAM after empty_cache: {ratio_after*100:.0f}% (limit {max_vram_ratio*100:.0f}%)")
            vram_exceeded = ratio_after > max_vram_ratio

    return ram_exceeded, vram_exceeded


def save_checkpoint(model, optimizer, scheduler, epoch, run_dir: Path, output_dir: Path, is_best: bool = False):
    run_last = run_dir / 'last.pth'
    run_best = run_dir / 'best.pth'
    out_last = output_dir / 'last.pth'
    out_best = output_dir / 'best.pth'

    def _write():
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }
        torch.save(ckpt, str(run_last))
        if is_best:
            shutil.copy2(str(run_last), str(run_best))
        shutil.copy2(str(run_last), str(out_last))
        if is_best:
            shutil.copy2(str(run_best), str(out_best))

    _IO_QUEUE.put((_write, (), {}))



def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None, batch_times=None, run_dir=None, epoch=0, n_epochs=0):
    global EXIT_FLAG
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    n_samples = len(loader.dataset)
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
    mem_check_interval = memory_cfg.get('check_interval', 0)
    
    model_name = model_cfg.get('model_name', 'hyper_fixer')
    is_mamba = model_name == 'mamba_fixer'
    sequential = dataset_cfg.get('sequential_mode', False) and is_mamba
    center_idx = dataset_cfg['num_frames'] // 2
    expert_counts = torch.zeros(model_cfg.get('num_experts', 100), device='cpu')
    use_sam = training_cfg.get('enable_sam', False)

    if is_mamba:
        model.reset_state(bs or 1, device)

    pbar = tqdm(total=n_samples, desc="Training", unit="sample", leave=False, miniters=10)
    t_batch_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if run_dir:
            pause_file = run_dir / '.pause'
            exit_file = run_dir / '.exit'
            
            if exit_file.exists():
                console.warning(f"\nExit signal detected (.exit file found in {run_dir}). Stopping gracefully...")
                EXIT_FLAG = True
                try:
                    exit_file.unlink()
                except Exception:
                    pass
            
            was_paused = False
            first_pause_msg = True
            while pause_file.exists() and not EXIT_FLAG:
                was_paused = True
                if first_pause_msg:
                    console.warning(f"\nTraining paused (.pause file found in {run_dir}). Delete the file to resume.")
                    first_pause_msg = False
                time.sleep(1.0)
                if exit_file.exists():
                    console.warning(f"\nExit signal detected during pause (.exit file found in {run_dir}). Stopping gracefully...")
                    EXIT_FLAG = True
                    try:
                        exit_file.unlink()
                    except Exception:
                        pass
            
            if was_paused and not EXIT_FLAG:
                console.success("Resuming training...")

        if EXIT_FLAG:
            break
        data_end = time.perf_counter()
        lr = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)

        def run_forward():
            moe_weight = config['loss_weights'].get('moe', 0.01)
            if sequential:
                model.reset_state(lr.size(0), device)

                for t in range(lr.size(1)):
                    if t == center_idx:
                        pred = model(lr[:, t].to(memory_format=torch.channels_last))
                        loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                        loss_dict['moe'] = model._balancing_loss * moe_weight if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                        if hasattr(model, '_balancing_loss'):
                            model._balancing_loss = None  # Clear graph reference immediately
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
                pred = model(lr[:, center_idx].to(memory_format=torch.channels_last))
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                loss_dict['moe'] = model._balancing_loss * moe_weight if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                if hasattr(model, '_balancing_loss'):
                    model._balancing_loss = None  # Clear graph reference immediately
                loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                loss = loss_dict['total'] / grad_accum
                if hasattr(model, '_last_expert_idx'):
                    idx_flat = model._last_expert_idx.cpu().flatten()
                    valid = idx_flat >= 0
                    if valid.any():
                        expert_counts.index_add_(0, idx_flat[valid],
                                                 torch.ones_like(idx_flat[valid], dtype=torch.float))
            return loss, loss_dict

        # Save initial Mamba state for the second pass of SAM
        if use_sam and is_mamba and hasattr(model, '_t_state') and model._t_state is not None:
            saved_state = model._t_state.clone()
        else:
            saved_state = None

        # First pass
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            loss, loss_dict = run_forward()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_last_accum = ((batch_idx + 1) % grad_accum == 0) or (batch_idx == n_batches - 1)
        if is_last_accum:
            if use_sam:
                # 1. SAM first step
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                # Check for NaNs/Infs in the gradients before taking the first step
                has_nan_or_inf = False
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_or_inf = True
                            break
                
                if has_nan_or_inf:
                    console.warning("NaN or Inf detected in gradients. Skipping SAM step.")
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.update()
                else:
                    optimizer.first_step(zero_grad=True)

                    # 2. Restore Mamba state for second pass
                    if saved_state is not None:
                        model._t_state = saved_state.clone()

                    # 3. Second pass (on adversarial weights)
                    with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                        loss_adv, loss_dict_adv = run_forward()

                    if scaler is not None:
                        scaler.scale(loss_adv).backward()
                        # Manually unscale gradients for the second pass to avoid PyTorch's double-unscale assertion
                        inv_scale = 1.0 / (scaler.get_scale() + 1e-8)
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    p.grad.data.mul_(inv_scale)
                    else:
                        loss_adv.backward()

                    # Check for NaNs/Infs in the second-pass gradients
                    has_nan_or_inf_adv = False
                    for p in model.parameters():
                        if p.grad is not None:
                            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                                has_nan_or_inf_adv = True
                                break
                    
                    if has_nan_or_inf_adv:
                        console.warning("NaN or Inf detected in second-pass gradients. Skipping SAM step.")
                        # Force the scaler to know that an inf was found in the second pass
                        if scaler is not None:
                            opt_state = scaler._per_optimizer_states.get(id(optimizer))
                            if opt_state is not None:
                                for dev in opt_state['found_inf_per_device'].keys():
                                    opt_state['found_inf_per_device'][dev].fill_(1.0)
                        # Restore original parameters manually
                        for group in optimizer.param_groups:
                            for p in group["params"]:
                                if p in optimizer.state and "old_p" in optimizer.state[p]:
                                    p.data.copy_(optimizer.state[p]["old_p"])
                                    del optimizer.state[p]["old_p"]
                        optimizer.zero_grad(set_to_none=True)
                        if scaler is not None:
                            scaler.update()
                    else:
                        # Gradient clipping
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                        # 4. SAM second step
                        optimizer.second_step(zero_grad=True)

                        # Update scaler scale factor if AMP is used
                        if scaler is not None:
                            scaler.update()

            else:
                # Standard optimization step
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    has_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                                  for p in model.parameters() if p.grad is not None)
                    if has_nan:
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    has_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                                  for p in model.parameters() if p.grad is not None)
                    if has_nan:
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

        # Print detailed Expert Leaderboard and Loss Breakdown every log_interval batches
        if (batch_idx + 1) % log_interval == 0:
            total_selections = expert_counts.sum().item()
            if total_selections > 0:
                n_experts = expert_counts.size(0)
                console.opt(colors=True).info(
                    "<bold><cyan>═══ Expert Utilization (Epoch {}/{}, Batch {}/{}) ═══</cyan></bold>",
                    epoch + 1, n_epochs, batch_idx + 1, n_batches,
                )
                console.opt(colors=True).info("<dim>Legend: </dim>"
                    "<red>≥20%</red> <dim>|</dim> <yellow>≥10%</yellow> <dim>|</dim> "
                    "<green>≥5%</green> <dim>|</dim> <blue>≥1%</blue> <dim>|</dim> <dim>&lt;1%</dim>")

                def heat_color(pct):
                    if pct >= 20: return "red"
                    if pct >= 10: return "yellow"
                    if pct >= 5:  return "green"
                    if pct >= 1:  return "blue"
                    return "dim"

                for row_start in range(0, n_experts, 10):
                    cells = []
                    for offset in range(10):
                        idx = row_start + offset
                        if idx >= n_experts:
                            break
                        cnt = expert_counts[idx].item()
                        pct = (cnt / total_selections) * 100
                        color = heat_color(pct)
                        cells.append(f"<{color}>E{idx:02d} {pct:>5.1f}%</{color}>")
                    console.opt(colors=True).info("  ".join(cells))
                divider()
                # Top-5 experts (plain text; color tags in substitutions won't render)
                top5 = expert_counts.topk(5)
                top_pairs = "  ".join(
                    f"E{idx} {cnt/total_selections*100:.1f}%"
                    for idx, cnt in zip(top5.indices.tolist(), top5.values.tolist())
                )
                # Loss breakdown (plain text; color only from template below)
                loss_parts = "  ".join(
                    f"{name}={val.item():.6f}"
                    for name, val in loss_dict.items() if name != 'total'
                )
                lr_val = optimizer.param_groups[0]['lr']
                console.opt(colors=True).info(
                    '<level>{}</level>  <green>loss</green>=<yellow>{:.6f}</yellow>  <cyan>lr={:.2e}</cyan>',
                    loss_parts, batch_loss, lr_val,
                )
                console.opt(colors=True).info('<green>Top-5:</green> {}', top_pairs)
                divider()

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0
        
        _postfix_order = ['ms_ssim', 'wavelet', 'fft', 'sobel', 'rgb', 'char', 'moe']
        count = batch_idx + 1
        _postfix_parts = [f"loss={running_losses['total'] / count:.6f}"]
        for name in _postfix_order:
            val = running_losses.get(name)
            if val is not None:
                _postfix_parts.append(f"{name}={val / count:.6f}")
        _postfix_parts.append(f"lr={optimizer.param_groups[0]['lr']:.2e}")
        pbar.set_postfix_str("  ".join(_postfix_parts))
        pbar.update(bs)

        if device.type == 'cuda' and (batch_idx + 1) % cuda_cache_interval == 0:
            torch.cuda.empty_cache()

        if mem_check_interval > 0 and (batch_idx + 1) % mem_check_interval == 0:
            check_memory(config)

    pbar.close()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if mem_check_interval > 0:
        check_memory(config, force=True)

    return total_loss / n_batches


@torch.no_grad()
def validate(model, val_loader, device, max_samples=100, num_vmaf_samples=0):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_fpsnr = 0.0
    n = 0
    n_vmaf = 0
    is_mamba = isinstance(model, MambaFixer)

    for batch in val_loader:
        if n >= max_samples:
            break

        lr = batch['lr_frames'].to(device, non_blocking=True)
        hr = batch['hr'].to(device, non_blocking=True)

        if is_mamba:
            model.reset_state(lr.size(0), device)
            pred = model(lr[:, lr.size(1)//2].to(memory_format=torch.channels_last))
        else:
            center = lr.size(1)//2
            pred = model(lr[:, center-1], lr[:, center], lr[:, center+1])

        # Model outputs ICtCp; convert to YUV for metrics.
        pred_yuv = ictcp_to_yuv(pred)
        hr_yuv = ictcp_to_yuv(hr)
        pred_rgb = yuv_to_rgb(pred_yuv)
        hr_rgb = yuv_to_rgb(hr_yuv)

        batch_size = pred.size(0)
        take = min(batch_size, max_samples - n)
        total_psnr += calculate_psnr_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        total_ssim += calculate_ssim_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        n += take

        if num_vmaf_samples > 0 and n_vmaf < num_vmaf_samples:
            from utils.evaluation.vmaf import compute_vmaf
            for b in range(min(take, num_vmaf_samples - n_vmaf)):
                total_fpsnr += compute_vmaf(pred_yuv[b:b+1], hr_yuv[b:b+1])
                n_vmaf += 1

    avg_vmaf = total_fpsnr / max(1, n_vmaf)
    return total_psnr / max(1, n), total_ssim / max(1, n), avg_vmaf


def main():
    global EXIT_FLAG
    start_io_worker()
    args = parse_args()
    config = safe_load(open(args.config))

    seed = args.seed or config.get('random_seed')
    if seed:
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

    model_name = model_cfg.get('model_name', 'hyper_fixer')

    section("Training Setup")

    if model_name == 'mamba_fixer':
        model = MambaFixer(
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
        console.info('Compiling model with torch.compile (reduce-overhead)...')
        try:
            model = torch.compile(model, mode='reduce-overhead')
        except Exception as e:
            console.warning(f'Compilation failed: {e}')
            console.warning('Falling back to uncompiled model')

    sub_section("Training")
    # ── Normalize loss_weights: list-of-schedule → dict + inline schedule ──
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

    n_epochs = training_cfg['num_epochs']
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

    console.info(f"[debug] scheduler._last_lr={scheduler.get_last_lr()}  optimizer lr={optimizer.param_groups[0]['lr']:.2e}")
    scaler = torch.amp.GradScaler('cuda') if training_cfg.get('use_mixed_precision', False) and device.type == 'cuda' else None

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

    sub_section("Data")
    train_loader = create_dataloader(
        datasets=dataset_cfg['dataset_paths'],
        batch_size=training_cfg['batch_size'],
        patch_size=dataset_cfg['patch_size'],
        frames=dataset_cfg['num_frames'],
        workers=dataset_cfg['num_workers'],
        is_train=True,
        clip_repeat=dataset_cfg.get('clip_repeat_factor', 1),
        sequential=dataset_cfg.get('sequential_mode', False),
        data_type=data_type,
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
        )
        metric(f"Val [{name}]", f"{len(val_loaders[name].dataset)} samples")

    global RUN_DIR
    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob('run_*'))
    next_id = max(int(d.name.split('_')[1]) for d in existing) + 1 if existing else 1
    run_dir = output_dir / f'run_{next_id:03d}'
    run_dir.mkdir()
    RUN_DIR = run_dir

    metric("Run Dir", str(run_dir))
    divider()

    # ── Baseline (untrained model, cached across runs) ────────────
    if val_loaders:
        import hashlib
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
                                                   num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                baseline[name] = {'psnr': b_psnr, 'ssim': b_ssim, 'vmaf': b_vmaf}
                metric(name, f"psnr={b_psnr:.2f}  ssim={b_ssim:.4f}  vmaf={b_vmaf:.4f}")
            json.dump(baseline, open(baseline_path, 'w'))
        json.dump(baseline, open(run_dir / 'baseline.json', 'w'))
        divider()
    else:
        sub_section("Baseline")
        console.info("  No validation sets configured, skipping baseline.")
        divider()

    # ── Per-epoch weights schedule helper ───────────────────────
    def _get_epoch_weights(epoch_idx: int, schedule: list, fallback: dict) -> dict | None:
        if not schedule:
            return None
        ep = epoch_idx + 1
        for entry in schedule:
            spec = entry['epoch']
            match = (isinstance(spec, int) and spec == ep)
            if isinstance(spec, str):
                if spec.endswith('+') and ep >= int(spec[:-1]):
                    match = True
                elif '-' in spec:
                    lo, hi = map(int, spec.split('-'))
                    if lo <= ep <= hi:
                        match = True
            if match:
                return dict(entry['weights'])
        return None

    try:
        for epoch in range(start_epoch, n_epochs):
            # Apply warmup weights if configured for this epoch
            wu_weights = _get_epoch_weights(epoch, config.get('schedule', []), config['loss_weights'])
            if wu_weights is not None:
                merged = dict(config['loss_weights'])
                merged.update(wu_weights)
                criterion.update_weights(merged)
                console.info(f"Epoch {epoch+1} weights: {', '.join(f'{k}={v}' for k, v in wu_weights.items() if k != 'moe')}")
            if run_dir:
                pause_file = run_dir / '.pause'
                exit_file = run_dir / '.exit'
                
                if exit_file.exists():
                    console.warning(f"\nExit signal detected (.exit file found in {run_dir}). Stopping gracefully...")
                    EXIT_FLAG = True
                    try:
                        exit_file.unlink()
                    except Exception:
                        pass
                
                was_paused = False
                first_pause_msg = True
                while pause_file.exists() and not EXIT_FLAG:
                    was_paused = True
                    if first_pause_msg:
                        console.warning(f"\nTraining paused (.pause file found in {run_dir}). Delete the file to resume.")
                        first_pause_msg = False
                    time.sleep(1.0)
                    if exit_file.exists():
                        console.warning(f"\nExit signal detected during pause (.exit file found in {run_dir}). Stopping gracefully...")
                        EXIT_FLAG = True
                        try:
                            exit_file.unlink()
                        except Exception:
                            pass
                
                if was_paused and not EXIT_FLAG:
                    console.success("Resuming training...")

            if EXIT_FLAG:
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
                    vmaf_str = f'  vmaf={vmaf:.4f}' if vmaf > 0 else ''
                    # Color-code VMAF
                    if vmaf >= 80:
                        vmaf_color = "green"
                    elif vmaf >= 60:
                        vmaf_color = "yellow"
                    else:
                        vmaf_color = "red"
                    bl = baseline.get(name, {})
                    psnr_delta = ""
                    ssim_delta = ""
                    vmaf_delta = ""
                    if bl:
                        d = psnr - bl['psnr']
                        psnr_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
                        if 'ssim' in bl:
                            d = ssim - bl['ssim']
                            ssim_delta = f" ({'+' if d >= 0 else ''}{d:.4f})"
                        if 'vmaf' in bl and vmaf > 0:
                            d = vmaf - bl['vmaf']
                            vmaf_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
                    color_msg = (
                        f"  <white>{{}}</white>"
                        f"  <dim>psnr={{:.2f}}{psnr_delta}  ssim={{:.4f}}{ssim_delta}</dim>"
                    )
                    if vmaf > 0:
                        color_msg += f"  <{vmaf_color}>vmaf={{:.2f}}{vmaf_delta}</{vmaf_color}>"
                    console.opt(colors=True).info(
                        color_msg, name, psnr, ssim, vmaf,
                    )
                divider()

                if logging_cfg.get('empty_cuda_cache_on_validation', False):
                    torch.cuda.empty_cache()

            save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
    finally:
        stop_io_worker()

if __name__ == '__main__':
    main()
