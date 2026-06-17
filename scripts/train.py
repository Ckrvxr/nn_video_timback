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
    # Ignore signal during interactive menu
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("\n\n=== Training Paused (Ctrl+C detected) ===")
    print("Select an option:")
    print("  [c] Continue training")
    if RUN_DIR:
        print("  [p] Pause training (creates .pause file, delete it to resume)")
    print("  [s] Save checkpoint and exit gracefully")
    print("  [e] Exit immediately without saving")
    
    while True:
        try:
            prompt = "Choice [c/p/s/e]: " if RUN_DIR else "Choice [c/s/e]: "
            choice = input(prompt).strip().lower()
            if choice == 'c':
                print("Resuming training...")
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
                    print(f"Created '{pause_file}'. Training is now paused.")
                    print("To resume: delete this file, or press Ctrl+C again to choose another option.")
                except Exception as e:
                    print(f"Error creating pause file: {e}")
                signal.signal(signal.SIGINT, sigint_handler)
                return
            elif choice == 's':
                print("Graceful exit requested. Will save checkpoint and stop at next batch/epoch.")
                EXIT_FLAG = True
                signal.signal(signal.SIGINT, lambda s, f: os._exit(1))
                return
            elif choice == 'e':
                print("Exiting immediately...")
                os._exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting immediately...")
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
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
warnings.filterwarnings('ignore', message='Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`')
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from yaml import safe_load
from tqdm import tqdm

from models import MambaFixer
from models.components.color_space import yuv_to_rgb, ictcp_to_yuv
from losses.composite import CompositeLoss
from utils.console import console
from utils.dataset import create_dataloader
from utils.metrics import calculate_psnr_batch, calculate_ssim_batch
from utils.sam import SAM


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
                console.warning(f"RAM 仍超限: {ratio_after*100:.0f}% (上限 {max_ram_ratio*100:.0f}%)")
                gc.collect()
            elif ratio_after > max_ram_ratio:
                console.info(f"GC 后 RAM: {ratio_after*100:.0f}% (上限 {max_ram_ratio*100:.0f}%)")
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
                console.warning(f"显存仍超限: {ratio_after*100:.0f}% (上限 {max_vram_ratio*100:.0f}%)")
            elif ratio_after > max_vram_ratio:
                console.info(f"empty_cache 后显存: {ratio_after*100:.0f}% (上限 {max_vram_ratio*100:.0f}%)")
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



def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None, batch_times=None, run_dir=None):
    global EXIT_FLAG
    device = torch.device(device)
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    
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
    prev_video_id = -1
    expert_counts = torch.zeros(model_cfg.get('num_experts', 100), device='cpu')
    use_sam = training_cfg.get('enable_sam', False)

    if is_mamba:
        model.reset_state(bs or 1, device)

    pbar = tqdm(total=n_batches, desc="Training", unit="batch", leave=False, miniters=10)
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
            nonlocal prev_video_id
            if sequential:
                video_id = batch.get('video_id', -1)
                if video_id != prev_video_id:
                    model.reset_state(lr.size(0), device)
                prev_video_id = video_id

                for t in range(lr.size(1)):
                    if t == center_idx:
                        pred = model(lr[:, t].to(memory_format=torch.channels_last))
                        loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                        loss_dict['moe'] = model._balancing_loss * 0.1 if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                        if hasattr(model, '_balancing_loss'):
                            model._balancing_loss = None  # Clear graph reference immediately
                        loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                        loss = loss_dict['total'] / grad_accum
                    else:
                        with torch.no_grad():
                            model(lr[:, t].to(memory_format=torch.channels_last), ssm_only=True)
                
                if hasattr(model, '_last_expert_idx'):
                    expert_counts.index_add_(0, model._last_expert_idx.cpu(),
                                             torch.ones_like(model._last_expert_idx, dtype=torch.float))
                model._t_state = model._t_state.detach()
                
            else:
                model.reset_state(lr.size(0), device)
                pred = model(lr[:, center_idx].to(memory_format=torch.channels_last))
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                loss_dict['moe'] = model._balancing_loss * 0.1 if getattr(model, '_balancing_loss', None) is not None else torch.tensor(0.0, device=device)
                if hasattr(model, '_balancing_loss'):
                    model._balancing_loss = None  # Clear graph reference immediately
                loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                loss = loss_dict['total'] / grad_accum
                if hasattr(model, '_last_expert_idx'):
                    expert_counts.index_add_(0, model._last_expert_idx.cpu(),
                                             torch.ones_like(model._last_expert_idx, dtype=torch.float))
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
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss

        # Print detailed Expert Leaderboard and Loss Breakdown every log_interval batches
        if (batch_idx + 1) % log_interval == 0:
            total_selections = expert_counts.sum().item()
            if total_selections > 0:
                n_experts = expert_counts.size(0)
                console.info(f"\n🏆 Expert Heatmap (Batch {batch_idx + 1}/{n_batches})")
                console.info("─" * 95)

                def heat_color(pct):
                    if pct >= 20: return "red"
                    if pct >= 10: return "yellow"
                    if pct >= 5:  return "green"
                    if pct >= 1:  return "blue"
                    return "white"

                for row_start in range(0, n_experts, 7):
                    cells = []
                    for offset in range(7):
                        idx = row_start + offset
                        if idx >= n_experts:
                            break
                        cnt = expert_counts[idx].item()
                        pct = (cnt / total_selections) * 100
                        color = heat_color(pct)
                        cells.append(f"<{color}>E{idx:02d} {pct:>5.1f}%</{color}>")
                    console.opt(colors=True).info("  ".join(cells))
                console.info("─" * 95)
                loss_parts = []
                for name, val in loss_dict.items():
                    if name != 'total':
                        loss_parts.append(f"{name}={val.item():.6f}")
                console.info("Loss: " + "  ".join(loss_parts))
                console.info("─" * 95)

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0
        
        # Format Top-3 experts for tqdm postfix
        total_sel = expert_counts.sum().item()
        if total_sel > 0:
            top_val, top_idx = torch.topk(expert_counts, k=min(3, len(expert_counts)))
            top_str = ",".join([f"E{idx.item()}" for idx, val in zip(top_idx, top_val) if val > 0])
        else:
            top_str = "None"
            
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            char=f"{loss_dict.get('char', torch.tensor(0.0)).item():.4f}",
            fft=f"{loss_dict.get('fft', torch.tensor(0.0)).item():.4f}",
            moe=f"{loss_dict.get('moe', torch.tensor(0.0)).item():.4f}",
            top=top_str,
            samples=f"{samples_sec:.0f}/s"
        )
        pbar.update(1)

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
            from utils.vmaf import compute_vmaf
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
    console.info(f'Using device: {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    model_cfg = config['model_architecture']
    training_cfg = config['training_settings']
    dataset_cfg = config['dataset']
    logging_cfg = config['logging_settings']

    model_name = model_cfg.get('model_name', 'hyper_fixer')
    console.info(f'Model: {model_name}')

    if model_name == 'hyper_fixer':
        # Removed
        pass
    elif model_name == 'mamba_fixer':
        model = MambaFixer(
            num_features=model_cfg.get('num_features', 16),
            state_dimension=model_cfg.get('state_dimension', 32),
            num_features_stream=model_cfg.get('num_features_stream', 2),
            num_experts=model_cfg.get('num_experts', 100),
            n_active=model_cfg.get('n_active', 4),
            dilation_rates=model_cfg.get('dilation_rates', [1, 2, 4, 32]),
        ).to(device, memory_format=torch.channels_last)

    if training_cfg.get('enable_torch_compile', False):
        console.info('Using torch.compile')
        model = torch.compile(model, mode='max-autotune')

    criterion = CompositeLoss(config['loss_weights'], device=device)

    use_sam = training_cfg.get('enable_sam', False)
    if use_sam:
        console.info("Using SAM (Sharpness-Aware Minimization) optimizer wrapper")
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optim.AdamW,
            rho=training_cfg.get('sam_rho', 0.05),
            lr=training_cfg['learning_rate'],
            weight_decay=training_cfg['weight_decay'],
            betas=(training_cfg['adam_beta1'], training_cfg['adam_beta2']),
            fused=device.type == 'cuda',
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_cfg['learning_rate'],
            weight_decay=training_cfg['weight_decay'],
            betas=(training_cfg['adam_beta1'], training_cfg['adam_beta2']),
            fused=device.type == 'cuda',
        )

    n_epochs = training_cfg['num_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=training_cfg['min_learning_rate'])
    scaler = torch.amp.GradScaler('cuda') if training_cfg.get('use_mixed_precision', False) and device.type == 'cuda' else None

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
    elif args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    multiple_datasets = len(dataset_cfg['dataset_paths']) > 1
    train_loader = None
    val_loader = None

    if not multiple_datasets:
        train_loader = create_dataloader(
            datasets=dataset_cfg['dataset_paths'],
            batch_size=training_cfg['batch_size'],
            patch_size=dataset_cfg['patch_size'],
            frames=dataset_cfg['num_frames'],
            workers=dataset_cfg['num_workers'],
            is_train=True,
            clip_repeat=dataset_cfg.get('clip_repeat_factor', 1),
            sequential=dataset_cfg.get('sequential_mode', False),
        )
        val_loader = create_dataloader(
            datasets=dataset_cfg['dataset_paths'],
            batch_size=dataset_cfg.get('validation_batch_size', 2),
            patch_size=dataset_cfg['patch_size'],
            frames=dataset_cfg['num_frames'],
            workers=dataset_cfg.get('validation_num_workers', 2),
            is_train=False,
        )

    global RUN_DIR
    output_dir = Path(config['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob('run_*'))
    next_id = max(int(d.name.split('_')[1]) for d in existing) + 1 if existing else 1
    run_dir = output_dir / f'run_{next_id:03d}'
    run_dir.mkdir()
    RUN_DIR = run_dir

    console.info(f"Training run directory created: {run_dir}")
    console.info(f"To PAUSE training: press Ctrl+C and choose [p], or create file '{run_dir}/.pause'")
    console.info(f"To EXIT gracefully: press Ctrl+C and choose [s], or create file '{run_dir}/.exit'")

    try:
        for epoch in range(start_epoch, n_epochs):
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

            if multiple_datasets:
                epoch_loss = 0.0
                total_batches = 0
                for ds_path in dataset_cfg['dataset_paths']:
                    if run_dir:
                        pause_file = run_dir / '.pause'
                        exit_file = run_dir / '.exit'
                        if exit_file.exists() or EXIT_FLAG:
                            EXIT_FLAG = True
                            break
                        was_paused = False
                        first_pause_msg = True
                        while pause_file.exists() and not EXIT_FLAG:
                            was_paused = True
                            if first_pause_msg:
                                console.warning(f"\nTraining paused (.pause file found in {run_dir}). Delete the file to resume.")
                                first_pause_msg = False
                            time.sleep(1.0)
                            if exit_file.exists():
                                EXIT_FLAG = True
                        if was_paused and not EXIT_FLAG:
                            console.success("Resuming training...")
                    
                    if EXIT_FLAG:
                        break

                    ds_loader = create_dataloader(
                        datasets=[ds_path],
                        batch_size=training_cfg['batch_size'],
                        patch_size=dataset_cfg['patch_size'],
                        frames=dataset_cfg['num_frames'],
                        workers=dataset_cfg['num_workers'],
                        is_train=True,
                        clip_repeat=dataset_cfg.get('clip_repeat_factor', 1),
                        sequential=dataset_cfg.get('sequential_mode', False),
                        persistent_workers=False,
                    )
                    
                    ds_loss = train_epoch(model, ds_loader, criterion, optimizer, device, config, scaler, run_dir=run_dir)
                    epoch_loss += ds_loss * len(ds_loader)
                    total_batches += len(ds_loader)
                    
                    del ds_loader
                
                train_loss = epoch_loss / total_batches if total_batches > 0 else 0.0
            else:
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, scaler, run_dir=run_dir)

            scheduler.step()

            if EXIT_FLAG:
                save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
                console.warning(f"Training gracefully stopped. Saved checkpoint for epoch {epoch}.")
                break

            if epoch % logging_cfg.get('validation_interval', 1) == 0:
                if logging_cfg.get('empty_cuda_cache_on_validation', False):
                    torch.cuda.empty_cache()
                
                if multiple_datasets:
                    total_psnr = 0.0
                    total_ssim = 0.0
                    total_vmaf = 0.0
                    val_datasets = dataset_cfg['dataset_paths']
                    for ds_path in val_datasets:
                        if EXIT_FLAG:
                            break
                        ds_val_loader = create_dataloader(
                            datasets=[ds_path],
                            batch_size=dataset_cfg.get('validation_batch_size', 2),
                            patch_size=dataset_cfg['patch_size'],
                            frames=dataset_cfg['num_frames'],
                            workers=dataset_cfg.get('validation_num_workers', 2),
                            is_train=False,
                            persistent_workers=False,
                        )
                        psnr, ssim, vmaf = validate(model, ds_val_loader, device, num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                        total_psnr += psnr
                        total_ssim += ssim
                        total_vmaf += vmaf
                        del ds_val_loader
                    
                    psnr = total_psnr / len(val_datasets)
                    ssim = total_ssim / len(val_datasets)
                    vmaf = total_vmaf / len(val_datasets)
                else:
                    psnr, ssim, vmaf = validate(model, val_loader, device, num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                
                console.info(f'Epoch {epoch}: loss={train_loss:.4f} | psnr={psnr:.2f} | ssim={ssim:.4f}')
                
                if logging_cfg.get('empty_cuda_cache_on_validation', False):
                    torch.cuda.empty_cache()

            save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
    finally:
        stop_io_worker()

if __name__ == '__main__':
    main()
