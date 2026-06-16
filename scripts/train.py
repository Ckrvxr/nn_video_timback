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

import numpy as np
import torch
warnings.filterwarnings('ignore', message='Cannot set number of intraop threads')
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from yaml import safe_load
from tqdm import tqdm

from models import MambaFixer
from models.components.color_space import yuv_to_rgb
from losses.composite import CompositeLoss
from utils.console import console
from utils.dataset import create_dataloader
from utils.frame_cache import FrameCache
from utils.blosc_cache import BloscCache
from utils.video_loader import load_video_frames_raw
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


def save_checkpoint(model, optimizer, scheduler, epoch, run_dir: Path, output_dir: Path, is_best: bool = False):
    ckpt = {
        'epoch': epoch,
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                                 for k, v in optimizer.state_dict().items()},
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    run_last = run_dir / 'last.pth'
    run_best = run_dir / 'best.pth'
    out_last = output_dir / 'last.pth'
    out_best = output_dir / 'best.pth'

    def _write():
        torch.save(ckpt, str(run_last))
        if is_best:
            shutil.copy2(str(run_last), str(run_best))
        shutil.copy2(str(run_last), str(out_last))
        if is_best:
            shutil.copy2(str(run_best), str(out_best))

    _IO_QUEUE.put((_write, (), {}))


def _cache_one_video(args: tuple) -> str:
    """Decode one video and write to BloscCache. Returns status string."""
    video_path, cache_root = args
    cache = BloscCache(cache_root)
    cp = cache.cache_path(video_path)
    if cp.exists():
        return f'skip  {Path(video_path).name}'
    raw_frames = load_video_frames_raw(video_path)
    cache.put_raw(video_path, raw_frames)
    return f'cached {Path(video_path).name} ({len(raw_frames)} frames)'


from concurrent.futures import ProcessPoolExecutor, as_completed

def clear_dataset_cache(dataset_path):
    cache_root = Path(dataset_path) / 'cache_yuv'
    if cache_root.exists():
        console.info(f"Clearing cache directory: {cache_root}")
        try:
            shutil.rmtree(str(cache_root))
            console.success(f"Successfully cleared cache: {cache_root}")
        except Exception as e:
            console.error(f"Failed to clear cache {cache_root}: {e}")

def warm_blosc_cache(config, target_dataset_path=None):
    if not config['dataset'].get('warm_cache_on_startup', True):
        console.info('Skipping cache warming')
        return

    datasets = config['dataset']['dataset_paths']
    if target_dataset_path is not None:
        datasets = [str(target_dataset_path)]
        
    limit = config['dataset'].get('limit', 0) # Fallback for CLI limit

    todos = []
    for ds_root_str in datasets:
        ds_root = Path(ds_root_str)
        cache_root = ds_root / 'cache_yuv'
        hr_dir = ds_root / 'HR'

        video_names = sorted(f.stem for f in hr_dir.glob('*.mp4'))
        if limit > 0:
            video_names = video_names[:limit]

        # Targets include HR and all other variant directories
        variant_dirs = [d for d in ds_root.iterdir() if d.is_dir() and d.name not in ('HR', 'cache_yuv')]
        targets = [hr_dir] + variant_dirs
        
        for vd in targets:
            for name in video_names:
                mp4 = vd / f'{name}.mp4'
                if mp4.exists():
                    blosc = BloscCache(cache_root)
                    if not blosc.cache_path(mp4).exists():
                        todos.append((str(mp4), str(cache_root)))

    if not todos:
        console.success(f"BloscCache already warm for {[Path(d).name for d in datasets]}")
        return

    console.info(f"Warming BloscCache for {[Path(d).name for d in datasets]}: {len(todos)} files (Sequential mode for 16GB RAM)...")
    # Force 1 worker to avoid OOM on 16GB RAM machines when processing 4K
    max_workers = 1 
    t0 = time.perf_counter()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_cache_one_video, todo) for todo in todos]
        for future in tqdm(as_completed(futures), total=len(todos), desc="Warming Cache", unit="vid"):
            try:
                future.result()
            except Exception as e:
                console.error(f'Failed to cache video: {e}')

    console.success(f"Cache warming complete ({time.perf_counter() - t0:.1f}s)")


def train_epoch(model, loader, criterion, optimizer, device, config, scaler=None, batch_times=None, run_dir=None):
    global EXIT_FLAG
    model.train()
    total_loss = 0.0
    n_batches = len(loader)
    
    training_cfg = config['training_settings']
    model_cfg = config['model_architecture']
    dataset_cfg = config['dataset']
    
    grad_accum = training_cfg.get('gradient_accumulation_steps', 1)
    clip_grad = training_cfg['gradient_clipping_threshold']
    bs = training_cfg['batch_size']
    
    model_name = model_cfg.get('model_name', 'hyper_fixer')
    is_mamba = model_name == 'mamba_fixer'
    sequential = dataset_cfg.get('sequential_mode', False) and is_mamba
    center_idx = dataset_cfg['num_frames'] // 2
    prev_video_id = -1
    expert_counts = torch.zeros(model_cfg.get('num_experts', 42), device='cpu')
    use_sam = training_cfg.get('enable_sam', False)

    if is_mamba:
        model.reset_state(bs or 1, device)

    pbar = tqdm(total=n_batches, desc="Training", unit="batch", leave=False, miniters=10)
    t_batch_start = time.perf_counter()
    optimizer.zero_grad()
    
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
                        pred = model(lr[:, t].to(memory_format=torch.channels_last), pre_ictcp=False)
                        loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                        loss_dict['moe'] = model._balancing_loss * 0.1
                        loss_dict['total'] = loss_dict['total'] + loss_dict['moe']
                        loss = loss_dict['total'] / grad_accum
                    else:
                        with torch.no_grad():
                            model(lr[:, t].to(memory_format=torch.channels_last), ssm_only=True, pre_ictcp=False)
                
                if hasattr(model, '_last_expert_idx'):
                    expert_counts.index_add_(0, model._last_expert_idx.cpu(),
                                             torch.ones_like(model._last_expert_idx, dtype=torch.float))
                model._t_state = model._t_state.detach()
                
            else:
                model.reset_state(lr.size(0), device)
                pred = model(lr[:, center_idx].to(memory_format=torch.channels_last), pre_ictcp=False)
                loss_dict = criterion(pred, hr.to(memory_format=torch.channels_last))
                loss_dict['moe'] = model._balancing_loss * 0.1
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
                optimizer.first_step(zero_grad=True)

                # 2. Restore Mamba state for second pass
                if saved_state is not None:
                    model._t_state = saved_state.clone()

                # 3. Second pass (on adversarial weights)
                with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
                    loss_adv, loss_dict_adv = run_forward()

                if scaler is not None:
                    scaler.scale(loss_adv).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss_adv.backward()

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
                optimizer.zero_grad()

        batch_loss = loss_dict['total'].item()
        total_loss += batch_loss

        t_now = time.perf_counter()
        batch_time = t_now - t_batch_start
        data_time = data_end - t_batch_start
        if batch_times is not None:
            batch_times.append({'batch': batch_time, 'data': data_time})
        t_batch_start = t_now

        samples_sec = bs / batch_time if batch_time > 0 else 0
        pbar.set_postfix(loss=f'{batch_loss:.4f}', samples=f'{samples_sec:.0f}/s')
        pbar.update(1)

    pbar.close()
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
            # Validation uses pre_ictcp=False because Dataloader provides YUV
            pred = model(lr[:, lr.size(1)//2].to(memory_format=torch.channels_last), pre_ictcp=False)
        else:
            center = lr.size(1)//2
            pred = model(lr[:, center-1], lr[:, center], lr[:, center+1])

        # CompositeLoss expects pred/hr in normalized YUV range.
        # But PSNR/SSIM calculation needs RGB.
        pred_rgb = yuv_to_rgb(pred)
        hr_rgb = yuv_to_rgb(hr)

        batch_size = pred.size(0)
        take = min(batch_size, max_samples - n)
        total_psnr += calculate_psnr_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        total_ssim += calculate_ssim_batch(pred_rgb[:take], hr_rgb[:take]).sum().item()
        n += take

        if num_vmaf_samples > 0 and n_vmaf < num_vmaf_samples:
            from utils.vmaf import compute_vmaf
            for b in range(min(take, num_vmaf_samples - n_vmaf)):
                total_fpsnr += compute_vmaf(pred[b:b+1], hr[b:b+1])
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
            num_experts=model_cfg.get('num_experts', 42),
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
        warm_blosc_cache(config)
        train_loader = create_dataloader(
            datasets=dataset_cfg['dataset_paths'],
            batch_size=training_cfg['batch_size'],
            patch_size=dataset_cfg['patch_size'],
            frames=dataset_cfg['num_frames'],
            workers=dataset_cfg['num_workers'],
            is_train=True,
            cache_max_videos=dataset_cfg.get('max_videos_in_cache'),
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
            cache_max_videos=dataset_cfg.get('max_videos_in_cache'),
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

                    warm_blosc_cache(config, target_dataset_path=ds_path)
                    
                    ds_loader = create_dataloader(
                        datasets=[ds_path],
                        batch_size=training_cfg['batch_size'],
                        patch_size=dataset_cfg['patch_size'],
                        frames=dataset_cfg['num_frames'],
                        workers=dataset_cfg['num_workers'],
                        is_train=True,
                        cache_max_videos=dataset_cfg.get('max_videos_in_cache'),
                        clip_repeat=dataset_cfg.get('clip_repeat_factor', 1),
                        sequential=dataset_cfg.get('sequential_mode', False),
                        persistent_workers=False,
                    )
                    
                    ds_loss = train_epoch(model, ds_loader, criterion, optimizer, device, config, scaler, run_dir=run_dir)
                    epoch_loss += ds_loss * len(ds_loader)
                    total_batches += len(ds_loader)
                    
                    del ds_loader
                    clear_dataset_cache(ds_path)
                
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
                        warm_blosc_cache(config, target_dataset_path=ds_path)
                        ds_val_loader = create_dataloader(
                            datasets=[ds_path],
                            batch_size=dataset_cfg.get('validation_batch_size', 2),
                            patch_size=dataset_cfg['patch_size'],
                            frames=dataset_cfg['num_frames'],
                            workers=dataset_cfg.get('validation_num_workers', 2),
                            is_train=False,
                            cache_max_videos=dataset_cfg.get('max_videos_in_cache'),
                            persistent_workers=False,
                        )
                        psnr, ssim, vmaf = validate(model, ds_val_loader, device, num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                        total_psnr += psnr
                        total_ssim += ssim
                        total_vmaf += vmaf
                        del ds_val_loader
                        clear_dataset_cache(ds_path)
                    
                    psnr = total_psnr / len(val_datasets)
                    ssim = total_ssim / len(val_datasets)
                    vmaf = total_vmaf / len(val_datasets)
                else:
                    psnr, ssim, vmaf = validate(model, val_loader, device, num_vmaf_samples=logging_cfg.get('num_vmaf_samples', 0))
                
                console.info(f'Epoch {epoch}: loss={train_loss:.4f} | psnr={psnr:.2f} | ssim={ssim:.4f}')

            save_checkpoint(model, optimizer, scheduler, epoch, run_dir, output_dir)
    finally:
        stop_io_worker()

if __name__ == '__main__':
    main()
