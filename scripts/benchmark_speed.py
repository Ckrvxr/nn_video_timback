"""Quick training-speed benchmark: data loader vs train step."""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from yaml import safe_load

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader


def avg(l):
    return sum(l) / len(l) if l else 0.0


def benchmark_loader(config, n_batches=20):
    loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['training']['batch_size'],
        scales=config['data'].get('scales', config['model']['scales']),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=config['data'].get('workers', 4),
        is_train=True,
        data_type=config['data'].get('data_type', 'compressed'),
    )
    times = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        times.append(time.time() - t0)
        t0 = time.time()
    return times, len(loader.dataset)


def benchmark_step(config, n_steps=20, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)
    criterion = CompositeLoss(config['loss'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None

    loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['training']['batch_size'],
        scales=config['data'].get('scales', config['model']['scales']),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=0,  # single-threaded to isolate GPU step time
        is_train=True,
        data_type=config['data'].get('data_type', 'compressed'),
    )

    model.train()
    step_times = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= n_steps:
            break
        lr = batch['lr_frames'].to(device)
        hr = batch['hr'].to(device)
        scale = batch['scale']

        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
                loss_dict = criterion(pred, hr)
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(lr[:, 0], lr[:, 1], lr[:, 2], scale=scale)
            loss_dict = criterion(pred, hr)
            loss_dict['total'].backward()
            optimizer.step()

        step_times.append(time.time() - t0)
        t0 = time.time()
    return step_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--n_steps', type=int, default=20)
    args = parser.parse_args()

    config = safe_load(open(args.config))
    print(f'Config: {args.config}')
    print(f"batch_size={config['training']['batch_size']}, workers={config['data'].get('workers', 4)}")

    print('\n[1/2] Benchmarking DataLoader...')
    lt, n_samples = benchmark_loader(config, args.n_batches)
    print(f'  dataset samples: {n_samples}')
    print(f'  first batch: {lt[0]:.3f}s')
    print(f'  avg batch (excl first): {avg(lt[1:]):.3f}s')
    print(f'  it/s: {1.0/avg(lt[1:]):.2f}')

    print('\n[2/2] Benchmarking train step (workers=0)...')
    st = benchmark_step(config, args.n_steps)
    print(f'  first step: {st[0]:.3f}s')
    print(f'  avg step (excl first): {avg(st[1:]):.3f}s')
    print(f'  it/s: {1.0/avg(st[1:]):.2f}')


if __name__ == '__main__':
    main()
