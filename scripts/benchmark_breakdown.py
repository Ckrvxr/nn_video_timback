"""Break down one training step: forward / loss / backward, with/without VGG."""
import time
import sys
from pathlib import Path

import torch
from yaml import safe_load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.av1_vsr import AV1VSR
from losses.composite import CompositeLoss
from utils.dataset import create_dataloader


def get_batch(config, device):
    loader = create_dataloader(
        root=config['data']['root'],
        batch_size=config['training']['batch_size'],
        scales=config['data'].get('scales', config['model']['scales']),
        patch_size=config['data']['patch_size'],
        frames=config['data']['frames'],
        workers=0,
        is_train=True,
        data_type=config['data'].get('data_type', 'compressed'),
    )
    for batch in loader:
        return {
            'lr_frames': batch['lr_frames'].to(device),
            'hr': batch['hr'].to(device),
            'scale': batch['scale'],
        }


def run(model, criterion, batch, scaler=None, n_warmup=3, n_iter=10):
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(n_warmup):
        optimizer.zero_grad()
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
                pred = model(batch['lr_frames'][:, 0], batch['lr_frames'][:, 1], batch['lr_frames'][:, 2], scale=batch['scale'])
                loss_dict = criterion(pred, batch['hr'])
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(batch['lr_frames'][:, 0], batch['lr_frames'][:, 1], batch['lr_frames'][:, 2], scale=batch['scale'])
            loss_dict = criterion(pred, batch['hr'])
            loss_dict['total'].backward()
            optimizer.step()

    fwd, loss_t, bwd, total = [], [], [], []
    for _ in range(n_iter):
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        pred = model(batch['lr_frames'][:, 0], batch['lr_frames'][:, 1], batch['lr_frames'][:, 2], scale=batch['scale'])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.time()
        loss_dict = criterion(pred, batch['hr'])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t2 = time.time()
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t3 = time.time()
        optimizer.step()
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t4 = time.time()
        fwd.append(t1 - t0)
        loss_t.append(t2 - t1)
        bwd.append(t3 - t2)
        total.append(t4 - t0)
    return fwd, loss_t, bwd, total


def avg(l):
    return sum(l) / len(l)


def main():
    config = safe_load(open('configs/default.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    batch = get_batch(config, device)
    print(f"batch shapes: lr={batch['lr_frames'].shape}, hr={batch['hr'].shape}, scale={batch['scale']}")

    model = AV1VSR(
        in_channels=3,
        n_features=config['model']['n_features'],
        n_blocks=config['model']['n_blocks'],
        scales=config['model']['scales'],
    ).to(device)
    scaler = torch.amp.GradScaler('cuda') if config['training'].get('amp', False) and device.type == 'cuda' else None

    criterion = CompositeLoss(config['loss'], device=device)
    fwd, loss_t, bwd, total = run(model, criterion, batch, scaler)
    print(f'  forward: {avg(fwd):.3f}s | loss: {avg(loss_t):.3f}s | backward: {avg(bwd):.3f}s | total: {avg(total):.3f}s')


if __name__ == '__main__':
    main()
