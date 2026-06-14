import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AV1CompressedVideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        scales: list[int] = None,
        patch_size: int = 256,
        frames: int = 3,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.scales = scales or [1, 2, 3, 4, 5, 6]
        self.patch_size = patch_size
        self.frames = frames
        self.is_train = is_train
        self.samples = self._scan()

    def _scan(self) -> list[dict]:
        samples = []
        hr_dir = self.root / 'HR'
        for video in sorted(hr_dir.iterdir()):
            if not video.is_dir():
                continue
            hr_frames = sorted(video.glob('*.png'))
            for encoder in ['svt', 'aom']:
                for crf in [30, 40, 50, 60]:
                    lr_dir = self.root / f'{encoder}_crf{crf}' / video.name
                    if not lr_dir.exists():
                        continue
                    for i in range(len(hr_frames)):
                        sample = {
                            'hr_frames': hr_frames,
                            'lr_dir': lr_dir,
                            'center_idx': i,
                        }
                        samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        center = sample['center_idx']
        half = self.frames // 2

        hr_frames = sample['hr_frames']
        lr_dir = sample['lr_dir']

        scale = random.choice(self.scales) if self.is_train else 4

        lr_frames = []
        for i in range(center - half, center + half + 1):
            i_clamped = max(0, min(i, len(hr_frames) - 1))
            hr_path = str(hr_frames[i_clamped])
            hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            h, w = hr.shape[:2]

            lr_h, lr_w = h, w
            if scale > 1:
                lr_h = h // scale
                lr_w = w // scale
                lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            else:
                lr = hr.copy()

            lr_patch = self._crop_or_pad(lr, self.patch_size)
            lr_frames.append(lr_patch)

        hr_patch = self._crop_or_pad(hr, self.patch_size)

        lr_tensor = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_tensor = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {
            'lr_frames': lr_tensor,
            'hr': hr_tensor,
            'scale': scale,
        }

    def _crop_or_pad(self, img: np.ndarray, size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if self.is_train:
            y = random.randint(0, max(0, h - size))
            x = random.randint(0, max(0, w - size))
            return img[y:y + size, x:x + size]
        else:
            return img


class CleanVSRDataset(AV1CompressedVideoDataset):
    def _scan(self) -> list[dict]:
        samples = []
        hr_dir = self.root / 'HR'
        for video in sorted(hr_dir.iterdir()):
            if not video.is_dir():
                continue
            hr_frames = sorted(video.glob('*.png'))
            for i in range(len(hr_frames)):
                samples.append({
                    'hr_frames': hr_frames,
                    'center_idx': i,
                })
        return samples

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        center = sample['center_idx']
        half = self.frames // 2

        hr_frames = sample['hr_frames']
        scale = random.choice(self.scales) if self.is_train else 4

        lr_frames = []
        for i in range(center - half, center + half + 1):
            i_clamped = max(0, min(i, len(hr_frames) - 1))
            hr = cv2.imread(str(hr_frames[i_clamped]), cv2.IMREAD_COLOR)
            hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            h, w = hr.shape[:2]

            if scale > 1:
                lr_h, lr_w = h // scale, w // scale
                lr = cv2.resize(hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            else:
                lr = hr.copy()
            lr = self._crop_or_pad(lr, self.patch_size)
            lr_frames.append(lr)

        hr_patch = self._crop_or_pad(hr, self.patch_size)
        lr_tensor = torch.from_numpy(np.stack(lr_frames, axis=0)).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        hr_tensor = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 127.5 - 1.0

        return {
            'lr_frames': lr_tensor,
            'hr': hr_tensor,
            'scale': scale,
        }


def create_dataloader(
    root: str,
    batch_size: int = 16,
    scales: list[int] = None,
    patch_size: int = 256,
    frames: int = 3,
    workers: int = 8,
    is_train: bool = True,
    data_type: str = 'compressed',
) -> DataLoader:
    cls = AV1CompressedVideoDataset if data_type == 'compressed' else CleanVSRDataset
    dataset = cls(
        root=root,
        scales=scales,
        patch_size=patch_size,
        frames=frames,
        is_train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=True,
        drop_last=is_train,
    )
