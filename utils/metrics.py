import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = np.clip((img1 + 1) * 127.5, 0, 255).astype(np.uint8)
    img2 = np.clip((img2 + 1) * 127.5, 0, 255).astype(np.uint8)
    return psnr(img1, img2, data_range=255)


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    img1 = np.clip((img1 + 1) * 127.5, 0, 255).astype(np.uint8)
    img2 = np.clip((img2 + 1) * 127.5, 0, 255).astype(np.uint8)
    return ssim(img1, img2, channel_axis=-1, data_range=255)
