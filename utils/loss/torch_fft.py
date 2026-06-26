import torch
import torch.nn.functional as F


def fft_loss(pred: torch.Tensor, target: torch.Tensor,
             alpha: float = 1.0) -> torch.Tensor:
    """FFT magnitude L1 loss.

    Args:
        pred, target: [B, C, H, W] NCHW tensors.
        alpha: scaling factor.

    Returns:
        Scalar loss.
    """
    pred_f = pred.float()
    target_f = target.float()
    pred_fft = torch.fft.rfft2(pred_f, norm='ortho')
    target_fft = torch.fft.rfft2(target_f, norm='ortho')
    loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return loss * alpha
