import torch
import torch.nn as nn

class FFTLoss(nn.Module):
    """Frequency-domain L1 Loss using 2D Real Fast Fourier Transform (rfft2d).
    
    This loss enforces the model to align the frequency spectrum (magnitudes and phases)
    of the predicted image with the target ground truth, leading to sharper details
    and better texture reconstruction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Input tensors shape: [B, C, H, W]
        # Compute 2D Fast Fourier Transform on the spatial dimensions (H, W)
        # norm="ortho" scales the FFT so that the forward and backward transforms are energy-preserving,
        # which stabilizes training.
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
        
        # Calculate L1 distance on both real and imaginary components of the spectrum
        loss = torch.mean(torch.abs(pred_fft.real - target_fft.real)) + \
               torch.mean(torch.abs(pred_fft.imag - target_fft.imag))
               
        return loss
