import numpy as np
import torch

from ..memory import check_gpu_allocation


def yuv_to_ictcp_np_batch(yuv_batch: np.ndarray) -> torch.Tensor:
    B, H, W, _ = yuv_batch.shape
    bits = 8
    if yuv_batch.dtype == np.uint16:
        max_val = int(yuv_batch.max())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    t = torch.from_numpy(yuv_batch.astype(np.float32, copy=False)).cuda()
    t = t.permute(0, 3, 1, 2).contiguous()
    t[:, 0:1] = t[:, 0:1] / peak * 255.0
    t[:, 1:] = (t[:, 1:] - center) / peak * 255.0 + 128.0
    t = t / 127.5 - 1.0

    from utils.color_space import yuv_to_ictcp
    with torch.no_grad():
        return yuv_to_ictcp(t).half()


def batch_yuv_to_ictcp(yuv: torch.Tensor, device: torch.device) -> torch.Tensor:
    orig_ndim = yuv.ndim
    if orig_ndim == 5:
        orig_B, F, H, W, C = yuv.shape
        yuv = yuv.flatten(0, 1)

    B, H, W, C = yuv.shape

    est_peak = B * H * W * C * 4 * 5
    if not check_gpu_allocation(est_peak, device, 0.85):
        pass

    bits = 8
    if yuv.dtype == torch.uint16:
        max_val = int(yuv.to(torch.int32).max().item())
        bits = 12 if max_val > 1023 else 10
    peak = float((1 << bits) - 1)
    center = float(1 << (bits - 1))

    yuv = yuv.to(device=device, dtype=torch.float32, non_blocking=True)
    yuv = yuv.permute(0, 3, 1, 2).contiguous()
    yuv[:, 0:1] = yuv[:, 0:1] / peak * 255.0
    yuv[:, 1:] = (yuv[:, 1:] - center) / peak * 255.0 + 128.0
    yuv = yuv / 127.5 - 1.0

    from utils.color_space import yuv_to_ictcp
    with torch.no_grad():
        ictcp = yuv_to_ictcp(yuv).half()

    if orig_ndim == 5:
        ictcp = ictcp.view(orig_B, F, 3, H, W)
    return ictcp
