import torch
from utils.color_space import yuv_to_rgb, ictcp_to_yuv
from utils.data.ictcp import batch_yuv_to_ictcp
from utils.evaluation.metrics import calculate_psnr_batch, calculate_ssim_batch


@torch.no_grad()
def validate(model, val_loader, device, num_vmaf_samples=0, baseline=False):
    from components import RealTimeUNet4K_PureCNN

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_fpsnr = 0.0
    n = 0
    n_vmaf = 0
    is_pure_cnn = isinstance(model, RealTimeUNet4K_PureCNN)

    for batch in val_loader:
        lr = batch_yuv_to_ictcp(batch['lr_frames'], device)
        hr = batch_yuv_to_ictcp(batch['hr'], device)

        if baseline:
            pred_yuv = ictcp_to_yuv(lr[:, lr.size(1) - 1])
        else:
            if is_pure_cnn:
                x = lr.reshape(lr.size(0), -1, lr.size(3), lr.size(4)).float().to(memory_format=torch.channels_last)
                pred = model(x)
            else:
                last = lr.size(1) - 1
                pred = model(lr[:, last-2], lr[:, last-1], lr[:, last])
            pred_yuv = ictcp_to_yuv(pred)
        hr_yuv = ictcp_to_yuv(hr)
        pred_rgb = yuv_to_rgb(pred_yuv)
        hr_rgb = yuv_to_rgb(hr_yuv)

        batch_size = lr.size(0)
        total_psnr += calculate_psnr_batch(pred_rgb, hr_rgb).sum().item()
        total_ssim += calculate_ssim_batch(pred_rgb, hr_rgb).sum().item()
        n += batch_size

        remaining = num_vmaf_samples - n_vmaf
        if remaining > 0:
            from utils.evaluation.vmaf import compute_vmaf
            for b in range(min(batch_size, remaining)):
                total_fpsnr += compute_vmaf(pred_yuv[b:b+1], hr_yuv[b:b+1])
                n_vmaf += 1

    avg_vmaf = total_fpsnr / max(1, n_vmaf)
    return total_psnr / max(1, n), total_ssim / max(1, n), avg_vmaf
