from utils.console import console


def get_epoch_weights(epoch_idx: int, schedule: list, fallback: dict) -> dict | None:
    if not schedule:
        return None
    ep = epoch_idx + 1
    for entry in schedule:
        spec = entry['epoch']
        match = (isinstance(spec, int) and spec == ep)
        if isinstance(spec, str):
            if spec.endswith('+') and ep >= int(spec[:-1]):
                match = True
            elif '-' in spec:
                lo, hi = map(int, spec.split('-'))
                if lo <= ep <= hi:
                    match = True
        if match:
            return dict(entry['weights'])
    return None


def log_validation(name: str, psnr: float, ssim: float, vmaf: float, baseline: dict | None):
    vmaf_str = f'  vmaf={vmaf:.4f}' if vmaf > 0 else ''
    if vmaf >= 80:
        vmaf_color = "green"
    elif vmaf >= 60:
        vmaf_color = "yellow"
    else:
        vmaf_color = "red"
    bl = baseline.get(name, {}) if baseline else {}
    psnr_delta = ""
    ssim_delta = ""
    vmaf_delta = ""
    if bl:
        d = psnr - bl['psnr']
        psnr_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
        if 'ssim' in bl:
            d = ssim - bl['ssim']
            ssim_delta = f" ({'+' if d >= 0 else ''}{d:.4f})"
        if 'vmaf' in bl and vmaf > 0:
            d = vmaf - bl['vmaf']
            vmaf_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
    color_msg = (
        f"  <white>{{}}</white>"
        f"  <dim>psnr={{:.2f}}{psnr_delta}  ssim={{:.4f}}{ssim_delta}</dim>"
    )
    if vmaf > 0:
        color_msg += f"  <{vmaf_color}>vmaf={{:.2f}}{vmaf_delta}</{vmaf_color}>"
    console.opt(colors=True).info(color_msg, name, psnr, ssim, vmaf)
