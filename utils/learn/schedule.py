from utils.console import console


def get_epoch_weights(epoch_idx: int, schedule: list, fallback: dict) -> dict | None:
    if not schedule:
        return None
    ep = epoch_idx + 1
    for entry in schedule:
        spec = entry["epoch"]
        match = isinstance(spec, int) and spec == ep
        if isinstance(spec, str):
            if spec.endswith("+") and ep >= int(spec[:-1]):
                match = True
            elif "-" in spec:
                lo, hi = map(int, spec.split("-"))
                if lo <= ep <= hi:
                    match = True
        if match:
            return dict(entry["weights"])
    return None


def log_validation(
    name: str,
    psnr: float,
    ssim: float,
    vmaf: float,
    baseline: dict | None,
    vgg: float | None = None,
):
    bl = baseline.get(name, {}) if baseline else {}
    psnr_delta = ""
    ssim_delta = ""
    vmaf_delta = ""
    if bl:
        d = psnr - bl["psnr"]
        psnr_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
        if "ssim" in bl:
            d = ssim - bl["ssim"]
            ssim_delta = f" ({'+' if d >= 0 else ''}{d:.4f})"
        if "vmaf" in bl and vmaf > 0:
            d = vmaf - bl["vmaf"]
            vmaf_delta = f" ({'+' if d >= 0 else ''}{d:.2f})"
    vmaf_color = "green" if vmaf >= 80 else ("yellow" if vmaf >= 60 else "red")
    vgg_str = ""
    vgg_delta = ""
    if vgg is not None:
        if bl and "vgg" in bl:
            d = vgg - bl["vgg"]
            vgg_delta = f" ({'+' if d >= 0 else ''}{d:.6f})"
        vgg_str = f"  vgg={vgg:.6f}{vgg_delta}"

    msg = f"  [white]{name}[/]"
    msg += f"  [dim]psnr={psnr:.2f}{psnr_delta}  ssim={ssim:.6f}{ssim_delta}[/]"
    if vmaf > 0:
        msg += f"  [{vmaf_color}]vmaf={vmaf:.6f}{vmaf_delta}[/{vmaf_color}]"
    msg += vgg_str
    console.info(msg)
