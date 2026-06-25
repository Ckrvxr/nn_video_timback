"""Composite training score: higher is better, above baseline → positive."""


def composite_score(metrics: dict, baseline: dict | None, train_loss: float) -> float:
    if not metrics:
        return -train_loss * 10

    bl = baseline or {}
    scores = []

    for name, m in metrics.items():
        b = bl.get(name, {})
        psnr_gain = m.get('psnr', 0.0) - b.get('psnr', 0.0)
        ssim_gain = m.get('ssim', 0.0) - b.get('ssim', 0.0)
        vmaf_gain = m.get('vmaf', 0.0) - b.get('vmaf', 0.0)
        gain = (psnr_gain / 10.0 + ssim_gain + vmaf_gain / 100.0) / 3.0
        scores.append(gain)

    avg_gain = sum(scores) / max(len(scores), 1)
    return avg_gain - train_loss * 10.0
