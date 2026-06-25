"""Composite training score: higher is better, above baseline → positive."""


def composite_score(metrics: dict, baseline: dict | None, train_loss: float) -> float:
    """综合得分。基于验证指标对 baseline 的相对改善，减去训练 loss 惩罚。

    Parameters
    ----------
    metrics : dict
        ``{name: {'psnr': ..., 'ssim': ..., 'vmaf': ...}}`` per validation set.
    baseline : dict | None
        Same structure as metrics, from ``compute_baseline``.
    train_loss : float
        Epoch-average training loss.

    Returns
    -------
    float
        Positive means better than baseline on average.
    """
    if not metrics:
        return -train_loss * 10

    bl = baseline or {}
    scores = []

    for name, m in metrics.items():
        b = bl.get(name, {})
        psnr_gain = m['psnr'] - b.get('psnr', 0.0)
        ssim_gain = m['ssim'] - b.get('ssim', 0.0)
        vmaf_gain = m['vmaf'] - b.get('vmaf', 0.0)
        # Normalise each gain to roughly [−1, +1] then average.
        gain = (psnr_gain / 10.0 + ssim_gain + vmaf_gain / 100.0) / 3.0
        scores.append(gain)

    avg_gain = sum(scores) / max(len(scores), 1)
    # Penalty: training loss > 0.1 drags the score down by ~1 per 0.1
    return avg_gain - train_loss * 10.0
