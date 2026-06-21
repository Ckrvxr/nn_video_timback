"""
Compare original vs JPEG Q=50 with Sobel + Wavelet + FFT.
"""
import sys
import io
import torch
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

DESKTOP = Path.home() / "Desktop"
DEFAULT_IMG = DESKTOP / "IMG_20260521_193539.jpg"

def haar_dwt(x):
    B, C, H, W = x.shape
    H2, W2 = H // 2 * 2, W // 2 * 2
    x = x[:, :, :H2, :W2]
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) / 2
    lh = (a - b + c - d) / 2
    hl = (a + b - c - d) / 2
    hh = (a - b - c + d) / 2
    return ll, lh, hl, hh

def sobel_np(img_np):
    """Sobel magnitude from RGB numpy array [0,1]."""
    from PIL import Image, ImageFilter
    g_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    g = Image.fromarray(g_u8).convert("L")
    gx = np.array(g.filter(ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)), dtype=np.float64)
    gy = np.array(g.filter(ImageFilter.Kernel((3, 3), [-1, -2, -1, 0, 0, 0, 1, 2, 1], scale=1)), dtype=np.float64)
    mag = np.sqrt(gx**2 + gy**2)
    return (mag / mag.max()).astype(np.float32)

def build_wavelet_composite(ll, lh, hl, hh):
    """Arrange 4 subbands into one image: LL top-left, LH top-right, HL bottom-left, HH bottom-right."""
    def norm(s):
        return ((s - s.min()) / (s.max() - s.min() + 1e-8) * 255).clip(0, 255).astype(np.uint8)
    ll_u8 = norm(ll)
    lh_u8 = norm(lh)
    hl_u8 = norm(hl)
    hh_u8 = norm(hh)
    Hh, Wh = ll_u8.shape
    comp = np.zeros((Hh * 2, Wh * 2), dtype=np.uint8)
    comp[:Hh, :Wh] = ll_u8
    comp[:Hh, Wh:] = lh_u8
    comp[Hh:, :Wh] = hl_u8
    comp[Hh:, Wh:] = hh_u8
    return comp

def fft_magnitude(gray_np):
    f = np.fft.fft2(gray_np)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag_log = np.log1p(mag)
    return (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-8)

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    w2, h2 = w // 2 * 2, h // 2 * 2
    if (w2, h2) != (w, h):
        img = img.crop((0, 0, w2, h2))
        w, h = w2, h2

    arr = np.array(img, dtype=np.float32) / 255.0

    # ── JPEG Q=50 ──
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB")
    jpeg_arr = np.array(jpeg, dtype=np.float32) / 255.0

    # ── Sobel ──
    sobel_orig = sobel_np(arr)
    sobel_jpeg = sobel_np(jpeg_arr)

    # ── Grayscale for wavelet/FFT ──
    def to_gray(a):
        return 0.299 * a[:,:,0] + 0.587 * a[:,:,1] + 0.114 * a[:,:,2]

    gray_orig = to_gray(arr)
    gray_jpeg = to_gray(jpeg_arr)

    # ── Wavelet ──
    def wavelet_analysis(g):
        t = torch.from_numpy(g).unsqueeze(0).unsqueeze(0)
        ll, lh, hl, hh = haar_dwt(t)
        return ll.squeeze().numpy(), lh.squeeze().numpy(), hl.squeeze().numpy(), hh.squeeze().numpy()

    wcomp_orig = build_wavelet_composite(*wavelet_analysis(gray_orig))
    wcomp_jpeg = build_wavelet_composite(*wavelet_analysis(gray_jpeg))

    # ── FFT ──
    fft_orig = fft_magnitude(gray_orig)
    fft_jpeg = fft_magnitude(gray_jpeg)

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    fig.suptitle("Original vs JPEG Q=50 — Sobel · Wavelet · FFT", fontsize=16, y=0.98)

    titles = [
        ("Original", arr),
        ("JPEG Q=50", jpeg_arr),
        ("Sobel (Original)", sobel_orig, "gray"),
        ("Sobel (JPEG Q=50)", sobel_jpeg, "gray"),
        ("Wavelet 1-level (Original)", wcomp_orig, "gray"),
        ("Wavelet 1-level (JPEG Q=50)", wcomp_jpeg, "gray"),
        ("FFT magnitude (Original)", fft_orig, "inferno"),
        ("FFT magnitude (JPEG Q=50)", fft_jpeg, "inferno"),
    ]

    for i, spec in enumerate(titles):
        ax = axes.flat[i]
        name = spec[0]
        data = spec[1]
        cmap = spec[2] if len(spec) > 2 else None
        ax.imshow(data, cmap=cmap)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    out_path = DESKTOP / "wavelet_fft_visualization.png"
    plt.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")
    print(f"Layout: Original|JPEG -> Sobel|Wavelet|FFT (4×2 grid)")

if __name__ == "__main__":
    main()
