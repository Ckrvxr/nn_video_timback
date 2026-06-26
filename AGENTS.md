# Environment

Package/environment manager: **uv**. Never use `pip` or `python` directly.
- Run scripts: `uv run <script>` (e.g., `uv run python ...`)
- Install packages: `uv add <pkg>`
- Run tests: `uv run pytest tests/ -v`
- Lint: `uv run ruff check .`
- Activate venv: `source .venv/bin/activate`

# Model

Single model file: `core/torch/artrt.py` (`ArtRT` class). No backward compatibility.
- Data format: **NCHW** `[B, C, H, W]`, values are **linear RGB [0, 1]** float32/float16.
- Pure CNN (no SSM, no attention, no wavelet). `NAFBlock` × N stacked in 3-level U-Net.
- pixel_unshuffle(r=8) reduces internal resolution to H/8 × W/8.
- Default: `dim=48, n1=2, n2=6, n3=8, nmid=12` (3.10M params) — tuned for 4K@30Hz
- Architecture: `head(192→C,1×1)` → `NAFUNet(C)` → `tail(C→192,1×1)` with pixel_unshuffle/shuffle.
- NAFUNet: 3-level encoder-decoder with pixel_unshuffle/shuffle down/up and skip connections.
- Each `NAFBlock`: `LayerNorm → expand(1×1, C→2C) → DWConv(3×3, groups=2C) → SimpleGate(split×multiply) → SCA → post(1×1, C→C) + shortcut`

# Data pipe

Canonical format:
- **yuv444p12le** (4:4:4, 12-bit, full range)
- **BT.2020 NC** matrix, **BT.709** transfer (gamma), BT.709 primaries
- ffmpeg handles all input formats: `-vf "zscale=matrix=bt2020nc:transfer=bt709:primaries=bt709:range=full" -pix_fmt yuv444p12le`

Conversion to linear RGB:
- BT.2020 YUV→RGB matrix → BT.709 gamma 2.2 expansion → clip [0, 1]

# Training

```
uv run python scripts/train_torch.py --config configs/production.yaml
```

Model runs in **float16** on CUDA (auto-converted in train script).

# Testing

```
uv run pytest tests/torch/ -v
```
