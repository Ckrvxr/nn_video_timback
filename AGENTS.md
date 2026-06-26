# Environment

Package/environment manager: **uv**. Never use `pip` or `python` directly.
- Run scripts: `uv run <script>` (e.g., `uv run python ...`)
- Install packages: `uv add <pkg>`
- Run tests: `uv run pytest tests/ -v`
- Lint: `uv run ruff check .`
- Activate venv: `source .venv/bin/activate`

# Model

PyTorch implementation in `core/torch/artrt_r2.py`.
- Data format: **NCHW** `[B, C, H, W]`
- Mamba-2 SSM backbone with VMamba-style SS2D (4-direction scan)
- pixel_unshuffle(r=8) reduces internal resolution to H/8 × W/8
- DetailPath bypass at full resolution preserves texture

# Training

```
uv run python scripts/train_torch.py --config configs/prod.yaml
```

Model runs in **bfloat16** on CUDA (auto-converted in train script).
CUDA runtime path (required for mamba-ssm):
```
LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

# Testing

```
uv run pytest tests/torch/ -v
```
