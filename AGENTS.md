# Environment

Package/environment manager: **uv**. Never use `pip` or `python` directly.
- Run scripts: `uv run <script>` (e.g., `uv run python ...`, `uv run pytest ...`)
- Install packages: `uv add <pkg>`
- Run tests: `uv run pytest tests/ -v`
- Lint: `uv run ruff check .`
- Activate venv: `source .venv/bin/activate`

# Model

JAX/Flax implementation in `components/timback_cnn.py`.
- Data format: **NHWC** `[B, H, W, C]` (NOT PyTorch's NCHW)
- Forward pass: `model.apply(params, x)` (params from `model.init(rng, x)`)
- JIT compile: `jax.jit(lambda p, x: model.apply(p, x))`
- ICtCpNet(scale=N) controls pixel_unshuffle/shuffle factor

# Testing

JAX XLA compilation is slow. Disable cuDNN autotuning for faster iteration:
```
XLA_FLAGS="--xla_gpu_autotune_level=0" uv run pytest tests/ -v
```
