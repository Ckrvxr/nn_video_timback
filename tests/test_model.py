import jax
import jax.numpy as jnp
import pytest
from core import ResBlock, ICtCpNetV2, Fusion, HeavyBranch, LightBranch


# ── helpers ──

def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))


def tree_all_finite(tree):
    return all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(tree))


# ── module-scoped pre-compile cache ──

@pytest.fixture(scope="module")
def compiled(request):
    cache = {}
    key = jax.random.PRNGKey(0)

    def _make(name):
        if name in cache:
            return cache[name]
        if name == "resblock_32":
            m = ResBlock(ch=32)
            x = jax.random.normal(key, (2, 16, 16, 32))
        elif name == "resblock_16":
            m = ResBlock(ch=16)
            x = jax.random.normal(key, (1, 16, 16, 16))
        elif name == "resblock_8_d1":
            m = ResBlock(ch=8, dilation=1)
            x = jax.random.normal(key, (1, 32, 32, 8))
        elif name == "resblock_8_d2":
            m = ResBlock(ch=8, dilation=2)
            x = jax.random.normal(key, (1, 32, 32, 8))
        elif name == "resblock_8_d4":
            m = ResBlock(ch=8, dilation=4)
            x = jax.random.normal(key, (1, 32, 32, 8))
        elif name == "resblock_8_d8":
            m = ResBlock(ch=8, dilation=8)
            x = jax.random.normal(key, (1, 32, 32, 8))
        elif name == "resblock_8_d16":
            m = ResBlock(ch=8, dilation=16)
            x = jax.random.normal(key, (1, 32, 32, 8))
        elif name == "fusion":
            m = Fusion()
            x = jax.random.normal(key, (1, 16, 16, 3))
        elif name == "heavy":
            m = HeavyBranch(out_ch=16)
            x = jax.random.normal(key, (1, 32, 32, 16))
        elif name == "light":
            m = LightBranch(out_ch=16)
            x = jax.random.normal(key, (1, 32, 32, 16))
        elif name.startswith("ictcp_v2_dp_mf_"):
            scale = int(name.split("_")[4])
            m = ICtCpNetV2(scale=scale, detail_path=True, multi_frame=True)
            x = jax.random.normal(key, (1, 64, 64, 9))
        elif name.startswith("ictcp_v2_dp_"):
            scale = int(name.split("_")[3])
            m = ICtCpNetV2(scale=scale, detail_path=True)
            x = jax.random.normal(key, (1, 64, 64, 3))
        elif name.startswith("ictcp_v2_mf_"):
            scale = int(name.split("_")[3])
            m = ICtCpNetV2(scale=scale, multi_frame=True)
            x = jax.random.normal(key, (1, 64, 64, 9))
        elif name.startswith("ictcp_v2_"):
            scale = int(name.split("_")[2])
            m = ICtCpNetV2(scale=scale)
            x = jax.random.normal(key, (1, 64, 64, 3))
        else:
            raise ValueError(name)
        params = m.init(key, x)
        fn = jax.jit(lambda p, x: m.apply(p, x))
        fn(params, x)
        jax.block_until_ready(fn(params, x))
        cache[name] = (params, fn, m)
        return cache[name]

    return _make


# ── ResBlock tests ──

class TestResBlock:
    def test_output_shape(self, compiled):
        params, fn, _ = compiled("resblock_32")
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 32))
        out = fn(params, x)
        assert out.shape == (2, 16, 16, 32)

    def test_gradient_flow(self, compiled):
        params, _, m = compiled("resblock_16")
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 16, 16, 16))

        def loss_fn(p):
            return m.apply(p, x).sum()

        grads = jax.grad(loss_fn)(params)
        assert tree_all_finite(grads)

    def test_dilation(self, compiled):
        for d in [1, 2, 4, 8, 16]:
            params, fn, _ = compiled(f"resblock_8_d{d}")
            x = jax.random.normal(jax.random.PRNGKey(2), (1, 32, 32, 8))
            out = fn(params, x)
            assert out.shape == (1, 32, 32, 8)


# ── Fusion tests ──

class TestFusion:
    def test_output_shape(self, compiled):
        params, fn, _ = compiled("fusion")
        x = jax.random.normal(jax.random.PRNGKey(3), (2, 16, 16, 3))
        out = fn(params, x)
        assert out.shape == (2, 16, 16, 3)


# ── Branch tests ──

class TestBranches:
    def test_heavy_branch_shape(self, compiled):
        params, fn, _ = compiled("heavy")
        x = jax.random.normal(jax.random.PRNGKey(4), (1, 32, 32, 16))
        out = fn(params, x)
        assert out.shape == (1, 32, 32, 16)

    def test_light_branch_shape(self, compiled):
        params, fn, _ = compiled("light")
        x = jax.random.normal(jax.random.PRNGKey(5), (1, 32, 32, 16))
        out = fn(params, x)
        assert out.shape == (1, 32, 32, 16)

    def test_heavy_branch_gradient(self, compiled):
        params, _, m = compiled("heavy")
        x = jax.random.normal(jax.random.PRNGKey(6), (1, 32, 32, 16))

        def loss_fn(p):
            return m.apply(p, x).sum()

        grads = jax.grad(loss_fn)(params)
        assert tree_all_finite(grads)


# ── ICtCpNetV2 tests ──

class TestICtCpNetV2:
    def test_output_shape(self, compiled):
        params, fn, _ = compiled("ictcp_v2_4")
        x = jax.random.normal(jax.random.PRNGKey(7), (2, 64, 64, 3))
        out = fn(params, x)
        assert out.shape == (2, 64, 64, 3)

    def test_forward_backward(self, compiled):
        params, _, m = compiled("ictcp_v2_4")
        x = jax.random.normal(jax.random.PRNGKey(8), (1, 32, 32, 3))

        def loss_fn(p):
            return m.apply(p, x).sum()

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
        assert any(jnp.any(g != 0) for g in leaves if g.size > 0)

    def test_residual(self, compiled):
        params, fn, _ = compiled("ictcp_v2_4")
        x = jax.random.normal(jax.random.PRNGKey(9), (1, 16, 16, 3))
        out = fn(params, x)
        delta = out - x
        assert jnp.max(jnp.abs(delta)) < 1.0

    def test_different_sizes(self, compiled):
        params, fn, _ = compiled("ictcp_v2_4")
        for h, w in [(16, 32), (32, 48), (64, 128)]:
            x = jax.random.normal(jax.random.PRNGKey(10), (1, h, w, 3))
            out = fn(params, x)
            assert out.shape == (1, h, w, 3)

    def test_scales(self, compiled):
        for scale in [1, 2, 4, 8]:
            params, fn, _ = compiled(f"ictcp_v2_{scale}")
            x = jax.random.normal(jax.random.PRNGKey(11), (1, 64, 64, 3))
            out = fn(params, x)
            assert out.shape == (1, 64, 64, 3)

    def test_scale_residuals(self, compiled):
        for scale in [2, 4, 8]:
            params, fn, _ = compiled(f"ictcp_v2_{scale}")
            x = jax.random.normal(jax.random.PRNGKey(12), (1, 64, 64, 3))
            out = fn(params, x)
            delta = out - x
            assert jnp.max(jnp.abs(delta)) < 1.0

    def test_params_reasonable(self, compiled):
        params, _, _ = compiled("ictcp_v2_4")
        n = count_params(params)
        assert 10_000 < n < 100_000


# ── Multi-frame ICtCpNetV2 tests ──

class TestICtCpNetV2MultiFrame:
    def test_output_shape(self, compiled):
        params, fn, _ = compiled("ictcp_v2_mf_4")
        x = jax.random.normal(jax.random.PRNGKey(20), (1, 64, 64, 9))
        out = fn(params, x)
        assert out.shape == (1, 64, 64, 3)

    def test_forward_backward(self, compiled):
        params, _, m = compiled("ictcp_v2_mf_4")
        x = jax.random.normal(jax.random.PRNGKey(21), (1, 32, 32, 9))

        def loss_fn(p):
            return m.apply(p, x).sum()

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
        assert any(jnp.any(g != 0) for g in leaves if g.size > 0)

    def test_residual(self, compiled):
        params, fn, _ = compiled("ictcp_v2_mf_4")
        x = jax.random.normal(jax.random.PRNGKey(22), (1, 16, 16, 9))
        out = fn(params, x)
        frame_center = x[..., 3:6]
        delta = out - frame_center
        assert jnp.max(jnp.abs(delta)) < 1.0

    def test_different_sizes(self, compiled):
        params, fn, _ = compiled("ictcp_v2_mf_4")
        for h, w in [(16, 32), (32, 48)]:
            x = jax.random.normal(jax.random.PRNGKey(23), (1, h, w, 9))
            out = fn(params, x)
            assert out.shape == (1, h, w, 3)


# ── DetailPath tests ──

class TestICtCpNetV2Detail:
    def test_output_shape(self, compiled):
        params, fn, _ = compiled("ictcp_v2_dp_4")
        x = jax.random.normal(jax.random.PRNGKey(30), (1, 64, 64, 3))
        out = fn(params, x)
        assert out.shape == (1, 64, 64, 3)

    def test_forward_backward(self, compiled):
        params, _, m = compiled("ictcp_v2_dp_4")
        x = jax.random.normal(jax.random.PRNGKey(31), (1, 32, 32, 3))

        def loss_fn(p):
            return m.apply(p, x).sum()

        grads = jax.grad(loss_fn)(params)
        leaves = jax.tree.leaves(grads)
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
        assert any(jnp.any(g != 0) for g in leaves if g.size > 0)

    def test_residual(self, compiled):
        params, fn, _ = compiled("ictcp_v2_dp_4")
        x = jax.random.normal(jax.random.PRNGKey(32), (1, 16, 16, 3))
        out = fn(params, x)
        delta = out - x
        assert jnp.max(jnp.abs(delta)) < 1.0

    def test_multi_frame(self, compiled):
        params, fn, _ = compiled("ictcp_v2_dp_mf_4")
        x = jax.random.normal(jax.random.PRNGKey(33), (1, 32, 32, 9))
        out = fn(params, x)
        assert out.shape == (1, 32, 32, 3)
        frame_center = x[..., 3:6]
        delta = out - frame_center
        assert jnp.max(jnp.abs(delta)) < 1.0
