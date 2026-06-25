import jax
import jax.numpy as jnp
import optax


def has_nan_or_inf_params(params):
    return any(
        jnp.any(jnp.isnan(g)) or jnp.any(jnp.isinf(g))
        for g in jax.tree.leaves(params)
    )


def make_train_step(model, optimizer, criterion):
    """Create a compiled JAX training step function."""

    def loss_fn(params, x, target):
        pred = model.apply(params, x)
        losses = criterion(pred, target)
        return losses['total'], losses

    @jax.jit
    def train_step(params, opt_state, x, target):
        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, losses

    return train_step
