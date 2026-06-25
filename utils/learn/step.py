import jax
import jax.numpy as jnp
import optax


def has_nan_or_inf(tree):
    return any(
        jnp.any(jnp.isnan(g)) | jnp.any(jnp.isinf(g))
        for g in jax.tree.leaves(tree)
    )


def make_train_step(model, optimizer, criterion):
    """Create a compiled JAX training step function."""

    def loss_fn(params, x, target):
        pred = model.apply(params, x)
        losses = criterion(pred, target)
        return losses['total'], losses

    def _apply_update(params, opt_state, grads, lr):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        # optax.scale_by_adam returns the ascent direction; negate to descend.
        updates = jax.tree_util.tree_map(lambda u: u * (-lr), updates)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def _skip_update(params, opt_state, grads, lr):
        return params, opt_state

    @jax.jit
    def train_step(params, opt_state, x, target, lr):
        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, target)
        # Detect non-finite loss or gradients and skip the update if anything is broken.
        loss_bad = jnp.logical_or(jnp.isnan(loss), jnp.isinf(loss))
        grad_bad = jax.tree_util.tree_reduce(
            lambda a, b: jnp.logical_or(a, b),
            jax.tree_util.tree_map(
                lambda g: jnp.logical_or(jnp.any(jnp.isnan(g)), jnp.any(jnp.isinf(g))),
                grads,
            ),
        )
        skip = jnp.logical_or(loss_bad, grad_bad)
        # Use cond so the JIT graph stays clean even when we skip.
        params, opt_state = jax.lax.cond(
            skip,
            _skip_update,
            _apply_update,
            params, opt_state, grads, lr,
        )
        return params, opt_state, loss, losses, skip

    return train_step
