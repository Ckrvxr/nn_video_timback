import jax.numpy as jnp


def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    return jnp.mean(jnp.sqrt(diff * diff + eps))
