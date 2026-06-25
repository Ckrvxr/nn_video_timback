import jax.numpy as jnp

from utils.colorspace import ictcp_to_rgb
from utils.loss.charbonnier import charbonnier_loss
from utils.loss.haarpsi import haarpsi_loss


class CompositeLoss:
    """Weighted combination of losses. Stateless — weights are passed at call time."""

    def __init__(self, config: dict):
        self.w_char = config.get('charbonnier', 1.0)
        self.w_rgb = config.get('rgb', 0.0)
        self.w_haarpsi = config.get('haarpsi', 0.0)

    def __call__(self, pred, target):
        losses = {}
        losses['char'] = charbonnier_loss(pred, target) * self.w_char
        if self.w_rgb > 0:
            pred_rgb = ictcp_to_rgb(pred)
            target_rgb = ictcp_to_rgb(target)
            losses['rgb'] = jnp.mean(jnp.abs(pred_rgb - target_rgb)) * self.w_rgb
        else:
            losses['rgb'] = jnp.array(0.0)
        if self.w_haarpsi > 0:
            losses['haarpsi'] = haarpsi_loss(pred, target) * self.w_haarpsi
        else:
            losses['haarpsi'] = jnp.array(0.0)
        losses['total'] = sum(losses.values())
        return losses

    def update_weights(self, weights: dict):
        weight_map = {
            'charbonnier': 'w_char',
            'rgb': 'w_rgb',
            'haarpsi': 'w_haarpsi',
        }
        for k, v in weights.items():
            attr = weight_map.get(k)
            if attr:
                setattr(self, attr, v)
