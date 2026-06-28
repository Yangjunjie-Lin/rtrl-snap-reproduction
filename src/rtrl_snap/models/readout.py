from __future__ import annotations

import jax.numpy as jnp


def linear_readout(
    params: dict[str, jnp.ndarray],
    hidden: jnp.ndarray,
) -> jnp.ndarray:
    """
    Apply a linear readout layer to hidden states.

    Args:
        params:
            Model parameter dictionary containing:
                w_hy: [hidden_size, output_size]
                b_y: [output_size]

        hidden:
            Hidden states with shape [batch_size, hidden_size].

    Returns:
        Logits with shape [batch_size, output_size].
    """

    return hidden @ params["w_hy"] + params["b_y"]