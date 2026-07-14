from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from rtrl_snap.models.readout import linear_readout


def validate_vanilla_rnn_args(
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> None:
    """
    Validate Vanilla RNN model dimensions.
    """

    if input_size <= 0:
        raise ValueError("input_size must be greater than 0.")

    if hidden_size <= 0:
        raise ValueError("hidden_size must be greater than 0.")

    if output_size <= 0:
        raise ValueError("output_size must be greater than 0.")


def initialize_vanilla_rnn_params(
    key: jax.Array,
    input_size: int,
    hidden_size: int,
    output_size: int,
) -> dict[str, jnp.ndarray]:
    """
    Initialize parameters for a Vanilla RNN.

    Model equations:

        h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
        y_t = h_t W_hy + b_y

    Args:
        key:
            JAX random key.

        input_size:
            Input vocabulary size or feature dimension.

        hidden_size:
            Number of recurrent hidden units.

        output_size:
            Output vocabulary size or number of classes.

    Returns:
        Parameter dictionary.
    """

    validate_vanilla_rnn_args(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    key_xh, key_hh, key_hy = jax.random.split(key, 3)

    w_xh_scale = 1.0 / math.sqrt(input_size)
    w_hh_scale = 1.0 / math.sqrt(hidden_size)
    w_hy_scale = 1.0 / math.sqrt(hidden_size)

    params = {
        "w_xh": jax.random.normal(
            key_xh,
            shape=(input_size, hidden_size),
        )
        * w_xh_scale,
        "w_hh": jax.random.normal(
            key_hh,
            shape=(hidden_size, hidden_size),
        )
        * w_hh_scale,
        "b_h": jnp.zeros(
            shape=(hidden_size,),
            dtype=jnp.float32,
        ),
        "w_hy": jax.random.normal(
            key_hy,
            shape=(hidden_size, output_size),
        )
        * w_hy_scale,
        "b_y": jnp.zeros(
            shape=(output_size,),
            dtype=jnp.float32,
        ),
    }

    return params


def rnn_step(
    params: dict[str, jnp.ndarray],
    x_t: jnp.ndarray,
    h_prev: jnp.ndarray,
) -> jnp.ndarray:
    """
    Run one Vanilla RNN step.

    Args:
        params:
            Model parameter dictionary.

        x_t:
            Input at current timestep with shape [batch_size, input_size].

        h_prev:
            Previous hidden state with shape [batch_size, hidden_size].

    Returns:
        Next hidden state with shape [batch_size, hidden_size].
    """

    pre_activation = (
        x_t @ params["w_xh"]
        + h_prev @ params["w_hh"]
        + params["b_h"]
    )

    return jnp.tanh(pre_activation)


def forward_sequence(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    initial_hidden: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """
    Run a Vanilla RNN over a full input sequence.

    Args:
        params:
            Model parameter dictionary.

        inputs:
            One-hot or continuous input sequence with shape
            [batch_size, sequence_length, input_size].

        initial_hidden:
            Optional initial hidden state with shape
            [batch_size, hidden_size].
            If None, a zero initial hidden state is used.

    Returns:
        Dictionary containing:

            hidden_states:
                [batch_size, sequence_length, hidden_size]

            logits:
                [batch_size, sequence_length, output_size]

            final_hidden:
                [batch_size, hidden_size]
    """

    batch_size = inputs.shape[0]
    hidden_size = params["b_h"].shape[0]

    if initial_hidden is None:
        initial_hidden = jnp.zeros(
            shape=(batch_size, hidden_size),
            dtype=jnp.float32,
        )

    time_major_inputs = jnp.swapaxes(inputs, 0, 1)

    def scan_step(
        h_prev: jnp.ndarray,
        x_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        h_next = rnn_step(
            params=params,
            x_t=x_t,
            h_prev=h_prev,
        )

        logits_t = linear_readout(
            params=params,
            hidden=h_next,
        )

        return h_next, (h_next, logits_t)

    final_hidden, (time_major_hidden_states, time_major_logits) = jax.lax.scan(
        scan_step,
        initial_hidden,
        time_major_inputs,
    )

    hidden_states = jnp.swapaxes(time_major_hidden_states, 0, 1)
    logits = jnp.swapaxes(time_major_logits, 0, 1)

    return {
        "hidden_states": hidden_states,
        "logits": logits,
        "final_hidden": final_hidden,
    }