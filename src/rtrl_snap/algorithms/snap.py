from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax.nn import log_softmax, softmax

from rtrl_snap.algorithms.rtrl import (
    _direct_influence,
    _dynamics_jacobian,
    flatten_recurrent_params,
    initialize_rtrl_state,
    unflatten_recurrent_grads,
)
from rtrl_snap.models.readout import linear_readout
from rtrl_snap.models.vanilla_rnn import forward_sequence, rnn_step
from rtrl_snap.training.losses import sequence_token_accuracy


def build_snap1_recurrent_mask(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """
    Build the SnAp-1 structural mask for Vanilla RNN recurrent parameters.

    SnAp-1 keeps only direct one-step influence from a recurrent parameter
    to the hidden unit it immediately feeds:

        w_xh[i, j] -> hidden unit j
        w_hh[k, j] -> hidden unit j
        b_h[j]     -> hidden unit j

    Returns:
        Boolean/float mask with shape [hidden_size, num_recurrent_params].
    """

    w_xh = params["w_xh"]
    w_hh = params["w_hh"]
    b_h = params["b_h"]

    input_size, hidden_size = w_xh.shape
    assert w_hh.shape == (hidden_size, hidden_size)
    assert b_h.shape == (hidden_size,)

    num_recurrent = w_xh.size + w_hh.size + b_h.size
    mask = jnp.zeros((hidden_size, num_recurrent), dtype=w_xh.dtype)

    # w_xh block: flat index for (i, j) is i * H + j
    i_idx = jnp.arange(input_size)[:, None]
    j_idx = jnp.arange(hidden_size)[None, :]
    w_xh_cols = (i_idx * hidden_size + j_idx).reshape(-1)
    w_xh_rows = jnp.broadcast_to(j_idx, (input_size, hidden_size)).reshape(-1)
    mask = mask.at[w_xh_rows, w_xh_cols].set(1.0)

    # w_hh block starts after w_xh
    w_hh_offset = w_xh.size
    k_idx = jnp.arange(hidden_size)[:, None]
    w_hh_cols = w_hh_offset + (k_idx * hidden_size + j_idx).reshape(-1)
    w_hh_rows = jnp.broadcast_to(j_idx, (hidden_size, hidden_size)).reshape(-1)
    mask = mask.at[w_hh_rows, w_hh_cols].set(1.0)

    # b_h block starts after w_xh and w_hh
    b_h_offset = w_xh.size + w_hh.size
    b_h_rows = jnp.arange(hidden_size)
    b_h_cols = b_h_offset + b_h_rows
    mask = mask.at[b_h_rows, b_h_cols].set(1.0)

    return mask


def snap1_loss_and_gradients(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute masked sequence loss and SnAp-1 approximate RTRL gradients.

    Uses the same recurrence as exact RTRL:

        J_t = I_t + D_t J_{t-1}

    then applies the SnAp-1 mask so only direct one-step influence entries
    are retained:

        J_t <- J_t * mask
    """

    batch_size, sequence_length, _ = inputs.shape
    hidden_size = params["b_h"].shape[0]

    _, metadata = flatten_recurrent_params(params)
    num_recurrent = (
        int(metadata["w_xh_size"])
        + int(metadata["w_hh_size"])
        + int(metadata["b_h_size"])
    )

    snap_mask = build_snap1_recurrent_mask(params)  # [H, P]

    h_prev = jnp.zeros((batch_size, hidden_size), dtype=params["w_xh"].dtype)
    j_prev = initialize_rtrl_state(params, batch_size)

    normalizer = jnp.maximum(jnp.sum(target_mask), 1.0)

    flat_recurrent_grad = jnp.zeros((num_recurrent,), dtype=params["w_xh"].dtype)
    w_hy_grad = jnp.zeros_like(params["w_hy"])
    b_y_grad = jnp.zeros_like(params["b_y"])
    total_loss = jnp.asarray(0.0, dtype=params["w_xh"].dtype)

    def step_fn(carry, t):
        (
            h_prev,
            j_prev,
            flat_recurrent_grad,
            w_hy_grad,
            b_y_grad,
            total_loss,
        ) = carry

        x_t = inputs[:, t, :]
        y_t = targets[:, t]
        mask_t = target_mask[:, t]

        h_t = rnn_step(params=params, x_t=x_t, h_prev=h_prev)
        logits_t = linear_readout(params=params, hidden=h_t)

        i_t = _direct_influence(x_t, h_prev, h_t)
        d_t = _dynamics_jacobian(params, h_t)
        j_t = i_t + jnp.einsum("bij,bjk->bik", d_t, j_prev)
        # SnAp-1: keep only direct one-step influence support.
        j_t = j_t * snap_mask[None, :, :]

        log_probs = log_softmax(logits_t, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs,
            y_t[:, None],
            axis=-1,
        ).squeeze(axis=-1)
        total_loss = total_loss + jnp.sum((-target_log_probs) * mask_t)

        probs = softmax(logits_t, axis=-1)
        one_hot = jax.nn.one_hot(
            y_t,
            num_classes=logits_t.shape[-1],
            dtype=logits_t.dtype,
        )
        dlogits = (probs - one_hot) * (mask_t / normalizer)[:, None]

        w_hy_grad = w_hy_grad + h_t.T @ dlogits
        b_y_grad = b_y_grad + jnp.sum(dlogits, axis=0)

        delta_h = dlogits @ params["w_hy"].T
        flat_recurrent_grad = flat_recurrent_grad + jnp.einsum(
            "bi,bik->k",
            delta_h,
            j_t,
        )

        new_carry = (
            h_t,
            j_t,
            flat_recurrent_grad,
            w_hy_grad,
            b_y_grad,
            total_loss,
        )
        return new_carry, None

    init_carry = (
        h_prev,
        j_prev,
        flat_recurrent_grad,
        w_hy_grad,
        b_y_grad,
        total_loss,
    )

    (
        _h_final,
        _j_final,
        flat_recurrent_grad,
        w_hy_grad,
        b_y_grad,
        total_loss,
    ), _ = jax.lax.scan(
        step_fn,
        init_carry,
        jnp.arange(sequence_length),
    )

    loss = total_loss / normalizer
    recurrent_grads = unflatten_recurrent_grads(flat_recurrent_grad, metadata)

    grads = {
        "w_xh": recurrent_grads["w_xh"],
        "w_hh": recurrent_grads["w_hh"],
        "b_h": recurrent_grads["b_h"],
        "w_hy": w_hy_grad,
        "b_y": b_y_grad,
    }

    return loss, grads


def create_snap1_train_step(
    optimizer: optax.GradientTransformation,
) -> Callable[
    [
        dict[str, jnp.ndarray],
        optax.OptState,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    tuple[dict[str, jnp.ndarray], optax.OptState, dict[str, jnp.ndarray]],
]:
    """
    Create one SnAp-1 training step.

    Accuracy is computed with a forward pass on the current (pre-update)
    parameters.
    """

    def train_step(
        params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        target_mask: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], optax.OptState, dict[str, jnp.ndarray]]:
        loss, grads = snap1_loss_and_gradients(
            params=params,
            inputs=inputs,
            targets=targets,
            target_mask=target_mask,
        )

        outputs = forward_sequence(params=params, inputs=inputs)
        accuracy = sequence_token_accuracy(
            logits=outputs["logits"],
            targets=targets,
            target_mask=target_mask,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }

        return params, opt_state, metrics

    return train_step
