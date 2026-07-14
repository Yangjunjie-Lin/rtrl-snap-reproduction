from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax.nn import log_softmax, softmax

from rtrl_snap.models.readout import linear_readout
from rtrl_snap.models.vanilla_rnn import forward_sequence, rnn_step
from rtrl_snap.training.losses import sequence_token_accuracy


def flatten_recurrent_params(
    params: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, dict]:
    """
    Flatten recurrent parameters into a 1-D vector.

    Recurrent parameters are ordered as:

        w_xh, w_hh, b_h

    Returns:
        flat_params:
            1-D array of recurrent parameters.

        metadata:
            Dictionary needed to unflatten gradients later.
    """

    w_xh = params["w_xh"]
    w_hh = params["w_hh"]
    b_h = params["b_h"]

    flat_params = jnp.concatenate(
        [
            w_xh.reshape(-1),
            w_hh.reshape(-1),
            b_h.reshape(-1),
        ]
    )

    metadata = {
        "w_xh_shape": w_xh.shape,
        "w_hh_shape": w_hh.shape,
        "b_h_shape": b_h.shape,
        "w_xh_size": w_xh.size,
        "w_hh_size": w_hh.size,
        "b_h_size": b_h.size,
    }

    return flat_params, metadata


def unflatten_recurrent_grads(
    flat_grads: jnp.ndarray,
    metadata: dict,
) -> dict[str, jnp.ndarray]:
    """
    Unflatten a 1-D recurrent gradient vector into a parameter dict.
    """

    w_xh_size = int(metadata["w_xh_size"])
    w_hh_size = int(metadata["w_hh_size"])

    offset_hh = w_xh_size
    offset_bh = offset_hh + w_hh_size

    w_xh_grad = flat_grads[:offset_hh].reshape(metadata["w_xh_shape"])
    w_hh_grad = flat_grads[offset_hh:offset_bh].reshape(metadata["w_hh_shape"])
    b_h_grad = flat_grads[offset_bh:].reshape(metadata["b_h_shape"])

    return {
        "w_xh": w_xh_grad,
        "w_hh": w_hh_grad,
        "b_h": b_h_grad,
    }


def initialize_rtrl_state(
    params: dict[str, jnp.ndarray],
    batch_size: int,
) -> jnp.ndarray:
    """
    Initialize the RTRL influence matrix to zeros.

    Shape:
        [batch_size, hidden_size, num_recurrent_params]
    """

    flat_params, _ = flatten_recurrent_params(params)
    hidden_size = params["b_h"].shape[0]
    num_recurrent_params = flat_params.shape[0]

    return jnp.zeros(
        shape=(batch_size, hidden_size, num_recurrent_params),
        dtype=params["w_xh"].dtype,
    )


def _direct_influence(
    x_t: jnp.ndarray,
    h_prev: jnp.ndarray,
    h_t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute I_t = ∂h_t / ∂θ_recurrent treating h_{t-1} as constant.

    Parameter order matches ``flatten_recurrent_params``: w_xh, w_hh, b_h.

    Returns:
        [batch_size, hidden_size, num_recurrent_params]
    """

    batch_size = x_t.shape[0]
    hidden_size = h_t.shape[1]
    input_size = x_t.shape[1]

    # diag(tanh'(pre_act)) entries: [batch_size, hidden_size]
    dh_dpre = 1.0 - h_t * h_t
    eye_h = jnp.eye(hidden_size, dtype=h_t.dtype)

    # w_xh[i, j] directly affects hidden unit j:
    #   ∂h[j] / ∂w_xh[i, j] = dh_dpre[j] * x[i]
    # values_xh[b, i, j] = dh_dpre[b, j] * x_t[b, i]
    values_xh = dh_dpre[:, None, :] * x_t[:, :, None]
    # Place into rows j, columns i * H + j:
    # [B, H_row, I, H_j] then flatten (I, H_j)
    i_w_xh = values_xh[:, None, :, :] * eye_h[None, :, None, :]
    i_w_xh = i_w_xh.reshape(batch_size, hidden_size, input_size * hidden_size)

    # w_hh[k, j] directly affects hidden unit j:
    #   ∂h[j] / ∂w_hh[k, j] = dh_dpre[j] * h_prev[k]
    values_hh = dh_dpre[:, None, :] * h_prev[:, :, None]
    i_w_hh = values_hh[:, None, :, :] * eye_h[None, :, None, :]
    i_w_hh = i_w_hh.reshape(
        batch_size,
        hidden_size,
        hidden_size * hidden_size,
    )

    # b_h[j] directly affects hidden unit j:
    #   ∂h[j] / ∂b_h[j] = dh_dpre[j]
    i_b_h = dh_dpre[:, :, None] * eye_h[None, :, :]

    return jnp.concatenate([i_w_xh, i_w_hh, i_b_h], axis=-1)


def _dynamics_jacobian(
    params: dict[str, jnp.ndarray],
    h_t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute D_t = ∂h_t / ∂h_{t-1}.

    Returns:
        [batch_size, hidden_size, hidden_size]
        with D[b, i, j] = (1 - h_t[b, i]^2) * W_hh[j, i]
    """

    dh_dpre = 1.0 - h_t * h_t
    # D[b] = diag(dh_dpre[b]) @ W_hh.T
    return dh_dpre[:, :, None] * params["w_hh"].T[None, :, :]


def rtrl_sequence_gradients(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Compute exact RTRL gradients for a full sequence.

    Returns a gradient PyTree with the same keys/shapes as ``params``.
    """

    _, grads = rtrl_loss_and_gradients(
        params=params,
        inputs=inputs,
        targets=targets,
        target_mask=target_mask,
    )
    return grads


def rtrl_loss_and_gradients(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Compute masked sequence loss and exact RTRL gradients.

    The loss matches ``sequence_cross_entropy_loss``. Recurrent gradients
    use the RTRL influence recurrence:

        J_t = I_t + D_t J_{t-1}

    Readout gradients use ordinary per-step output derivatives.
    """

    batch_size, sequence_length, _ = inputs.shape
    hidden_size = params["b_h"].shape[0]

    _, metadata = flatten_recurrent_params(params)
    num_recurrent = (
        int(metadata["w_xh_size"])
        + int(metadata["w_hh_size"])
        + int(metadata["b_h_size"])
    )

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


def create_rtrl_train_step(
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
    Create one RTRL training step.

    Accuracy is computed with a forward pass on the current (pre-update)
    parameters so metrics stay comparable to the BPTT train step.
    """

    def train_step(
        params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        target_mask: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], optax.OptState, dict[str, jnp.ndarray]]:
        loss, grads = rtrl_loss_and_gradients(
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
