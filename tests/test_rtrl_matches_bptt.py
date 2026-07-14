from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rtrl_snap.algorithms.bptt import bptt_loss
from rtrl_snap.algorithms.rtrl import rtrl_loss_and_gradients
from rtrl_snap.models.vanilla_rnn import initialize_vanilla_rnn_params
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch


def _build_tiny_setup(seed: int = 0):
    """
    Build a tiny Copy Task batch and Vanilla RNN for gradient checks.
    """

    batch_size = 2
    copy_length = 3
    hidden_size = 4

    key = jax.random.PRNGKey(seed)
    key, batch_key, model_key = jax.random.split(key, 3)

    batch = generate_copy_batch(
        key=batch_key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    params = initialize_vanilla_rnn_params(
        key=model_key,
        input_size=COPY_VOCAB_SIZE,
        hidden_size=hidden_size,
        output_size=COPY_VOCAB_SIZE,
    )

    return params, batch


def test_rtrl_loss_matches_bptt_loss():
    params, batch = _build_tiny_setup()

    loss_bptt = bptt_loss(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )
    loss_rtrl, _ = rtrl_loss_and_gradients(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )

    np.testing.assert_allclose(
        np.asarray(loss_rtrl),
        np.asarray(loss_bptt),
        rtol=1e-4,
        atol=1e-4,
    )


def test_rtrl_gradients_have_correct_shapes():
    params, batch = _build_tiny_setup()

    _, grads = rtrl_loss_and_gradients(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )

    assert set(grads.keys()) == set(params.keys())
    for key in params:
        assert grads[key].shape == params[key].shape


def test_rtrl_gradients_match_bptt_gradients():
    params, batch = _build_tiny_setup()

    grads_bptt = jax.grad(bptt_loss)(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )
    _, grads_rtrl = rtrl_loss_and_gradients(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )

    for key in params:
        np.testing.assert_allclose(
            np.asarray(grads_rtrl[key]),
            np.asarray(grads_bptt[key]),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Gradient mismatch for parameter '{key}'",
        )
