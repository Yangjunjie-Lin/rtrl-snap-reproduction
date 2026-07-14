from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from rtrl_snap.algorithms.rtrl import flatten_recurrent_params
from rtrl_snap.algorithms.snap import (
    build_snap1_recurrent_mask,
    snap1_loss_and_gradients,
)
from rtrl_snap.models.vanilla_rnn import initialize_vanilla_rnn_params
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch


def _tiny_params(seed: int = 0):
    key = jax.random.PRNGKey(seed)
    return initialize_vanilla_rnn_params(
        key=key,
        input_size=COPY_VOCAB_SIZE,
        hidden_size=4,
        output_size=COPY_VOCAB_SIZE,
    )


def test_snap1_mask_shape():
    params = _tiny_params()
    mask = build_snap1_recurrent_mask(params)

    _, metadata = flatten_recurrent_params(params)
    num_recurrent = (
        int(metadata["w_xh_size"])
        + int(metadata["w_hh_size"])
        + int(metadata["b_h_size"])
    )
    hidden_size = params["b_h"].shape[0]

    assert mask.shape == (hidden_size, num_recurrent)


def test_snap1_mask_preserves_direct_w_xh_influences():
    params = _tiny_params()
    mask = np.asarray(build_snap1_recurrent_mask(params))

    input_size, hidden_size = params["w_xh"].shape
    for i in range(input_size):
        for j in range(hidden_size):
            col = i * hidden_size + j
            assert mask[j, col] == 1.0
            # Other hidden rows should be zero for this column.
            for row in range(hidden_size):
                if row != j:
                    assert mask[row, col] == 0.0


def test_snap1_mask_preserves_direct_w_hh_influences():
    params = _tiny_params()
    mask = np.asarray(build_snap1_recurrent_mask(params))

    hidden_size = params["b_h"].shape[0]
    w_xh_size = params["w_xh"].size
    offset = w_xh_size

    for k in range(hidden_size):
        for j in range(hidden_size):
            col = offset + k * hidden_size + j
            assert mask[j, col] == 1.0
            for row in range(hidden_size):
                if row != j:
                    assert mask[row, col] == 0.0


def test_snap1_mask_preserves_direct_b_h_influences():
    params = _tiny_params()
    mask = np.asarray(build_snap1_recurrent_mask(params))

    hidden_size = params["b_h"].shape[0]
    offset = params["w_xh"].size + params["w_hh"].size

    for j in range(hidden_size):
        col = offset + j
        assert mask[j, col] == 1.0
        for row in range(hidden_size):
            if row != j:
                assert mask[row, col] == 0.0


def test_snap1_gradients_have_correct_shapes():
    params = _tiny_params()
    key = jax.random.PRNGKey(1)
    batch = generate_copy_batch(key=key, batch_size=2, copy_length=3)

    _, grads = snap1_loss_and_gradients(
        params,
        batch.inputs,
        batch.targets,
        batch.target_mask,
    )

    assert set(grads.keys()) == set(params.keys())
    for key_name in params:
        assert grads[key_name].shape == params[key_name].shape
