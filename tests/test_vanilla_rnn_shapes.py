import jax
import jax.numpy as jnp
import pytest

from rtrl_snap.models.vanilla_rnn import (
    forward_sequence,
    initialize_vanilla_rnn_params,
    rnn_step,
)
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch


def test_initialize_vanilla_rnn_params_shapes() -> None:
    key = jax.random.PRNGKey(0)

    input_size = 4
    hidden_size = 16
    output_size = 4

    params = initialize_vanilla_rnn_params(
        key=key,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    assert params["w_xh"].shape == (input_size, hidden_size)
    assert params["w_hh"].shape == (hidden_size, hidden_size)
    assert params["b_h"].shape == (hidden_size,)
    assert params["w_hy"].shape == (hidden_size, output_size)
    assert params["b_y"].shape == (output_size,)


def test_rnn_step_shape() -> None:
    key = jax.random.PRNGKey(1)

    batch_size = 3
    input_size = 4
    hidden_size = 12
    output_size = 4

    params = initialize_vanilla_rnn_params(
        key=key,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )

    x_t = jnp.ones(
        shape=(batch_size, input_size),
        dtype=jnp.float32,
    )

    h_prev = jnp.zeros(
        shape=(batch_size, hidden_size),
        dtype=jnp.float32,
    )

    h_next = rnn_step(
        params=params,
        x_t=x_t,
        h_prev=h_prev,
    )

    assert h_next.shape == (batch_size, hidden_size)
    assert bool(jnp.all(jnp.isfinite(h_next)))


def test_forward_sequence_shapes() -> None:
    key = jax.random.PRNGKey(2)
    data_key, model_key = jax.random.split(key)

    batch_size = 5
    copy_length = 6
    sequence_length = 2 * copy_length + 1
    hidden_size = 20

    batch = generate_copy_batch(
        key=data_key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    params = initialize_vanilla_rnn_params(
        key=model_key,
        input_size=COPY_VOCAB_SIZE,
        hidden_size=hidden_size,
        output_size=COPY_VOCAB_SIZE,
    )

    outputs = forward_sequence(
        params=params,
        inputs=batch.inputs,
    )

    assert outputs["hidden_states"].shape == (
        batch_size,
        sequence_length,
        hidden_size,
    )

    assert outputs["logits"].shape == (
        batch_size,
        sequence_length,
        COPY_VOCAB_SIZE,
    )

    assert outputs["final_hidden"].shape == (
        batch_size,
        hidden_size,
    )

    assert bool(jnp.all(jnp.isfinite(outputs["hidden_states"])))
    assert bool(jnp.all(jnp.isfinite(outputs["logits"])))
    assert bool(jnp.all(jnp.isfinite(outputs["final_hidden"])))


def test_forward_sequence_accepts_custom_initial_hidden() -> None:
    key = jax.random.PRNGKey(3)
    data_key, model_key = jax.random.split(key)

    batch_size = 4
    copy_length = 5
    hidden_size = 10

    batch = generate_copy_batch(
        key=data_key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    params = initialize_vanilla_rnn_params(
        key=model_key,
        input_size=COPY_VOCAB_SIZE,
        hidden_size=hidden_size,
        output_size=COPY_VOCAB_SIZE,
    )

    initial_hidden = jnp.ones(
        shape=(batch_size, hidden_size),
        dtype=jnp.float32,
    )

    outputs = forward_sequence(
        params=params,
        inputs=batch.inputs,
        initial_hidden=initial_hidden,
    )

    assert outputs["final_hidden"].shape == (
        batch_size,
        hidden_size,
    )


def test_invalid_input_size_raises_error() -> None:
    key = jax.random.PRNGKey(4)

    with pytest.raises(ValueError):
        initialize_vanilla_rnn_params(
            key=key,
            input_size=0,
            hidden_size=8,
            output_size=4,
        )


def test_invalid_hidden_size_raises_error() -> None:
    key = jax.random.PRNGKey(5)

    with pytest.raises(ValueError):
        initialize_vanilla_rnn_params(
            key=key,
            input_size=4,
            hidden_size=0,
            output_size=4,
        )


def test_invalid_output_size_raises_error() -> None:
    key = jax.random.PRNGKey(6)

    with pytest.raises(ValueError):
        initialize_vanilla_rnn_params(
            key=key,
            input_size=4,
            hidden_size=8,
            output_size=0,
        )