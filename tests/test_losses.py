import jax
import jax.numpy as jnp
import pytest

from rtrl_snap.models.vanilla_rnn import (
    forward_sequence,
    initialize_vanilla_rnn_params,
)
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch
from rtrl_snap.training.losses import (
    sequence_cross_entropy_loss,
    sequence_token_accuracy,
)


def test_sequence_cross_entropy_loss_is_scalar() -> None:
    key = jax.random.PRNGKey(0)
    data_key, model_key = jax.random.split(key)

    batch_size = 3
    copy_length = 5
    hidden_size = 16

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

    loss = sequence_cross_entropy_loss(
        logits=outputs["logits"],
        targets=batch.targets,
        target_mask=batch.target_mask,
    )

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_sequence_token_accuracy_is_scalar() -> None:
    key = jax.random.PRNGKey(1)
    data_key, model_key = jax.random.split(key)

    batch_size = 4
    copy_length = 6
    hidden_size = 12

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

    accuracy = sequence_token_accuracy(
        logits=outputs["logits"],
        targets=batch.targets,
        target_mask=batch.target_mask,
    )

    assert accuracy.shape == ()
    assert bool(jnp.isfinite(accuracy))
    assert 0.0 <= float(accuracy) <= 1.0


def test_perfect_logits_have_low_loss_and_full_accuracy() -> None:
    targets = jnp.array(
        [
            [3, 3, 0, 1],
            [3, 3, 1, 0],
        ],
        dtype=jnp.int32,
    )

    target_mask = jnp.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    batch_size, sequence_length = targets.shape
    vocab_size = 4

    logits = jnp.full(
        shape=(batch_size, sequence_length, vocab_size),
        fill_value=-10.0,
        dtype=jnp.float32,
    )

    logits = logits.at[
        jnp.arange(batch_size)[:, None],
        jnp.arange(sequence_length)[None, :],
        targets,
    ].set(10.0)

    loss = sequence_cross_entropy_loss(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    accuracy = sequence_token_accuracy(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    assert float(loss) < 1e-3
    assert float(accuracy) == 1.0


def test_target_mask_ignores_incorrect_unmasked_positions() -> None:
    targets = jnp.array(
        [
            [0, 1, 1, 0],
        ],
        dtype=jnp.int32,
    )

    target_mask = jnp.array(
        [
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    logits = jnp.full(
        shape=(1, 4, 4),
        fill_value=-10.0,
        dtype=jnp.float32,
    )

    wrong_predictions = jnp.array(
        [
            [3, 3, 1, 0],
        ],
        dtype=jnp.int32,
    )

    logits = logits.at[
        jnp.arange(1)[:, None],
        jnp.arange(4)[None, :],
        wrong_predictions,
    ].set(10.0)

    loss = sequence_cross_entropy_loss(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    accuracy = sequence_token_accuracy(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    assert float(loss) < 1e-3
    assert float(accuracy) == 1.0


def test_invalid_logits_shape_raises_error() -> None:
    logits = jnp.zeros((2, 4), dtype=jnp.float32)
    targets = jnp.zeros((2, 4), dtype=jnp.int32)
    target_mask = jnp.ones((2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError):
        sequence_cross_entropy_loss(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
        )


def test_invalid_targets_shape_raises_error() -> None:
    logits = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    targets = jnp.zeros((2, 4, 1), dtype=jnp.int32)
    target_mask = jnp.ones((2, 4), dtype=jnp.float32)

    with pytest.raises(ValueError):
        sequence_cross_entropy_loss(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
        )


def test_invalid_target_mask_shape_raises_error() -> None:
    logits = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    targets = jnp.zeros((2, 4), dtype=jnp.int32)
    target_mask = jnp.ones((2, 4, 1), dtype=jnp.float32)

    with pytest.raises(ValueError):
        sequence_cross_entropy_loss(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
        )


def test_mismatched_batch_or_sequence_dimensions_raise_error() -> None:
    logits = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    targets = jnp.zeros((2, 5), dtype=jnp.int32)
    target_mask = jnp.ones((2, 5), dtype=jnp.float32)

    with pytest.raises(ValueError):
        sequence_cross_entropy_loss(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
        )