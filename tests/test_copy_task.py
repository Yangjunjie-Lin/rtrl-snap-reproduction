import jax
import jax.numpy as jnp
import pytest

from rtrl_snap.tasks.copy_task import (
    BLANK_TOKEN,
    COPY_VOCAB_SIZE,
    DELIMITER_TOKEN,
    generate_copy_batch,
)


def test_generate_copy_batch_shapes() -> None:
    key = jax.random.PRNGKey(0)

    batch_size = 3
    copy_length = 5
    sequence_length = 2 * copy_length + 1

    batch = generate_copy_batch(
        key=key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    assert batch.input_tokens.shape == (batch_size, sequence_length)
    assert batch.inputs.shape == (
        batch_size,
        sequence_length,
        COPY_VOCAB_SIZE,
    )
    assert batch.targets.shape == (batch_size, sequence_length)
    assert batch.target_mask.shape == (batch_size, sequence_length)


def test_inputs_are_one_hot() -> None:
    key = jax.random.PRNGKey(1)

    batch = generate_copy_batch(
        key=key,
        batch_size=4,
        copy_length=6,
    )

    one_hot_sums = jnp.sum(batch.inputs, axis=-1)

    assert jnp.allclose(one_hot_sums, 1.0)


def test_delimiter_position_is_correct() -> None:
    key = jax.random.PRNGKey(2)

    batch_size = 4
    copy_length = 7

    batch = generate_copy_batch(
        key=key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    delimiter_column = batch.input_tokens[:, copy_length]

    assert jnp.all(delimiter_column == DELIMITER_TOKEN)


def test_input_blanks_after_delimiter() -> None:
    key = jax.random.PRNGKey(3)

    batch_size = 4
    copy_length = 7

    batch = generate_copy_batch(
        key=key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    blank_region = batch.input_tokens[:, copy_length + 1 :]

    assert jnp.all(blank_region == BLANK_TOKEN)


def test_targets_copy_original_bits() -> None:
    key = jax.random.PRNGKey(4)

    batch_size = 5
    copy_length = 8

    batch = generate_copy_batch(
        key=key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    original_bits = batch.input_tokens[:, :copy_length]
    copied_targets = batch.targets[:, copy_length + 1 :]

    assert jnp.array_equal(original_bits, copied_targets)


def test_target_mask_only_keeps_copy_region() -> None:
    key = jax.random.PRNGKey(5)

    batch_size = 2
    copy_length = 4

    batch = generate_copy_batch(
        key=key,
        batch_size=batch_size,
        copy_length=copy_length,
    )

    ignored_region = batch.target_mask[:, : copy_length + 1]
    copied_region = batch.target_mask[:, copy_length + 1 :]

    assert jnp.all(ignored_region == 0.0)
    assert jnp.all(copied_region == 1.0)


def test_invalid_batch_size_raises_error() -> None:
    key = jax.random.PRNGKey(6)

    with pytest.raises(ValueError):
        generate_copy_batch(
            key=key,
            batch_size=0,
            copy_length=4,
        )


def test_invalid_copy_length_raises_error() -> None:
    key = jax.random.PRNGKey(7)

    with pytest.raises(ValueError):
        generate_copy_batch(
            key=key,
            batch_size=4,
            copy_length=0,
        )