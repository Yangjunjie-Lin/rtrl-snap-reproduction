from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


ZERO_TOKEN = 0
ONE_TOKEN = 1
DELIMITER_TOKEN = 2
BLANK_TOKEN = 3

COPY_VOCAB_SIZE = 4


@dataclass(frozen=True)
class CopyBatch:
    """
    A batch for the synthetic Copy Task.

    input_tokens:
        Integer token sequence with shape [batch_size, sequence_length].

    inputs:
        One-hot encoded input sequence with shape
        [batch_size, sequence_length, vocab_size].

    targets:
        Integer target sequence with shape [batch_size, sequence_length].

    target_mask:
        Float mask with shape [batch_size, sequence_length].
        Positions with value 1.0 are used for loss computation.
        Positions with value 0.0 are ignored.
    """

    input_tokens: jnp.ndarray
    inputs: jnp.ndarray
    targets: jnp.ndarray
    target_mask: jnp.ndarray


def validate_copy_task_args(batch_size: int, copy_length: int) -> None:
    """
    Validate Copy Task generation arguments.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if copy_length <= 0:
        raise ValueError("copy_length must be greater than 0.")


def tokens_to_one_hot(
    tokens: jnp.ndarray,
    vocab_size: int = COPY_VOCAB_SIZE,
) -> jnp.ndarray:
    """
    Convert integer tokens to one-hot vectors.

    Args:
        tokens:
            Integer token array with shape [batch_size, sequence_length].

        vocab_size:
            Number of tokens in the vocabulary.

    Returns:
        One-hot encoded array with shape
        [batch_size, sequence_length, vocab_size].
    """

    return jax.nn.one_hot(tokens, num_classes=vocab_size, dtype=jnp.float32)


def generate_copy_batch(
    key: jax.Array,
    batch_size: int,
    copy_length: int,
) -> CopyBatch:
    """
    Generate a batch for the Copy Task.

    The generated sequence has the following structure:

        input:
            bits, delimiter, blanks

        target:
            blanks, copied bits

    Example with copy_length = 4:

        input:
            1 0 1 1 <delimiter> <blank> <blank> <blank> <blank>

        target:
            <blank> <blank> <blank> <blank> <blank> 1 0 1 1

    The total sequence length is:

        2 * copy_length + 1

    Args:
        key:
            JAX random key.

        batch_size:
            Number of sequences in the batch.

        copy_length:
            Number of binary tokens to remember and copy.

    Returns:
        CopyBatch containing integer tokens, one-hot inputs, targets,
        and a target mask.
    """

    validate_copy_task_args(batch_size=batch_size, copy_length=copy_length)

    bits = jax.random.bernoulli(
        key,
        p=0.5,
        shape=(batch_size, copy_length),
    ).astype(jnp.int32)

    delimiter = jnp.full(
        shape=(batch_size, 1),
        fill_value=DELIMITER_TOKEN,
        dtype=jnp.int32,
    )

    input_blanks = jnp.full(
        shape=(batch_size, copy_length),
        fill_value=BLANK_TOKEN,
        dtype=jnp.int32,
    )

    input_tokens = jnp.concatenate(
        [bits, delimiter, input_blanks],
        axis=1,
    )

    target_blanks = jnp.full(
        shape=(batch_size, copy_length + 1),
        fill_value=BLANK_TOKEN,
        dtype=jnp.int32,
    )

    targets = jnp.concatenate(
        [target_blanks, bits],
        axis=1,
    )

    ignored_positions = jnp.zeros(
        shape=(batch_size, copy_length + 1),
        dtype=jnp.float32,
    )

    copied_positions = jnp.ones(
        shape=(batch_size, copy_length),
        dtype=jnp.float32,
    )

    target_mask = jnp.concatenate(
        [ignored_positions, copied_positions],
        axis=1,
    )

    inputs = tokens_to_one_hot(input_tokens)

    return CopyBatch(
        input_tokens=input_tokens,
        inputs=inputs,
        targets=targets,
        target_mask=target_mask,
    )