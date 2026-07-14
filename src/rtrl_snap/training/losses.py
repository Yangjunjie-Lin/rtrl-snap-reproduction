from __future__ import annotations

import jax.numpy as jnp
from jax.nn import log_softmax


def validate_sequence_loss_shapes(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> None:
    """
    Validate shapes for sequence-level cross-entropy loss.

    Args:
        logits:
            [batch_size, sequence_length, vocab_size]

        targets:
            [batch_size, sequence_length]

        target_mask:
            [batch_size, sequence_length]
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch_size, sequence_length, vocab_size].")

    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch_size, sequence_length].")

    if target_mask.ndim != 2:
        raise ValueError("target_mask must have shape [batch_size, sequence_length].")

    if logits.shape[:2] != targets.shape:
        raise ValueError("logits and targets must share batch and sequence dimensions.")

    if targets.shape != target_mask.shape:
        raise ValueError("targets and target_mask must have the same shape.")


def sequence_cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked sequence cross-entropy loss.

    Only positions where target_mask == 1 are included in the loss.

    Args:
        logits:
            Model logits with shape [batch_size, sequence_length, vocab_size].

        targets:
            Integer target tokens with shape [batch_size, sequence_length].

        target_mask:
            Float mask with shape [batch_size, sequence_length].
            Positions with 1.0 are included in the loss.
            Positions with 0.0 are ignored.

    Returns:
        Scalar masked cross-entropy loss.
    """

    validate_sequence_loss_shapes(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    log_probs = log_softmax(logits, axis=-1)

    target_log_probs = jnp.take_along_axis(
        log_probs,
        targets[..., None],
        axis=-1,
    ).squeeze(axis=-1)

    negative_log_likelihood = -target_log_probs

    masked_nll = negative_log_likelihood * target_mask

    normalizer = jnp.maximum(jnp.sum(target_mask), 1.0)

    return jnp.sum(masked_nll) / normalizer


def sequence_token_accuracy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute masked token accuracy.

    Only positions where target_mask == 1 are included.

    Args:
        logits:
            Model logits with shape [batch_size, sequence_length, vocab_size].

        targets:
            Integer target tokens with shape [batch_size, sequence_length].

        target_mask:
            Float mask with shape [batch_size, sequence_length].

    Returns:
        Scalar masked token accuracy.
    """

    validate_sequence_loss_shapes(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
    )

    predictions = jnp.argmax(logits, axis=-1)

    correct = (predictions == targets).astype(jnp.float32)
    masked_correct = correct * target_mask

    normalizer = jnp.maximum(jnp.sum(target_mask), 1.0)

    return jnp.sum(masked_correct) / normalizer