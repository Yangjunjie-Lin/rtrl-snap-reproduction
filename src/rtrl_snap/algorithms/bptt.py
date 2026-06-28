from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from rtrl_snap.models.vanilla_rnn import forward_sequence
from rtrl_snap.training.losses import (
    sequence_cross_entropy_loss,
    sequence_token_accuracy,
)


def bptt_loss(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute BPTT sequence loss for a full input sequence.

    Args:
        params:
            Vanilla RNN parameters.

        inputs:
            Input sequence with shape [batch_size, sequence_length, input_size].

        targets:
            Target tokens with shape [batch_size, sequence_length].

        target_mask:
            Loss mask with shape [batch_size, sequence_length].

    Returns:
        Scalar masked sequence cross-entropy loss.
    """

    outputs = forward_sequence(
        params=params,
        inputs=inputs,
    )

    return sequence_cross_entropy_loss(
        logits=outputs["logits"],
        targets=targets,
        target_mask=target_mask,
    )


def bptt_metrics(
    params: dict[str, jnp.ndarray],
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """
    Compute BPTT evaluation metrics without updating parameters.

    Returns:
        Dictionary containing loss and accuracy.
    """

    outputs = forward_sequence(
        params=params,
        inputs=inputs,
    )

    loss = sequence_cross_entropy_loss(
        logits=outputs["logits"],
        targets=targets,
        target_mask=target_mask,
    )

    accuracy = sequence_token_accuracy(
        logits=outputs["logits"],
        targets=targets,
        target_mask=target_mask,
    )

    return {
        "loss": loss,
        "accuracy": accuracy,
    }


def create_bptt_train_step(
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
    Create one JIT-compiled BPTT training step.

    The returned function performs:

        forward pass
        loss computation
        gradient computation
        optimizer update
        metric computation

    Args:
        optimizer:
            Optax optimizer.

    Returns:
        JIT-compiled train_step function.
    """

    @jax.jit
    def train_step(
        params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        target_mask: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], optax.OptState, dict[str, jnp.ndarray]]:
        def loss_with_aux(
            current_params: dict[str, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            outputs = forward_sequence(
                params=current_params,
                inputs=inputs,
            )

            loss = sequence_cross_entropy_loss(
                logits=outputs["logits"],
                targets=targets,
                target_mask=target_mask,
            )

            accuracy = sequence_token_accuracy(
                logits=outputs["logits"],
                targets=targets,
                target_mask=target_mask,
            )

            return loss, accuracy

        (loss, accuracy), grads = jax.value_and_grad(
            loss_with_aux,
            has_aux=True,
        )(params)

        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }

        return params, opt_state, metrics

    return train_step