from __future__ import annotations

import jax
import optax

from rtrl_snap.algorithms.bptt import create_bptt_train_step
from rtrl_snap.algorithms.rtrl import create_rtrl_train_step
from rtrl_snap.algorithms.snap import create_snap1_train_step
from rtrl_snap.models.vanilla_rnn import initialize_vanilla_rnn_params
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch


def _run_smoke(
    algorithm: str,
    create_train_step,
    *,
    num_steps: int = 3,
    batch_size: int = 2,
    copy_length: int = 2,
    hidden_size: int = 4,
    learning_rate: float = 0.001,
    seed: int = 0,
) -> None:
    """
    Run a few training steps for one algorithm as a smoke test.
    """

    key = jax.random.PRNGKey(seed)
    key, model_key = jax.random.split(key)

    params = initialize_vanilla_rnn_params(
        key=model_key,
        input_size=COPY_VOCAB_SIZE,
        hidden_size=hidden_size,
        output_size=COPY_VOCAB_SIZE,
    )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    train_step = create_train_step(optimizer)

    for _ in range(num_steps):
        key, batch_key = jax.random.split(key)
        batch = generate_copy_batch(
            key=batch_key,
            batch_size=batch_size,
            copy_length=copy_length,
        )
        params, opt_state, metrics = train_step(
            params,
            opt_state,
            batch.inputs,
            batch.targets,
            batch.target_mask,
        )
        # Touch metrics so silent NaNs become visible if conversion fails.
        float(metrics["loss"])
        float(metrics["accuracy"])


def main() -> None:
    _run_smoke("bptt", create_bptt_train_step)
    print("BPTT smoke test passed")

    _run_smoke("rtrl", create_rtrl_train_step)
    print("RTRL smoke test passed")

    _run_smoke("snap1", create_snap1_train_step)
    print("SnAp-1 smoke test passed")


if __name__ == "__main__":
    main()
