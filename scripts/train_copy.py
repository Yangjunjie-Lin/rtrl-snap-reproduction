from __future__ import annotations

import argparse
from pathlib import Path

import jax
import optax
from tqdm import trange

from rtrl_snap.algorithms.bptt import create_bptt_train_step
from rtrl_snap.algorithms.rtrl import create_rtrl_train_step
from rtrl_snap.models.vanilla_rnn import initialize_vanilla_rnn_params
from rtrl_snap.tasks.copy_task import COPY_VOCAB_SIZE, generate_copy_batch
from rtrl_snap.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Vanilla RNN on the Copy Task.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def _create_train_step(algorithm: str, optimizer):
    if algorithm == "bptt":
        return create_bptt_train_step(optimizer)
    if algorithm == "rtrl":
        return create_rtrl_train_step(optimizer)
    raise NotImplementedError(
        f"Unsupported algorithm '{algorithm}'. "
        "Supported algorithms: bptt, rtrl."
    )


def train_copy(config: dict) -> None:
    """
    Train Vanilla RNN on Copy Task with the configured algorithm.
    """

    algorithm = str(config["algorithm"]).lower()
    seed = int(config["seed"])

    task_config = config["task"]
    model_config = config["model"]
    training_config = config["training"]

    batch_size = int(task_config["batch_size"])
    copy_length = int(task_config["copy_length"])

    hidden_size = int(model_config["hidden_size"])

    num_steps = int(training_config["num_steps"])
    learning_rate = float(training_config["learning_rate"])
    log_every = int(training_config["log_every"])

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

    train_step = _create_train_step(algorithm, optimizer)

    progress_bar = trange(
        1,
        num_steps + 1,
        desc=f"Training {algorithm.upper()}",
    )

    for step in progress_bar:
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

        if step == 1 or step % log_every == 0:
            loss = float(metrics["loss"])
            accuracy = float(metrics["accuracy"])

            progress_bar.set_postfix(
                loss=f"{loss:.4f}",
                accuracy=f"{accuracy:.4f}",
            )

            print(
                f"step={step:05d} "
                f"loss={loss:.6f} "
                f"accuracy={accuracy:.6f}"
            )


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    train_copy(config)


if __name__ == "__main__":
    main()
