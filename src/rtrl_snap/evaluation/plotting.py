from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_training_csv(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a Copy Task training CSV written by ``scripts/train_copy.py``.

    Expected columns:
        step, loss, accuracy, algorithm, copy_length, hidden_size,
        learning_rate, seed
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV log not found: {path}")

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    return {
        "step": np.asarray(data["step"], dtype=np.float64),
        "loss": np.asarray(data["loss"], dtype=np.float64),
        "accuracy": np.asarray(data["accuracy"], dtype=np.float64),
        "algorithm": str(data["algorithm"][0]),
    }


def plot_training_curves(
    csv_paths: list[str | Path],
    output_path: str | Path,
    metric: str = "loss",
) -> Path:
    """
    Plot training curves from one or more CSV logs and save a figure.

    Args:
        csv_paths:
            Paths to CSV logs.

        output_path:
            Where to save the figure (PNG).

        metric:
            Either ``loss`` or ``accuracy``.

    Returns:
        Path to the saved figure.
    """

    if metric not in {"loss", "accuracy"}:
        raise ValueError("metric must be 'loss' or 'accuracy'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for csv_path in csv_paths:
        log = load_training_csv(csv_path)
        ax.plot(
            log["step"],
            log[metric],
            label=log["algorithm"],
            linewidth=2,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Copy Task training {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
