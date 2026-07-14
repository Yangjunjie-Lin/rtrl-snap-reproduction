from __future__ import annotations

import argparse
from pathlib import Path

from rtrl_snap.evaluation.plotting import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Copy Task training curves from CSV logs.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV log paths.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="loss",
        choices=["loss", "accuracy"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/copy_training.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_training_curves(
        csv_paths=args.csv,
        output_path=Path(args.output),
        metric=args.metric,
    )
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
