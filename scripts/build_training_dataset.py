import argparse
from pathlib import Path

import pandas as pd

from churn_prediction.dataset import build_training_dataset
from churn_prediction.labels import CHURN_HORIZON_DAYS


DEFAULT_CUTOFFS = [
    pd.Timestamp("2024-02-01T00:00:00Z"),
    pd.Timestamp("2024-03-01T00:00:00Z"),
    pd.Timestamp("2024-04-01T00:00:00Z"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the churn model training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/synthetic_events.parquet"),
        help="Path to the synthetic event dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_dataset.parquet"),
        help="Path for the training dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    events = pd.read_parquet(args.input)

    training_dataset = build_training_dataset(
        events,
        cutoffs=DEFAULT_CUTOFFS,
        horizon_days=CHURN_HORIZON_DAYS,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_dataset.to_parquet(args.output, index=False)

    print(f"Training rows: {len(training_dataset)}")
    print(
        f"Unique customers: "
        f"{training_dataset['customer_id'].nunique()}"
    )
    print(
        f"Churn rate: "
        f"{training_dataset['churn'].mean() * 100:.1f}%"
    )
    print(f"Wrote output to {args.output.resolve()}")

    print("\nRows by cutoff:")
    print(
        training_dataset.groupby("cutoff")
        .agg(
            rows=("customer_id", "size"),
            churned=("churn", "sum"),
            churn_rate=("churn", "mean"),
        )
    )


if __name__ == "__main__":
    main()