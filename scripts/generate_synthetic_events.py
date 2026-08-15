import argparse
from pathlib import Path
from churn_prediction.features import load_events
from churn_prediction.synthetic import generate_synthetic_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic customer event histories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/events.json"),
        help="Path to the source event JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthetic_events.parquet"),
        help="Path for the generated Parquet file.",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=1000,
        help="Number of synthetic customers to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    events = load_events(str(args.input))

    synthetic_events = generate_synthetic_events(
        events,
        customer_count=args.customers,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic_events.to_parquet(args.output, index=False)

    customer_count = synthetic_events["customer_id"].nunique()

    print(f"Generated {len(synthetic_events)} synthetic events")
    print(f"Generated {customer_count} synthetic customers")
    print(f"Wrote output to {args.output.resolve()}")


if __name__ == "__main__":
    main()