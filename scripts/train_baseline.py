import argparse
from pathlib import Path
import json
import pandas as pd

from churn_prediction.modeling import (
    MODEL_FEATURES,
    evaluate_model,
    evaluate_thresholds,
    split_by_customer,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate churn classification models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/training_dataset.parquet"),
        help="Path to the training dataset.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/churn_model.json"),
        help="Path for the selected XGBoost model artifact.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("artifacts/churn_model_metadata.json"),
        help="Path for model metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = pd.read_parquet(args.input)

    train, test = split_by_customer(
        dataset,
        test_size=0.2,
        seed=42,
    )

    models = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "Gradient Boosting": train_gradient_boosting,
        "XGBoost": train_xgboost,
    }

    print(f"Training rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(
        f"Training customers: {train['customer_id'].nunique()}"
    )
    print(
        f"Test customers: {test['customer_id'].nunique()}"
    )

    print("\nModel features:")
    for feature in MODEL_FEATURES:
        print(f"  {feature}")

    for model_name, train_model in models.items():
        model = train_model(train)

        metrics = evaluate_model(
            model,
            test,
        )

        threshold_results = evaluate_thresholds(
            model,
            test,
        )

        print(f"\n{model_name}")
        print("-" * len(model_name))

        for name, value in metrics.items():
            print(f"  {name}: {value:.3f}")

        print("\nThreshold tradeoff:")
        print(
            threshold_results.to_string(
                index=False,
                formatters={
                    "threshold": "{:.2f}".format,
                    "precision": "{:.3f}".format,
                    "recall": "{:.3f}".format,
                    "flagged_customers_pct": "{:.3f}".format,
                },
            )
        )
    selected_model = train_xgboost(train)

    args.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_model.save_model(args.model_output)
    metadata = {
    "model_type": "xgboost",
    "model_version": "xgb-v1",
    "feature_version": "rfm-v1",
    "features": MODEL_FEATURES,
    }

    args.metadata_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.metadata_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved selected XGBoost model to "
        f"{args.model_output.resolve()}"
    )


if __name__ == "__main__":
    main()