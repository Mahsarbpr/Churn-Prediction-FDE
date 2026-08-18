from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

from churn_prediction.explainability import (
    explain_prediction,
    global_feature_importance,
)
from churn_prediction.modeling import (
    MODEL_FEATURES,
    predict_churn_probability,
    split_by_customer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate explainability outputs for the churn model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "artifacts/training_dataset.parquet"
        ),
        help="Path to the training dataset.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "artifacts/churn_model.json"
        ),
        help="Path to the trained XGBoost model.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/explainability"
        ),
        help="Directory for explainability artifacts.",
    )

    return parser.parse_args()


def save_global_importance_plot(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = importance.sort_values(
        "importance_share",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.barh(
        plot_data["feature"],
        plot_data["importance_share"],
    )

    axis.set_xlabel(
        "Share of total feature gain"
    )
    axis.set_ylabel("Feature")
    axis.set_title(
        "XGBoost Global Feature Importance"
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def save_local_explanation_plot(
    explanation: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = explanation.sort_values(
        "contribution",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.barh(
        plot_data["feature"],
        plot_data["contribution"],
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel(
        "Contribution to raw churn score"
    )
    axis.set_ylabel("Feature")
    axis.set_title(
        "Example Customer Prediction Explanation"
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = pd.read_parquet(
        args.input
    )

    _, test = split_by_customer(
        dataset,
        test_size=0.2,
        seed=42,
    )

    model = XGBClassifier()
    model.load_model(args.model)

    importance = global_feature_importance(
        model
    )

    importance_csv = (
        args.output_dir
        / "global_feature_importance.csv"
    )

    importance_png = (
        args.output_dir
        / "global_feature_importance.png"
    )

    importance.to_csv(
        importance_csv,
        index=False,
    )

    save_global_importance_plot(
        importance,
        importance_png,
    )

    probabilities = predict_churn_probability(
        model,
        test,
    )

    example_index = probabilities.idxmax()

    example = test.loc[
        [example_index]
    ].copy()

    example_probability = float(
        probabilities.loc[example_index]
    )

    explanation = explain_prediction(
        model,
        example,
    )

    explanation_csv = (
        args.output_dir
        / "example_prediction_explanation.csv"
    )

    explanation_png = (
        args.output_dir
        / "example_prediction_explanation.png"
    )

    explanation.to_csv(
        explanation_csv,
        index=False,
    )

    save_local_explanation_plot(
        explanation,
        explanation_png,
    )

    print("\nGlobal feature importance:")
    print(
        importance.to_string(
            index=False,
            formatters={
                "total_gain": "{:.3f}".format,
                "importance_share": "{:.3f}".format,
            },
        )
    )

    print(
        "\nExample held-out prediction:"
    )

    print(
        f"  customer_id: "
        f"{example.iloc[0]['customer_id']}"
    )

    if "cutoff" in example.columns:
        print(
            f"  cutoff: "
            f"{example.iloc[0]['cutoff']}"
        )

    print(
        f"  actual churn: "
        f"{int(example.iloc[0]['churn'])}"
    )

    print(
        f"  predicted probability: "
        f"{example_probability:.3f}"
    )

    print(
        "\nFeature contributions:"
    )

    print(
        explanation[
            [
                "feature",
                "value",
                "contribution",
                "direction",
            ]
        ].to_string(
            index=False,
            formatters={
                "value": "{:.3f}".format,
                "contribution": "{:.3f}".format,
            },
        )
    )

    print(
        f"\nSaved explainability outputs to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()