from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from churn_prediction.modeling import (
    MODEL_FEATURES,
    predict_churn_probability,
    split_by_customer,
)


DATASET_PATH = Path("artifacts/training_dataset.parquet")
MODEL_PATH = Path("artifacts/churn_model.json")
OUTPUT_PATH = Path("artifacts/bias_fairness_note.md")

THRESHOLD = 0.20

PROTECTED_ATTRIBUTES = {
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "disability",
}


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)

    _, test = split_by_customer(
        dataset,
        test_size=0.2,
        seed=42,
    )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    test = test.copy()

    test["churn_probability"] = predict_churn_probability(
        model,
        test,
    )

    test["predicted_churn"] = (
        test["churn_probability"] >= THRESHOLD
    ).astype(int)

    # 1. Check whether protected attributes are available.
    protected_found = sorted(
        PROTECTED_ATTRIBUTES.intersection(dataset.columns)
    )

    # 2. Check representation of cold-start customers.
    history_counts = (
        dataset["has_meaningful_history"]
        .value_counts()
        .sort_index()
    )

    history_yes = int(history_counts.get(1, 0))
    history_no = int(history_counts.get(0, 0))

    # 3. Review errors across the complete held-out evaluation set.
    tn, fp, fn, tp = confusion_matrix(
        test["churn"],
        test["predicted_churn"],
        labels=[0, 1],
    ).ravel()

    # 4. Check campaign-selection rates by recent spend.
    test["spend_group"] = test["revenue_90d"].gt(0).map(
        {
            True: "recent_spend",
            False: "no_recent_spend",
        }
    )

    selection = (
        test.groupby("spend_group")["predicted_churn"]
        .agg(["count", "mean"])
        .rename(
            columns={
                "count": "rows",
                "mean": "selected_rate",
            }
        )
    )

    print("\nBias / fairness check")
    print("---------------------")

    print(f"\nModel features: {MODEL_FEATURES}")

    print(
        "\nProtected attributes found:",
        protected_found if protected_found else "none",
    )

    print("\nHistory representation:")
    print(f"  meaningful history: {history_yes}")
    print(f"  limited/no history: {history_no}")

    print(
        f"\nHeld-out predictions reviewed: {len(test)}"
    )

    print(f"Decision threshold: {THRESHOLD:.2f}")

    print("\nPrediction outcomes:")
    print(f"  true positives:  {tp}")
    print(f"  true negatives:  {tn}")
    print(f"  false positives: {fp}")
    print(f"  false negatives: {fn}")

    print("\nCampaign selection by recent spend:")
    print(selection)

    note = f"""# Bias and Fairness Check

The supplied dataset does not contain protected demographic attributes such as age, gender, race, ethnicity, or disability. Demographic fairness therefore cannot be directly measured from this dataset, and I do not claim that the model is unbiased across protected groups.

## What I checked

- Reviewed the model inputs for direct protected attributes.
- Reviewed representation of customers with limited historical activity.
- Reviewed all {len(test)} held-out prediction rows for false positives and false negatives at a {THRESHOLD:.2f} campaign threshold.
- Reviewed campaign selection rates for customers with and without recent spend because revenue is used as a model signal and could affect access to retention incentives.

## What I found

- No protected demographic attributes are used by the model.
- Customers with meaningful history: {history_yes}
- Customers with limited/no meaningful history: {history_no}
- True positives: {tp}
- True negatives: {tn}
- False positives: {fp}
- False negatives: {fn}
- Campaign selection rate for customers with recent spend: {selection.loc["recent_spend", "selected_rate"]:.1%}
- Campaign selection rate for customers without recent spend: {selection.loc["no_recent_spend", "selected_rate"]:.1%}

Customers with little prior history are severely underrepresented, so model performance for cold-start customers cannot be established confidently from this dataset.

The difference in selection rates is not evidence of demographic unfairness because recent revenue is itself a behavioral model input. However, it should be reviewed with product stakeholders if campaign selection determines access to discounts or other valuable retention benefits.

## Recommendation

Before production use, I would evaluate false-positive rates, false-negative rates, recall, and calibration across appropriate protected groups where those attributes can legally and appropriately be used for model auditing. I would also collect more cold-start examples and monitor campaign-selection outcomes for unintended disparities.
"""

    OUTPUT_PATH.write_text(
        note,
        encoding="utf-8",
    )

    print(
        f"\nSaved note to {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()