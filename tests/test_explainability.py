import pandas as pd

from churn_prediction.explainability import (
    explain_prediction,
    global_feature_importance,
)
from churn_prediction.modeling import (
    MODEL_FEATURES,
    train_xgboost,
)


def build_test_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": f"cust_{i:05d}",
                "recency_days": float(i * 3),
                "has_meaningful_history": 1,
                "sessions_30d": i % 4,
                "sessions_90d": i % 8,
                "purchase_count_90d": i % 3,
                "revenue_90d": float(i % 3) * 10.0,
                "churn": i % 2,
            }
            for i in range(40)
        ]
    )


def test_global_feature_importance_contains_all_features() -> None:
    dataset = build_test_dataset()

    model = train_xgboost(
        dataset
    )

    importance = global_feature_importance(
        model
    )

    assert set(
        importance["feature"]
    ) == set(MODEL_FEATURES)

    assert (
        importance["importance_share"] >= 0.0
    ).all()


def test_explain_prediction_returns_one_row_per_feature() -> None:
    dataset = build_test_dataset()

    model = train_xgboost(
        dataset
    )

    explanation = explain_prediction(
        model,
        dataset.iloc[[0]],
    )

    assert len(explanation) == len(
        MODEL_FEATURES
    )

    assert set(
        explanation["feature"]
    ) == set(MODEL_FEATURES)

    assert explanation[
        "contribution"
    ].notna().all()