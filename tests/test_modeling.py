import pandas as pd

from churn_prediction.modeling import (
    evaluate_recency_baseline,
    split_by_customer,
    predict_churn_probability,
    train_xgboost,
)

def test_split_by_customer_keeps_customers_separate() -> None:
    rows = []

    for customer_number in range(20):
        customer_id = f"cust_{customer_number:05d}"

        for cutoff in [
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
        ]:
            rows.append(
                {
                    "customer_id": customer_id,
                    "cutoff": pd.Timestamp(cutoff),
                    "churn": customer_number % 2,
                }
            )

    dataset = pd.DataFrame(rows)

    train, test = split_by_customer(
        dataset,
        test_size=0.2,
        seed=42,
    )

    train_customers = set(train["customer_id"])
    test_customers = set(test["customer_id"])

    assert train_customers.isdisjoint(test_customers)
    assert len(train_customers) == 16
    assert len(test_customers) == 4



def test_predict_churn_probability_returns_valid_probabilities() -> None:
    dataset = pd.DataFrame(
        [
            {
                "customer_id": f"cust_{i:05d}",
                "recency_days": float(i * 5),
                "has_meaningful_history": 1,
                "sessions_30d": i % 4,
                "sessions_90d": i % 8,
                "purchase_count_90d": i % 3,
                "revenue_90d": float(i % 3) * 10.0,
                "churn": i % 2,
            }
            for i in range(20)
        ]
    )

    model = train_xgboost(dataset)

    probabilities = predict_churn_probability(
        model,
        dataset,
    )

    assert len(probabilities) == len(dataset)
    assert probabilities.between(0.0, 1.0).all()
    assert probabilities.name == "churn_probability"

def test_recency_baseline_uses_60_day_threshold() -> None:
    dataset = pd.DataFrame(
        [
            {
                "recency_days": 10.0,
                "churn": 0,
            },
            {
                "recency_days": 45.0,
                "churn": 0,
            },
            {
                "recency_days": 60.0,
                "churn": 1,
            },
            {
                "recency_days": 90.0,
                "churn": 1,
            },
        ]
    )

    metrics = evaluate_recency_baseline(
        dataset,
        threshold_days=60.0,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0