import pandas as pd

from churn_prediction.dataset import build_training_dataset


def test_build_training_dataset_combines_multiple_cutoffs() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-05T00:00:00Z"),
                "properties": {"duration_sec": 120},
            },
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-02-10T00:00:00Z"),
                "properties": {"duration_sec": 180},
            },
            {
                "customer_id": "cust_00001",
                "event_type": "purchase",
                "timestamp": pd.Timestamp("2024-03-10T00:00:00Z"),
                "properties": {"amount_usd": 9.99},
            },
            {
                "customer_id": "cust_00002",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-20T00:00:00Z"),
                "properties": {"duration_sec": 90},
            },
        ]
    )

    dataset = build_training_dataset(
        events,
        cutoffs=[
            pd.Timestamp("2024-02-01T00:00:00Z"),
            pd.Timestamp("2024-03-01T00:00:00Z"),
        ],
        horizon_days=60,
    )

    assert set(dataset["cutoff"].dt.date) == {
        pd.Timestamp("2024-02-01").date(),
        pd.Timestamp("2024-03-01").date(),
    }

    assert "churn" in dataset.columns
    assert "recency_days" in dataset.columns
    assert "sessions_30d" in dataset.columns


def test_training_dataset_excludes_customers_not_seen_before_cutoff() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-10T00:00:00Z"),
                "properties": {"duration_sec": 120},
            },
            {
                "customer_id": "cust_00002",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-03-10T00:00:00Z"),
                "properties": {"duration_sec": 120},
            },
        ]
    )

    dataset = build_training_dataset(
        events,
        cutoffs=[
            pd.Timestamp("2024-02-01T00:00:00Z"),
        ],
        horizon_days=60,
    )

    assert set(dataset["customer_id"]) == {"cust_00001"}