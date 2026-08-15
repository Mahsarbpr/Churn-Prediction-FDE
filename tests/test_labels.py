import pandas as pd

from churn_prediction.labels import build_churn_labels


def test_build_churn_labels_marks_future_activity_as_active() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-15T00:00:00Z"),
            },
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-02-15T00:00:00Z"),
            },
            {
                "customer_id": "cust_00002",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-20T00:00:00Z"),
            },
        ]
    )

    labels = build_churn_labels(
        events,
        cutoff=pd.Timestamp("2024-02-01T00:00:00Z"),
        horizon_days=60,
    )

    result = labels.set_index("customer_id")["churn"].to_dict()

    assert result["cust_00001"] == 0
    assert result["cust_00002"] == 1


def test_push_sent_does_not_count_as_customer_activity() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-15T00:00:00Z"),
            },
            {
                "customer_id": "cust_00001",
                "event_type": "push_sent",
                "timestamp": pd.Timestamp("2024-02-10T00:00:00Z"),
            },
        ]
    )

    labels = build_churn_labels(
        events,
        cutoff=pd.Timestamp("2024-02-01T00:00:00Z"),
        horizon_days=60,
    )

    assert labels.loc[0, "churn"] == 1


def test_customer_first_seen_after_cutoff_is_excluded() -> None:
    events = pd.DataFrame(
        [
            {
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-15T00:00:00Z"),
            },
            {
                "customer_id": "cust_00002",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-02-10T00:00:00Z"),
            },
        ]
    )

    labels = build_churn_labels(
        events,
        cutoff=pd.Timestamp("2024-02-01T00:00:00Z"),
        horizon_days=60,
    )

    assert set(labels["customer_id"]) == {"cust_00001"}