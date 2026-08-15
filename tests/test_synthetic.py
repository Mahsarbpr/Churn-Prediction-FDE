import pandas as pd
from churn_prediction.synthetic import generate_synthetic_events


def test_generate_synthetic_events_is_reproducible() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "evt_000001",
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "properties": {"duration_sec": 120},
            },
            {
                "event_id": "evt_000002",
                "customer_id": "cust_00001",
                "event_type": "purchase",
                "timestamp": pd.Timestamp("2024-01-10T00:00:00Z"),
                "properties": {"amount_usd": 9.99},
            },
            {
                "event_id": "evt_000003",
                "customer_id": "cust_00002",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-05T00:00:00Z"),
                "properties": {"duration_sec": 240},
            },
            {
                "event_id": "evt_000004",
                "customer_id": "cust_00002",
                "event_type": "purchase",
                "timestamp": pd.Timestamp("2024-01-20T00:00:00Z"),
                "properties": {"amount_usd": 19.99},
            },
        ]
    )

    first = generate_synthetic_events(
        events,
        customer_count=5,
        seed=42,
    )

    second = generate_synthetic_events(
        events,
        customer_count=5,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)


def test_generate_synthetic_events_uses_expected_id_format() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "evt_000001",
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-01T00:00:00Z"),
                "properties": {"duration_sec": 120},
            },
            {
                "event_id": "evt_000002",
                "customer_id": "cust_00001",
                "event_type": "session",
                "timestamp": pd.Timestamp("2024-01-10T00:00:00Z"),
                "properties": {"duration_sec": 180},
            },
        ]
    )

    synthetic = generate_synthetic_events(
        events,
        customer_count=3,
        seed=42,
    )

    assert synthetic["customer_id"].str.match(
        r"syn_cust_\d{5}"
    ).all()

    assert synthetic["event_id"].str.match(
        r"syn_evt_\d{8}"
    ).all()