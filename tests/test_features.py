import pandas as pd

from churn_prediction.features import build_rfm_features


AS_OF = pd.Timestamp("2024-06-01T12:00:00Z")


def make_event(
    event_id: str,
    customer_id: str,
    event_type: str,
    timestamp: str,
    properties: dict,
) -> dict:
    return {
        "event_id": event_id,
        "customer_id": customer_id,
        "event_type": event_type,
        "timestamp": pd.Timestamp(timestamp),
        "properties": properties,
    }


def test_build_rfm_features() -> None:
    events = pd.DataFrame(
        [
            make_event(
                "e1",
                "customer_a",
                "session",
                "2024-05-27T12:00:00Z",
                {"duration_sec": 100},
            ),
            make_event(
                "e2",
                "customer_a",
                "session",
                "2024-05-20T12:00:00Z",
                {"duration_sec": 120},
            ),
            make_event(
                "e3",
                "customer_a",
                "purchase",
                "2024-05-15T12:00:00Z",
                {"amount_usd": 25.50},
            ),
            make_event(
                "e4",
                "customer_b",
                "session",
                "2024-03-10T12:00:00Z",
                {"duration_sec": 90},
            ),
        ]
    )

    features = build_rfm_features(
        events,
        as_of=AS_OF,
    ).set_index("customer_id")

    assert len(features) == 2

    assert features.loc["customer_a", "recency_days"] == 5
    assert features.loc["customer_a", "sessions_30d"] == 2
    assert features.loc["customer_a", "sessions_90d"] == 2
    assert features.loc["customer_a", "purchase_count_90d"] == 1
    assert features.loc["customer_a", "revenue_90d"] == 25.50

    assert features.loc["customer_b", "sessions_30d"] == 0
    assert features.loc["customer_b", "sessions_90d"] == 1
    assert features.loc["customer_b", "purchase_count_90d"] == 0
    assert features.loc["customer_b", "revenue_90d"] == 0.0


def test_future_events_are_not_used() -> None:
    events = pd.DataFrame(
        [
            make_event(
                "e1",
                "customer_a",
                "session",
                "2024-05-20T12:00:00Z",
                {"duration_sec": 100},
            ),
            make_event(
                "e2",
                "customer_a",
                "session",
                "2024-06-10T12:00:00Z",
                {"duration_sec": 100},
            ),
        ]
    )

    features = build_rfm_features(
        events,
        as_of=AS_OF,
    ).set_index("customer_id")

    assert features.loc["customer_a", "recency_days"] == 12
    assert features.loc["customer_a", "sessions_30d"] == 1