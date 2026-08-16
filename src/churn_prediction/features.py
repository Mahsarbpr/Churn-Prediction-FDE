from __future__ import annotations

import pandas as pd


DEFAULT_AS_OF = pd.Timestamp("2024-06-01T12:00:00Z")
MEANINGFUL_ACTIVITY = {
    "session",
    "purchase",
    "push_open",
    "in_app_event",
    "campaign_click",
    "support_ticket",
}

def load_events(path: str) -> pd.DataFrame:
    """Load and normalize raw event JSON."""

    events = pd.read_json(path)

    required_columns = {
        "event_id",
        "customer_id",
        "event_type",
        "timestamp",
        "properties",
    }

    missing = required_columns - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    return events


def build_rfm_features(
    events: pd.DataFrame,
    as_of: pd.Timestamp = DEFAULT_AS_OF,
) -> pd.DataFrame:
    """
    Build one RFM feature row per customer.
    Recency: Days since the customer's last session before as_of.
    Frequency: Session counts during the previous 30 and 90 days.
    Monetary: Purchase count and total purchase revenue during the previous 90 days.
    """

    events = events.copy()

    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    # Only information available at scoring time may be used.
    events = events[events["timestamp"] < as_of]

    customer_ids = pd.Index(
        sorted(events["customer_id"].unique()),
        name="customer_id",
    )

    features = pd.DataFrame(index=customer_ids)

    # Recency
    meaningful_events = events[
        events["event_type"].isin(MEANINGFUL_ACTIVITY)
    ]

    last_activity = (
        meaningful_events.groupby("customer_id")["timestamp"].max()
    )

    features["recency_days"] = (
        (as_of - last_activity).dt.total_seconds() / 86400
    )
    features["has_meaningful_history"] = (
        features["recency_days"].notna().astype(int)
    )

    features["recency_days"] = features["recency_days"].fillna(0.0)

    # Sessions are used for the frequency features below.
    sessions = events[events["event_type"] == "session"]

    # Frequency
    start_30d = as_of - pd.Timedelta(days=30)
    start_90d = as_of - pd.Timedelta(days=90)

    sessions_30d = sessions[
        (sessions["timestamp"] > start_30d)
        & (sessions["timestamp"] <= as_of)
    ]

    sessions_90d = sessions[
        (sessions["timestamp"] > start_90d)
        & (sessions["timestamp"] <= as_of)
    ]

    features["sessions_30d"] = (
        sessions_30d.groupby("customer_id").size()
    )

    features["sessions_90d"] = (
        sessions_90d.groupby("customer_id").size()
    )

    # Monetary
    purchases = events[
        (events["event_type"] == "purchase")
        & (events["timestamp"] > start_90d)
        & (events["timestamp"] <= as_of)
    ].copy()

    purchases["amount_usd"] = purchases["properties"].apply(
        lambda properties: float(properties.get("amount_usd", 0.0))
    )

    features["purchase_count_90d"] = (
        purchases.groupby("customer_id").size()
    )

    features["revenue_90d"] = (
        purchases.groupby("customer_id")["amount_usd"].sum()
    )

    # Customers with no events of a particular type should receive 0.
    count_columns = [
        "sessions_30d",
        "sessions_90d",
        "purchase_count_90d",
    ]

    features[count_columns] = (
        features[count_columns]
        .fillna(0)
        .astype(int)
    )

    features["revenue_90d"] = (
        features["revenue_90d"]
        .fillna(0.0)
        .astype(float)
        .round(2)
    )

    return features.reset_index()