import numpy as np
import pandas as pd


AS_OF_TIMESTAMP = pd.Timestamp("2024-06-01T12:00:00Z")

MEANINGFUL_ACTIVITY = {
    "session",
    "purchase",
    "push_open",
    "in_app_event",
    "campaign_click",
    "support_ticket",
}


def generate_synthetic_events(
    events: pd.DataFrame,
    customer_count: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    events = events.copy().sort_values(["customer_id", "timestamp"])

    event_counts = (
        events.groupby("customer_id")
        .size()
        .to_numpy()
    )

    event_type_probabilities = (
        events["event_type"]
        .value_counts(normalize=True)
        .sort_index()
    )

    event_types = event_type_probabilities.index.to_numpy()
    event_probabilities = event_type_probabilities.to_numpy()

    gaps = (
        events.groupby("customer_id")["timestamp"]
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(86400)
        .to_numpy()
    )

    session_durations = (
        events.loc[events["event_type"] == "session", "properties"]
        .apply(lambda properties: properties["duration_sec"])
        .to_numpy()
    )

    purchase_amounts = (
        events.loc[events["event_type"] == "purchase", "properties"]
        .apply(lambda properties: properties["amount_usd"])
        .to_numpy()
    )

    meaningful_events = events[
        events["event_type"].isin(MEANINGFUL_ACTIVITY)
    ]

    real_last_activity = (
        meaningful_events.groupby("customer_id")["timestamp"]
        .max()
    )

    real_recency_days = (
        (AS_OF_TIMESTAMP - real_last_activity)
        .dt.total_seconds()
        .div(86400)
        .to_numpy()
    )

    min_timestamp = events["timestamp"].min()

    rows: list[dict] = []

    for customer_number in range(1, customer_count + 1):
        customer_id = f"syn_cust_{customer_number:05d}"

        event_count = int(rng.choice(event_counts))

        customer_event_types = rng.choice(
            event_types,
            size=event_count,
            p=event_probabilities,
        )

        gap_count = max(event_count - 1, 0)

        customer_gaps = rng.choice(
            gaps,
            size=gap_count,
        ).astype(float)

        relative_days = np.concatenate(
            (
                np.array([0.0]),
                np.cumsum(customer_gaps),
            )
        )

        meaningful_indexes = [
            index
            for index, event_type in enumerate(customer_event_types)
            if event_type in MEANINGFUL_ACTIVITY
        ]

        if meaningful_indexes:
            last_meaningful_index = meaningful_indexes[-1]
        else:
            # Very unlikely, but ensures every customer can be positioned.
            last_meaningful_index = event_count - 1

        target_recency_days = float(
            rng.choice(real_recency_days)
        )

        target_last_activity = (
            AS_OF_TIMESTAMP
            - pd.Timedelta(days=target_recency_days)
        )

        start_timestamp = (
            target_last_activity
            - pd.Timedelta(
                days=float(relative_days[last_meaningful_index])
            )
        )

        # If the generated history would begin before the source data,
        # resample the gaps until it fits.
        while start_timestamp < min_timestamp:
            customer_gaps = rng.choice(
                gaps,
                size=gap_count,
            ).astype(float)

            relative_days = np.concatenate(
                (
                    np.array([0.0]),
                    np.cumsum(customer_gaps),
                )
            )

            start_timestamp = (
                target_last_activity
                - pd.Timedelta(
                    days=float(relative_days[last_meaningful_index])
                )
            )

        for event_index, event_type in enumerate(
            customer_event_types
        ):
            timestamp = (
                start_timestamp
                + pd.Timedelta(
                    days=float(relative_days[event_index])
                )
            )

            properties: dict = {}

            if event_type == "session":
                properties["duration_sec"] = max(
                    1,
                    int(rng.choice(session_durations)),
                )

            elif event_type == "purchase":
                properties["amount_usd"] = round(
                    float(rng.choice(purchase_amounts)),
                    2,
                )

            elif event_type in {
                "push_sent",
                "push_open",
                "campaign_click",
            }:
                properties["campaign_id"] = (
                    f"camp_{rng.integers(1, 21):02d}"
                )

            elif event_type == "in_app_event":
                properties["event_name"] = (
                    f"event_{rng.integers(1, 11):02d}"
                )

            elif event_type == "support_ticket":
                properties["category"] = rng.choice(
                    ["billing", "bug", "account", "other"]
                )

            rows.append(
                {
                    "event_id": f"syn_evt_{len(rows) + 1:08d}",
                    "customer_id": customer_id,
                    "event_type": event_type,
                    "timestamp": timestamp,
                    "properties": properties,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["customer_id", "timestamp"])
        .reset_index(drop=True)
    )